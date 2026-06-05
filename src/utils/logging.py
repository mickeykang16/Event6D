import logging
import os
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
from PIL import Image
import numpy as np


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


class LevelsFilter(logging.Filter):
    def __init__(self, levels):
        self.levels = [getattr(logging, level) for level in levels]

    def filter(self, record):
        return record.levelno in self.levels


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        import tqdm

        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def start_disable_output(logfile):
    # Open the logfile for append
    with open(logfile, "a") as log_file:
        # Save the original stdout file descriptor
        original_stdout = os.dup(1)

        # Redirect stdout to the log file
        os.dup2(log_file.fileno(), 1)

        # Return the original stdout file descriptor
        return original_stdout


def stop_disable_output(original_stdout):
    # Restore the original stdout file descriptor
    os.dup2(original_stdout, 1)


def log_image(logger, name, path=None):
    if isinstance(logger, WandbLogger):
        assert isinstance(path, str), "image must be a path to an image"
        logger.experiment.log(
            {f"{name}": wandb.Image(path)},
        )
    elif isinstance(logger, TensorBoardLogger):
        image = Image.open(path)
        image = torch.tensor(np.array(image) / 255.0)
        image = image.permute(2, 0, 1).float()
        assert isinstance(image, torch.Tensor), "image must be a tensor"
        logger.experiment.add_image(
            f"{name}",
            image,
        )

# -*- coding: utf-8 -*-
"""
Rank-0 전용 DDP 로깅/체크포인트 유틸 (wandb + tqdm + 시각화 저장)

사용 방법(요약):

    from ddp_logger import DdpLogger

    logger = DdpLogger(
        debug=args.debug,
        save_vis_path=save_vis_path,
        save_checkpoint_path=save_checkpoint_path,
    )

    # rank0 only: wandb init
    logger.start_wandb(
        project=args.wandb_project,
        run_name=name,
        tags=[args.method, "pose_refinement", "ho3d"],
        config={
            "epochs": num_epochs,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.lr,
            "stride": args.stride,
            "init_gt_pose": args.init_gt_pose,
            "method": args.method,
            "iteration": args.iteration,
            "crop_ratio": args.crop_ratio,
            "input_resize": args.input_resize,
            "workers": args.workers,
            "amp": args.amp,
            "model_config": dict(cfg.model),
            "train_config": dict(cfg.train),
            "dataset_root": args.video_dirs,
        },
        save_code=True,
    )

    for epoch in range(num_epochs):
        ...
        for batch_idx, batch in enumerate(iterator):
            ...
            # 배치별 로그/시각화/진행바 업데이트
            logger.log_batch(
                loss=float(loss.detach().cpu().item()),
                lr=float(optimizer.param_groups[0]['lr']),
                epoch=epoch,
                batch_idx=batch_idx,
                iterator=iterator,
                vis_image=results.get('vis', None),   # HxWxC BGR (cv2.imwrite 호환)
            )

        # 에폭 단위 평균 loss(전 rank 합산) 계산
        avg_loss = logger.reduce_epoch_avg_loss(total_loss, num_batches, device)

        # 에폭 종료 로그/체크포인트 저장 (ddp_model은 DDP로 감싼 모델)
        best_loss = logger.log_epoch_end(
            avg_loss=avg_loss,
            epoch=epoch,
            num_epochs=num_epochs,
            ddp_model=ddp_model,
            best_loss=best_loss,
        )

    logger.finish()
"""
# from __future__ import annotations

from typing import Any, Dict, Optional
import os

import torch.distributed as dist

# 옵셔널 의존성들 (rank0에서만 실제 사용):
# - tqdm 타입 확인
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # noqa: N816
# - wandb는 rank0 + not debug일 때만 import

# ---------------- DDP helpers ----------------

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


# ---------------- Logger class ----------------

class DdpLogger:
    def __init__(
        self,
        debug: bool,
        save_vis_path: Optional[str] = None,
        save_checkpoint_path: Optional[str] = None,
        quiet: bool = False,
    ) -> None:
        self.debug = bool(debug)
        self.save_vis_path = save_vis_path
        self.save_checkpoint_path = save_checkpoint_path
        self.quiet = bool(quiet)
        self._wandb = None  # lazy import/handle
        self._using_wandb = False

        if get_rank() == 0:
            # 디렉토리 준비(rank0에서만)
            if self.save_vis_path:
                os.makedirs(self.save_vis_path, exist_ok=True)
            if self.save_checkpoint_path:
                os.makedirs(self.save_checkpoint_path, exist_ok=True)

    # --------- wandb ---------
    def start_wandb(
        self,
        project: str,
        run_name: str,
        tags: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        *,
        save_code: bool = True,
    ) -> None:
        """rank0 + not debug일 때만 wandb init. 다른 rank는 noop."""
        if self.debug or get_rank() != 0:
            return
        try:
            import wandb  # local import
            self._wandb = wandb
            self._wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags or [],
                save_code=save_code,
            )
            self._using_wandb = True
        except Exception as e:
            self._using_wandb = False
            if not self.quiet:
                print(f"[DdpLogger] wandb init failed: {e}")

    def log_batch(
        self,
        *,
        loss: float,
        lr: float,
        epoch: int,
        batch_idx: int,
        iterator: Optional[Any] = None,
        vis_image: Optional[Any] = None,  # OpenCV BGR image (HxWxC, uint8)
        sub_loss = None
        # loss_t = None,
        # loss_r = None,
        # loss_triplet = None,
        # loss_reg = None,
    ) -> None:
        """배치 단위로 진행바 갱신/시각화 저장/로그.
        - rank0 에서만 동작. debug=True면 wandb 전송 생략.
        - iterator가 tqdm일 경우 set_postfix로 진행바 업데이트.
        - vis_image가 있으면 파일 저장 + wandb.Image 로깅.
        """
        if get_rank() != 0:
            return

        # tqdm postfix
        if iterator is not None and hasattr(iterator, "set_postfix"):

            postfix = {}
            for loss_name, value in sub_loss.items():
                postfix.update({
                    loss_name: f"{value:.5f}",
                })
            postfix.update({
                "loss": f"{loss:.4f}",
                "lr": f"{lr:.4e}",
            })
            iterator.set_postfix(postfix)


        # 시각화 저장/로그
        if vis_image is not None and self.save_vis_path is not None:
            try:
                import cv2
                out_path = os.path.join(self.save_vis_path, f"epoch_{epoch:04d}_iteration_{batch_idx}.jpg")
                cv2.imwrite(out_path, vis_image)
                if self._using_wandb and not self.debug and self._wandb is not None:
                    self._wandb.log({
                        "visualization": self._wandb.Image(vis_image[..., ::-1], caption=f"Epoch {epoch}, Batch {batch_idx}"),
                        "epoch": epoch,
                        "batch": batch_idx,
                    })
            except Exception as e:
                if not self.quiet:
                    print(f"[DdpLogger] vis save/log failed: {e}")

    # --------- epoch reduce & log ---------
    def reduce_epoch_avg_loss(self, total_loss: float, num_batches: int, device: torch.device) -> float:
        """전 rank 합산으로 평균 loss 계산."""
        tl = torch.tensor([float(total_loss), float(num_batches)], device=device, dtype=torch.float64)
        if is_dist():
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
        global_total, global_batches = tl.tolist()
        if global_batches <= 0:
            return float("inf")
        return float(global_total / global_batches)

    def log_epoch_end(
        self,
        *,
        avg_loss: float,
        epoch: int,
        num_epochs: int,
        ddp_model: Optional[torch.nn.Module] = None,
        best_loss: Optional[float] = None,
    ) -> float:
        """rank0에서 에폭 종료 출력/체크포인트/로그. 새로운 best를 반환."""
        if get_rank() != 0:
            return float(best_loss if best_loss is not None else avg_loss)

        print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {avg_loss:.6f}")
        if self._using_wandb and not self.debug and self._wandb is not None:
            self._wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        # 체크포인트 저장
        new_best = best_loss if best_loss is not None else float("inf")
        if ddp_model is not None and self.save_checkpoint_path is not None:
            try:
                # best
                if avg_loss < new_best:
                    new_best = avg_loss
                    best_path = os.path.join(self.save_checkpoint_path, "best_model.pth")
                    torch.save(getattr(ddp_model, "module", ddp_model).state_dict(), best_path)
                    print(f"Best model saved to {best_path} with loss: {new_best:.6f}")
                    if self._using_wandb and not self.debug and self._wandb is not None:
                        self._wandb.log({"best_loss": new_best, "best_epoch": epoch + 1})
                        try:
                            self._wandb.save(best_path)
                        except Exception:
                            pass
                # epoch snapshot
                ep_path = os.path.join(self.save_checkpoint_path, f"epoch_{epoch:04d}.pth")
                torch.save(getattr(ddp_model, "module", ddp_model).state_dict(), ep_path)
                print(f"Model saved to {ep_path}")
            except Exception as e:
                if not self.quiet:
                    print(f"[DdpLogger] checkpoint save failed: {e}")

        return float(new_best)

    # --------- finalize ---------
    def finish(self) -> None:
        if self._using_wandb and not self.debug and self._wandb is not None and get_rank() == 0:
            try:
                self._wandb.finish()
            except Exception:
                pass
