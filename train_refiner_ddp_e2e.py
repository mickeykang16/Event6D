#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP training for HO3D scorer (converted from your single-GPU/DP script).
- Launch with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 train_scorer_ddp.py \
      --name exp --video_dirs /data/dataset/ho3d --config cfg.yaml --workers 8 --amp
- Rank0-only: wandb/tqdm/visualization/checkpoints/dir-creation
- Uses DistributedSampler; shuffle handled by sampler; set_epoch(epoch) applied
- Wraps ONLY pose_scorer.model in DDP (refiner stays eval-only on each rank)
- All-reduce to compute global average loss per epoch
- Loads DP/DDP/vanilla checkpoints (auto strip 'module.' keys)
"""

from __future__ import annotations
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import math
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from src.utils.logging import DdpLogger
from src.utils.ddp_utils import DdpSafeStepper
from tqdm import tqdm
import wandb
import nvdiffrast.torch as dr
from Utils import make_mesh_tensors_minimal

# --- project imports (as in your script) ---
from src.dataloader.collate_fn import custom_collate_fn
from src.refiner.foundationpose.training.predict_pose_refine import PoseRefinePredictor
from loss import loss_refine
from src.utils.basics import load_yaml_cfg
from src.utils.geometry import augment_pose, depth2xyzmap_batch
from src.utils.batchsampler import DistributedGroupBatchSampler
from src.dataloader import get_dataset
from src.utils.scheduler import get_scheduler
from einops import rearrange
import torch.nn.functional as F
import lpips
import sys
from src.depth_extrapolation import models as evfi_models
from src.e2vid.utils.loading_utils import load_model

from timers import TimerDummy as CudaTimer
# ---------------- DDP helpers ----------------

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world() -> int:
    return dist.get_world_size() if is_dist() else 1


def ddp_init(backend: str = "nccl") -> torch.device:
    if is_dist():
        # already initialized (e.g., if called twice)
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")


def ddp_cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int):
    seed = seed + get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------- utils ----------------

def _strip_module_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    first = next(iter(state_dict))
    if first.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ---------------- main training ----------------

class _Tee:
    """stdout/stderr를 터미널과 파일 양쪽에 동시 출력."""
    def __init__(self, stream, filepath):
        self._stream = stream
        self._file = open(filepath, 'a', buffering=1)
    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def fileno(self):          # subprocess 호환
        return self._stream.fileno()
    def close(self):
        self._file.close()


def tracking_ddp(args):
    device = ddp_init()
    set_seed(42)

    # ----- Run naming & dirs (rank0 creates) -----
    now = datetime.now(); timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{timestamp}-{args.name}"
    default_root = './outputs'
    video_dirs = args.video_dirs

    save_results_est_path = f'{default_root}/debug' if args.debug else f'{default_root}/{name}'
    save_vis_path = f'{save_results_est_path}/vis'
    save_checkpoint_path = f'{save_results_est_path}/model'

    if get_rank() == 0:
        os.makedirs(save_results_est_path, exist_ok=True)
        os.makedirs(save_vis_path, exist_ok=True)
        os.makedirs(save_checkpoint_path, exist_ok=True)
        print(save_results_est_path)
        log_path = os.path.join(save_results_est_path, f'train_rank{get_rank()}.log')
        sys.stdout = _Tee(sys.stdout, log_path)
    if is_dist():
        dist.barrier()  # ensure dirs exist

    # ----- config -----
    cfg = load_yaml_cfg(args.config) if args.config else argparse.Namespace()
    num_epochs = int(cfg.train.epochs)

    logger = DdpLogger(
        debug=args.debug,
        save_vis_path=save_vis_path,
        save_checkpoint_path=save_checkpoint_path,
    )
    logger.start_wandb(
        project=cfg.dataset.name + '-' + args.wandb_project,
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

    # ----- dataset & loader -----
    # dataset = HO3DDataset_v2(root_dir=video_dirs, split="train", window=1, stride=2, return_mesh=False)
    dataset = get_dataset(**cfg.dataset)
    
    sampler = DistributedGroupBatchSampler(
        group_ids=dataset.obj_ids,
        local_batch_size=cfg.train.batch_size,   # 각 GPU의 배치 4
        drop_last=False,       # 고정 배치 크기 보장
        pad_to_full=True,    # 필요하면 True + drop_last=False로
        shuffle=True,
        seed=123,
    )
    
    dataloader = DataLoader(
        dataset,
        num_workers=args.workers,
        collate_fn=custom_collate_fn,
        pin_memory=False,
        persistent_workers=(args.workers > 0),
        batch_sampler=sampler,
        # prefetch_factor=1
    )

    cc = torch.cuda.get_device_capability()
    if cc[0] < 8:  # Ampere 미만이면 Flash-SDP 금지
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    
    # ----- models -----
    pose_refiner = PoseRefinePredictor(load_pretrained=None, for_ev=cfg.model.get('for_ev', False))
    # pose_refiner.model = pose_refiner.model.to(device)
    model = pose_refiner.model

    # optional resume before wrapping
    if args.load_from is not None:
        assert os.path.isfile(args.load_from), f"Checkpoint not found: {args.load_from}"
        if get_rank() == 0:
            print(f"Load checkpoint from {args.load_from}!")
        state_dict = torch.load(args.load_from, map_location='cpu')
        state_dict = _strip_module_keys(state_dict)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(state_dict, strict=False)

    ddp_model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, output_device=device.index, find_unused_parameters=False)
    pose_refiner.model = ddp_model  # ensure methods use the wrapped module

    # --- E2VID (encoder frozen, decoder trainable) ---
    _e2vid_raw = torch.load(args.e2vid_ckpt, map_location='cpu')
    _e2vid_meta = {'arch': _e2vid_raw['arch'], 'model': _e2vid_raw['model']}
    e2vid_front_end, _ = load_model(args.e2vid_ckpt)
    e2vid_front_end = e2vid_front_end.to(device).eval()
    for p in e2vid_front_end.parameters():
        p.requires_grad_(False)
    _unet = e2vid_front_end.unetrecurrent
    for p in _unet.decoders.parameters():
        p.requires_grad_(True)
    for p in _unet.pred.parameters():
        p.requires_grad_(True)
    _unet.decoders.train()
    _unet.pred.train()
    e2vid_ddp = DDP(e2vid_front_end, device_ids=[device.index] if device.type == 'cuda' else None,
                    output_device=device.index, find_unused_parameters=True)

    # --- LPIPS loss (frozen VGG) ---
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # --- EventVFI (trainable) ---
    evfi_spec = torch.load(args.depth_extrapolation_ckpt, map_location='cpu')['model']
    evfi_meta = {k: v for k, v in evfi_spec.items() if k != 'sd'}  # name, args 등 보존
    eventvfi_model = evfi_models.make(evfi_spec, load_sd=True).to(device)
    eventvfi_ddp = DDP(eventvfi_model, device_ids=[device.index] if device.type == 'cuda' else None,
                       output_device=device.index, find_unused_parameters=True)

    # ----- optim/amp/loss/render ctx -----
    train_lr = float(cfg.train.lr)
    optimizer = optim.Adam(
        list(ddp_model.parameters()) + list(eventvfi_ddp.parameters()) +
        [p for p in e2vid_front_end.parameters() if p.requires_grad],
        lr=train_lr, weight_decay=pose_refiner.cfg['weight_decay']
    )
    scaler = GradScaler(enabled=args.amp)
    glctx = dr.RasterizeCudaContext(device=device.index if device.type == 'cuda' else None)

    # import pdb; pdb.set_trace()
    scheduler, scheduler_per_step = get_scheduler(getattr(cfg.train, 'scheduler', None),
                                                  len(dataloader),
                                                  num_epochs=num_epochs,
                                                  optimizer=optimizer,
                                                  lr=train_lr)
    
    combined_for_stepper = nn.ModuleList([ddp_model, eventvfi_ddp, e2vid_ddp])
    stepper = DdpSafeStepper(
        model=combined_for_stepper,
        optimizer=optimizer,
        scaler=scaler,          # AMP 안 쓰면 None
        device=device,
        max_grad_norm=1.0,
        return_loss='local',    # 'mean'으로 바꾸면 전-rank 평균 반환
    )
    
    best_loss = float('inf')
    lambda_recon = getattr(cfg.model.loss, 'recon_weight', 0.1)

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        ddp_model.train()
        eventvfi_ddp.train()
        total_loss = 0.0
        num_batches = 0

        iterator = dataloader if get_rank() != 0 else tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # import pdb; pdb.set_trace()
        for batch_idx, batch in enumerate(iterator):
            last = (batch_idx == len(dataloader) - 1)
            vis_interval = getattr(args, 'vis_interval', 100)
            do_vis = (batch_idx % vis_interval == 0) or last
            # print(batch['objMeshDm'].device)

            # --- prepare data ---
            with CudaTimer(device=device, timer_name="Prepare"):
                color = batch['color'].to(device, non_blocking=True)         # (B, 2, 3, H, W)
                recon_event = batch['recon_event'].to(device, non_blocking=True)  # (B, 2, 10, H, W)
                depth_event = batch['depth_event'].to(device, non_blocking=True)  # (B, 2, 10, H, W)
                depth = batch['depth'].to(device, non_blocking=True)          # (B, 2, H, W)
                object_pose = batch['objPose'].to(device, non_blocking=True)  # (B, 2, 4, 4)
                K = batch['camMat'][:, -1].to(device, non_blocking=True)      # (B, 3, 3)

                mesh = batch['objMesh']
                mesh_diameter = batch['objMeshDm'].to(device, non_blocking=True)

                if len(set(batch['objMeshID'])) == 1:
                    mesh = None
                    mesh_diameter = mesh_diameter[0]
                    mesh_diameter_ = mesh_diameter/2 if cfg.model.augment.trans == 'from_mesh' else cfg.model.augment.trans
                    mesh_tensors = batch['objMeshTensor'][0]
                    for k, v in mesh_tensors.items():
                        mesh_tensors[k] = v.to(device)
                else:
                    import pdb; pdb.set_trace()
                    raise NotImplementedError

                # --- E2VID forward (encoder frozen, decoder trainable) ---
                recon_ev = recon_event  # (B, 10, H, W)
                nonzero = (recon_ev != 0)
                num_nz = nonzero.float().sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
                ev_mean = (recon_ev * nonzero.float()).sum(dim=(1, 2, 3), keepdim=True) / num_nz
                ev_std = (((recon_ev ** 2) * nonzero.float()).sum(dim=(1, 2, 3), keepdim=True) / num_nz
                          - ev_mean ** 2 + 1e-9).sqrt()
                recon_ev_norm = nonzero.float() * (recon_ev - ev_mean) / ev_std
                recon_img, _, latent = e2vid_ddp(recon_ev_norm, None)
                # recon_img: (B, 1, H, W) float, sigmoid [0,1]

                # --- EventVFI forward (trainable) ---
                start_depth = depth[:, 0].unsqueeze(1) - 1.0   # (B, 1, H, W)
                depth_ev = depth_event.float()                   # (B, 10, H, W)
                depth_pred, _ = eventvfi_ddp(x=start_depth, event=depth_ev, latent=latent)
                # depth_pred: (B, 1, H, W)

                # --- depth supervision loss ---
                depth_gt = depth[:, -1].unsqueeze(1)            # (B, 1, H, W)
                valid_mask = (depth_gt > 0) & torch.isfinite(depth_gt)
                depth_loss = F.l1_loss(depth_pred[valid_mask], depth_gt[valid_mask] - 1.0)

                # --- E2VID reconstruction loss (LPIPS) ---
                gt_gray = (0.299 * color[:, -1, 0:1] + 0.587 * color[:, -1, 1:2] + 0.114 * color[:, -1, 2:3]) / 255.0
                recon_3ch = recon_img.repeat(1, 3, 1, 1) * 2 - 1   # [0,1] -> [-1,1]
                gt_3ch    = gt_gray.repeat(1, 3, 1, 1) * 2 - 1
                recon_loss = lpips_fn(recon_3ch, gt_3ch).mean()

                # --- build B-side inputs for FP refiner ---
                depth_out = depth_pred.squeeze(1) + 1.0         # (B, H, W)
                xyz_map = depth2xyzmap_batch(depth_out, K, zfar=torch.inf)
                cam = color[:, -1]                               # (B, 3, H, W) GT end-frame color

                # --- pose augmentation ---
                aug_object_pose, delta_t_gt, delta_R_gt = augment_pose(
                    object_pose[:, -1], mesh_diameter=mesh_diameter_,
                    rot_range_deg=cfg.model.augment.rot
                )

                misc = {'amp': args.amp}
            
            with CudaTimer(device=device, timer_name="Main"):
                results = pose_refiner.refine_pose_with_render_and_model(
                    mesh,
                    mesh_tensors,
                    aug_object_pose,
                    K,
                    cam,
                    depth_out,
                    xyz_map,
                    glctx,
                    mesh_diameter,
                    vis=do_vis and (dist.get_rank() == 0), misc=misc
                )
            delta_t_pred = results['trans_delta']
            delta_R_pred = results['rot_mat_delta']
            
            cfg.model.loss['trans_range'] = mesh_diameter_
            
            with CudaTimer(device=device, timer_name="loss_and_backprop"):
                loss, loss_t, loss_R = loss_refine(delta_t_pred, delta_t_gt, delta_R_pred, delta_R_gt, loss_cfg=cfg.model.loss)
                loss = loss + args.lambda_depth * depth_loss + lambda_recon * recon_loss

                stepped, loss_val = stepper.step(loss, epoch, batch_idx)

            if scheduler is not None and scheduler_per_step and stepped:
                scheduler.step()
            
            logger.log_batch(loss=float(loss_val) if np.isfinite(loss_val) else float('nan'),
                            lr=float(optimizer.param_groups[0]['lr']),
                            epoch=epoch, batch_idx=batch_idx,
                            iterator=iterator,
                            vis_image=results.get('vis', None) if isinstance(results, dict) else None,
                            sub_loss=dict(
                            loss_t=loss_t.item(),
                            loss_r=loss_R.item(),
                            loss_depth=depth_loss.item(),
                            loss_recon=recon_loss.item(),
                            loss_total=float(loss_val) if np.isfinite(loss_val) else float('nan'))
                            )
            
            # --- wandb logging (rank0 only, single log call per batch) ---
            if get_rank() == 0 and logger._using_wandb and logger._wandb is not None:
                wandb_dict = {
                    "loss": float(loss_val) if np.isfinite(loss_val) else float('nan'),
                    "lr": float(optimizer.param_groups[0]['lr']),
                    "loss_t": loss_t.item(),
                    "loss_r": loss_R.item(),
                    "loss_depth": depth_loss.item(),
                    "loss_recon": recon_loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                }
                if do_vis:
                    def _to_colormap(depth_tensor):
                        arr = depth_tensor.detach().float().cpu().numpy()
                        arr = arr - arr.min()
                        denom = arr.max()
                        if denom > 1e-6:
                            arr = arr / denom
                        arr = (arr * 255).clip(0, 255).astype(np.uint8)
                        return cv2.applyColorMap(arr, cv2.COLORMAP_MAGMA)

                    def _to_gray255(img_tensor):
                        arr = img_tensor.squeeze().detach().float().cpu().numpy()
                        arr = arr - arr.min()
                        denom = arr.max()
                        if denom > 1e-6:
                            arr = arr / denom
                        return (arr * 255).clip(0, 255).astype(np.uint8)

                    e2vid_gray = _to_gray255(recon_img[0])
                    wandb_dict["e2vid_recon"] = logger._wandb.Image(e2vid_gray, caption="E2VID recon")
                    depth_pred_vis = _to_colormap(depth_pred[0, 0])
                    depth_gt_vis   = _to_colormap(depth_gt[0, 0])
                    side_by_side = cv2.hconcat([depth_pred_vis, depth_gt_vis])
                    wandb_dict["depth_pred_vs_gt"] = logger._wandb.Image(
                        side_by_side[..., ::-1], caption="Left: EventVFI pred, Right: GT"
                    )
                logger._wandb.log(wandb_dict)

            # --- iter checkpoint (every 100 batches, rank0 only) ---
            if get_rank() == 0 and stepped and (batch_idx + 1) % 100 == 0:
                stem = f'iter_e{epoch:04d}_b{batch_idx+1:06d}'
                evfi_ckpt   = {'model': {**evfi_meta, 'sd': getattr(eventvfi_ddp, 'module', eventvfi_ddp).state_dict()}, 'epoch': epoch}
                torch.save(evfi_ckpt,   os.path.join(save_checkpoint_path, f'{stem}_depth_extrapolation.pth'))

            if math.isfinite(loss_val):
                if stepped:            # 유효 스텝만 평균에 반영
                    total_loss += loss_val
                    num_batches += 1

            # torch.cuda.empty_cache()

        avg_loss = logger.reduce_epoch_avg_loss(total_loss, num_batches, device)
        best_loss = logger.log_epoch_end(avg_loss=avg_loss, epoch=epoch, num_epochs=num_epochs,
        ddp_model=None, best_loss=best_loss)

        # --- 에폭 체크포인트 저장 (rank0 only) ---
        if get_rank() == 0 and save_checkpoint_path is not None:
            evfi_ckpt   = {'model': {**evfi_meta, 'sd': getattr(eventvfi_ddp, 'module', eventvfi_ddp).state_dict()}, 'epoch': epoch}
            stem = f'epoch_{epoch:04d}'
            torch.save(evfi_ckpt,   os.path.join(save_checkpoint_path, f'{stem}_depth_extrapolation.pth'))
            if avg_loss <= best_loss:
                torch.save(evfi_ckpt,   os.path.join(save_checkpoint_path, 'best_depth_extrapolation.pth'))
        
        if scheduler is not None and not scheduler_per_step:
            scheduler.step()
        
        if hasattr(iterator, 'close'):
            iterator.close()
        
        if is_dist():
            dist.barrier()

    # if not args.debug and get_rank() == 0:
        # wandb.finish()
    logger.finish()

    ddp_cleanup()


# ---------------- CLI ----------------

def build_argparser():
    parser = argparse.ArgumentParser(description='HO3D scorer DDP training')
    parser.add_argument('--name', type=str, default="debug")
    parser.add_argument('--video_dirs', type=str, default="/data/dataset/ho3d")
    parser.add_argument('--json_path', type=str, default="/workspace/dataset/ycbv/models/models_info.json")
    parser.add_argument('--eval_folder_name', type=str, default='0_eval_metric')
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument("--init_gt_pose", action="store_true", default=True)
    parser.add_argument("--rgb_fp", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="foundationpose", choices=["foundationpose", "megapose", "track6d"])
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument('--iteration', type=int, default=2)
    parser.add_argument('--crop_ratio', type=float, default=1.2)
    parser.add_argument('--input_resize', type=int, nargs=2, default=(160, 160))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='FP-refiner')
    parser.add_argument("--multi_batch", action="store_true", default=False)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--e2vid_ckpt', type=str, required=True)
    parser.add_argument('--depth_extrapolation_ckpt', type=str, required=True)
    parser.add_argument('--lambda_depth', type=float, default=0.1)
    parser.add_argument('--vis_interval', type=int, default=100, help='log depth/e2vid vis to wandb every N batches')
    return parser


def main():
    logging.basicConfig(level=logging.WARNING)
    args = build_argparser().parse_args()
    tracking_ddp(args)


if __name__ == '__main__':
    main()
