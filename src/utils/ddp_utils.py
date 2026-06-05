# ddp_safe_step.py
# -*- coding: utf-8 -*-
"""
DDP 학습 중 NaN/Inf 등 invalid loss/grad가 발생했을 때
모든 rank에서 '동일하게' 스텝을 스킵하여 hang을 방지하는 유틸.

업데이트: loss 값을 메인 루프에서 바로 쓰도록 반환합니다.
- `step(...) -> (stepped: bool, loss_value: float)`
- `loss_value`는 기본적으로 rank0의 로컬 loss를 float으로 반환
- 옵션으로 전-rank 평균/합산값을 반환하도록 설정 가능(return_loss='local'|'mean'|'sum')

사용법(학습 루프 내):
    stepper = DdpSafeStepper(ddp_model, optimizer, scaler, device,
                             max_grad_norm=1.0, return_loss='local')
    ...
    stepped, loss_val = stepper.step(loss, epoch, step_idx)
    if stepped and math.isfinite(loss_val):
        total_loss += loss_val
        num_batches += 1
    logger.log_batch(loss=loss_val, lr=..., epoch=epoch, batch_idx=step_idx, iterator=iterator, vis_image=...)
"""
from __future__ import annotations
from typing import Optional, Literal
import math
import torch
import torch.distributed as dist

# ---------- DDP helpers ----------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world() -> int:
    return dist.get_world_size() if is_dist() else 1

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

# ---------- finite checks ----------
@torch.no_grad()
def all_ranks_finite_loss(loss: Optional[torch.Tensor], device: torch.device) -> bool:
    ok = torch.tensor([1], device=device, dtype=torch.int32)
    if (loss is None) or (not torch.is_tensor(loss)) or (not torch.isfinite(loss).all()):
        ok.fill_(0)
    if is_dist():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())

@torch.no_grad()
def all_ranks_finite_grads(model: torch.nn.Module, device: torch.device) -> bool:
    ok = torch.tensor([1], device=device, dtype=torch.int32)
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            ok.fill_(0)
            break
    if is_dist():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())

# ---------- main stepper ----------
class DdpSafeStepper:
    """
    DDP 안전 스텝 유틸리티.
    - loss 유효성(all-reduce) → backward → unscale/clip → grad 유효성(all-reduce)
      → (유효) step & update / (무효) 전-rank 스킵
    - 메인 루프 로깅을 위해 loss scalar를 반환
    """
    def __init__(
        self,
        model: torch.nn.Module,                # DDP로 래핑된 모델 권장
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        device: torch.device,
        max_grad_norm: Optional[float] = 1.0,
        barrier_on_skip: bool = True,
        quiet: bool = False,
        return_loss: Literal['local', 'mean', 'sum'] = 'local',
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.barrier_on_skip = barrier_on_skip
        self.quiet = quiet
        self.return_loss = return_loss

    def _log_rank0(self, msg: str):
        if (not self.quiet) and get_rank() == 0:
            print(msg)

    @torch.no_grad()
    def _loss_scalar_for_log(self, loss: torch.Tensor) -> float:
        """설정에 따라 로컬/전역(mean/sum) loss 스칼라 반환."""
        try:
            if self.return_loss == 'local' or not is_dist():
                return float(loss.detach().to(dtype=torch.float32, device='cpu').item())
            # 전역 합산/평균
            t = loss.detach().to(device=self.device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if self.return_loss == 'mean':
                t = t / get_world()
            return float(t.item())
        except Exception:
            return float('nan')

    def step(self, loss: Optional[torch.Tensor], epoch: int, step_idx: int) -> tuple[bool, float]:
        """
        Returns:
            (stepped, loss_value)
            - stepped: optimizer.step() 수행 여부
            - loss_value: 로깅용 스칼라(설정에 따라 local/mean/sum). invalid이면 NaN.
        """
        # 1) loss 유효성(전-rank 동기화)
        if not all_ranks_finite_loss(loss, self.device):
            self._log_rank0(f"[epoch {epoch} step {step_idx}] invalid loss → skip step (all ranks)")
            self.optimizer.zero_grad(set_to_none=True)
            # if self.scaler is not None:
                # self.scaler.update()
            if self.barrier_on_skip and is_dist():
                dist.barrier()
            return False, float('nan')

        # 로깅용 스칼라(여기서 계산하면 backward 이전/이후 동일)
        loss_value = self._loss_scalar_for_log(loss) if torch.is_tensor(loss) else float('nan')

        # 2) backward (+ AMP)
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        # 3) grad clipping
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        # 4) grad 유효성(전-rank 동기화)
        if not all_ranks_finite_grads(self.model, self.device):
            self._log_rank0(f"[epoch {epoch} step {step_idx}] non-finite grad → skip step (all ranks)")
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.update()
            if self.barrier_on_skip and is_dist():
                dist.barrier()
            return False, float('nan')

        # 5) optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return True, loss_value
