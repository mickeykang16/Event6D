import torch
import math

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Per-step warmup + cosine decay.
    - warmup_steps: 선형 워밍업 스텝 수
    - total_steps : 전체 스텝 수 (워밍업 포함)
    - min_lr_ratio: 최저 lr = base_lr * min_lr_ratio
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = int(max(0, warmup_steps))
        self.total_steps = int(max(1, total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # scheduler.step() 호출 횟수
        if step <= self.warmup_steps and self.warmup_steps > 0:
            w = step / max(1, self.warmup_steps)
            return [base_lr * w for base_lr in self.base_lrs]

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)
            for base_lr in self.base_lrs
        ]
        
        
def get_scheduler(sched_cfg,
                  steps_per_epoch,
                  num_epochs,
                  optimizer,
                  lr):
    total_training_steps = steps_per_epoch * num_epochs

    if sched_cfg is not None:
        sched_type = getattr(sched_cfg, 'type', 'cosine_warmup')
        min_lr_ratio = float(getattr(sched_cfg, 'min_lr_ratio', 0.01))

        if sched_type == 'cosine_warmup':
            warm = getattr(sched_cfg, 'warmup', None)
            warmup_steps = 0
            if warm is not None:
                # steps가 있으면 우선, 없으면 ratio 기반
                steps_val = getattr(warm, 'steps', None)
                ratio_val = getattr(warm, 'ratio', None)
                if steps_val is not None:
                    warmup_steps = int(steps_val)
                elif ratio_val is not None:
                    warmup_steps = int(total_training_steps * float(ratio_val))
            scheduler = WarmupCosineLR(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_training_steps,
                min_lr_ratio=min_lr_ratio
            )
            scheduler_per_step = True

        elif sched_type == 'cosine_epoch':
            # epoch 단위 코사인. eta_min = base_lr * min_lr_ratio
            eta_min = lr * min_lr_ratio
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=eta_min
            )
            scheduler_per_step = False
        else:
            scheduler = None
            scheduler_per_step = False
            
    return scheduler, scheduler_per_step