import torch
import math


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_factor: float = 0.,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of warmup steps.
            total_steps (int): Total number of steps (including warmup).
            min_factor (float): Minimum learning rate as a factor of the base learning rate (e.g., 0 for 0% or 0.01 for 1%).
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_factor = min_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch

        if current_step < self.warmup_steps:
            # Linearly increase the learning rate during warmup
            warmup_factor = current_step / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            decay_steps = current_step - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps
            cosine_decay = 0.5 * (
                1 + math.cos(math.pi * decay_steps / total_decay_steps)
            )
            return [
                base_lr * (self.min_factor + (1 - self.min_factor) * cosine_decay)
                for base_lr in self.base_lrs
            ]
