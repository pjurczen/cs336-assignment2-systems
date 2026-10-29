from typing import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    l2_norm = torch.stack([p.grad.pow(2).sum() for p in params]).sum().sqrt()
    scaling_factor = (max_l2_norm / (l2_norm + eps)).clamp(max=1.0)
    for param in params:
        param.grad = param.grad.mul_(scaling_factor)
    return l2_norm
