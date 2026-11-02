import math
from typing import Optional, Callable, Iterable

import torch


class AdamWOptimizer(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[Callable]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                m = state["m"] if "m" in state else torch.zeros_like(p)
                v = state["v"] if "v" in state else torch.zeros_like(p)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                lr_t = lr * (math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t))  # Compute adjusted 𝛼 for iteration 𝑡
                p.data -= lr * weight_decay * p.data  # Apply weight decay
                m = betas[0] * m + (1 - betas[0]) * grad  # Update the first moment estimate
                v = betas[1] * v + (1 - betas[1]) * grad ** 2  # Update the second moment estimate
                p.data -= lr_t * m / (v.sqrt() + eps)  # Apply moment-adjusted weight updates
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss
