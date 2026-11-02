import math
from typing import Optional, Callable, Iterable

import torch


def newtonSchulz(G: torch.Tensor, a: float = 3.4445, b: float = -4.7750, c: float = 2.0315) -> torch.Tensor:
    eps = 1e-7
    X = G.bfloat16()
    X = X / (X.pow(2).sum().sqrt() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


class MuonOptimizer(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, weight_decay: float = 0.01, momentum: float = 0.95) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[Callable]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                m = state["m"] if "m" in state else torch.zeros_like(p)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = momentum * m + grad  # Update the momentum
                u = grad + momentum * m
                o = newtonSchulz(u)
                n = lr * 0.2 * math.sqrt(max(p.shape))
                p.data -= lr * weight_decay * p.data  # Apply weight decay
                p.data -= n * o  # Apply momentum-adjusted weight updates
                state["m"] = m
        return loss
