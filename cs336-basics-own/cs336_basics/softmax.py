import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_norm = x.subtract(x_max)
    x_norm_exp = x_norm.exp()
    return x_norm_exp / x_norm_exp.sum(dim=dim, keepdim=True)
