import torch

from cs336_basics.linear import Linear


class SwiGLU(torch.nn.Module):
    w1: Linear  # (d_ff, d_model)
    w2: Linear  # (d_model, d_ff)
    w3: Linear  # (d_ff, d_model)

    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_ff is None:
            d_ff = round(8 * d_model / 3 / 64) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    # x (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(
            x))  # => [FLOPs] (b, n, d) @ (8/3 d, d) + (b, n, d) @ (8/3 d, d) + (d, 8/3 d) @ (b, n, 8/3 d) = 16/3 bnd² + 16/3 bnd² + 16/3 bnd² = 16 bnd²
    # => [FLOPs] (b, n, d) @ (d_ff, d) + (b, n, d) @ (d_ff, d) + (d, d_ff) @ (b, n, d_ff) = 2bndd_ff + 2 bndd_ff + 2 bndd_ff = 6 bndd_ff


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
