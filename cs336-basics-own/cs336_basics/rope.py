import torch
from einops import rearrange


class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        k = torch.arange(0, d_k, 2, device=device)
        positions = torch.arange(0, max_seq_len, 1, device=device).reshape(max_seq_len, 1)
        angles = positions / theta ** (k / d_k)
        self.register_buffer('sin', torch.sin(angles), persistent=False)
        self.register_buffer('cos', torch.cos(angles), persistent=False)

    # x (batch, seq_len, d_k), token_positions (batch, seq_len)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... seq_len (half two) -> ... seq_len half two", two=2)
        a = x[..., 0]  # (..., seq, half)
        b = x[..., 1]  # (..., seq, half)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        a_out = a * cos - b * sin
        b_out = a * sin + b * cos
        x_out = torch.stack([a_out, b_out], dim=-1)  # (..., seq_len, d_k/2 2)
        x_out = rearrange(x_out, "... seq_len half two -> ... seq_len (half two)")
        return x_out
