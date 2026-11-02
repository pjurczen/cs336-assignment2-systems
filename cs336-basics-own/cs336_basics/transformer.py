import torch

from cs336_basics.attention import CasualMultiHeadSelfAttentionOptimized
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    ln1: RMSNorm
    attn: CasualMultiHeadSelfAttentionOptimized

    ffn: SwiGLU
    ln2: RMSNorm

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, eps: float = 1e-5, max_seq_len: int | None = None, theta: float | None = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None, qk_norm: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.attn = CasualMultiHeadSelfAttentionOptimized(d_model, num_heads, max_seq_len, theta, qk_norm=qk_norm, eps=eps, device=device,
                                                          dtype=dtype)  # => [FLOPs] 8bnd²
        self.ln2 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:  # => [FLOPS] 8 bnd² + 4bn²d + 6 bndd_ff
        y = x + self.attn(self.ln1(x), token_positions)  # => [FLOPs] 8 bnd² + 4bn²d = 8 × 1024 × 1600² + 4 × 1024² × 1600 = 27'682'406'400
        y = y + self.ffn(self.ln2(y))  # => [FLOPs] 16 bnd² / 6 bnd d_ff = 6 × 1024 × 1600 × 4288 = 42'152'755'200
        return y
