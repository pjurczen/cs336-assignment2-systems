import math

import torch
from jaxtyping import Bool, Float

from cs336_basics.linear import Linear
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... keys d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None
) -> Float[torch.Tensor, " ... seq_len d_v"]:
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)  # => [FLOPs] (b, n, d) @ (b, n, d) = 2bn²d
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-Inf"))
    qk = softmax(qk, dim=-1)
    return einsum(qk, V, "... queries keys, ... keys d_v -> ... queries d_v")  # => [FLOPs] (b, n, n) @ (b, n, d) = 2bn²d


import torch
from einops import einsum, rearrange


class CasualMultiHeadSelfAttention(torch.nn.Module):
    device: torch.device | None
    d_k: int
    d_v: int
    d_model: int
    num_heads: int
    q_proj: Linear  # (num_heads * d_k, d_model) = (hd_k, d_model)
    k_proj: Linear  # (num_heads * d_k, d_model) = (hd_k, d_model)
    v_proj: Linear  # (num_heads * d_v, d_model) = (hd_v, d_model)
    output_proj: Linear  # (d_model, num_heads * d_v) = (d_model, hd_v)
    rope: RotaryPositionalEmbedding | None

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise Exception("d_model must be divisible by num_heads!")
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = round(d_model / num_heads)
        self.d_v = self.d_k
        self.q_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    # x (..., seq_len, d_model)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        casual_mask = (torch.tril(torch.ones(seq_len, seq_len, device=self.device)) == 1)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=self.device)
        qkv_proj = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=-2)
        w_x = einsum(x, qkv_proj, "... seq_len d_model, d_qkv d_model -> ... seq_len d_qkv")
        # (..., seq_len, hd_k + hd_k + hd_v) => [FLOPs] (b, n, d) @ (d, 3d) = 6bnd²
        wq_x = w_x[..., :self.num_heads * self.d_k]
        wk_x = w_x[..., self.num_heads * self.d_k:2 * self.num_heads * self.d_k]
        wv_x = w_x[..., 2 * self.num_heads * self.d_k:]
        wq_x_i = rearrange(wq_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        wk_x_i = rearrange(wk_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        wv_x_i = rearrange(wv_x, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)  # (..., h, seq_len, d_v)
        if self.rope:
            wq_x_i = self.rope(wq_x_i, token_positions)
            wk_x_i = self.rope(wk_x_i, token_positions)
        result = scaled_dot_product_attention(wq_x_i, wk_x_i, wv_x_i, casual_mask)  # (..., h, seq_len, d_v)
        result = rearrange(result, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(result)


class CasualMultiHeadSelfAttentionOptimized(torch.nn.Module):
    device: torch.device | None
    d_k: int
    d_v: int
    d_model: int
    num_heads: int
    qkv_proj: Linear  # (hd_k + hd_k + hd_v, d_model)
    q_norm: RMSNorm
    k_norm: RMSNorm
    qk_norm: bool
    output_proj: Linear  # (d_model, num_heads * d_v) = (d_model, hd_v)
    rope: RotaryPositionalEmbedding | None

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, qk_norm: bool = False, eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise Exception("d_model must be divisible by num_heads!")
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = round(d_model / num_heads)
        self.d_v = self.d_k
        self.qkv_proj = Linear(d_model, num_heads * self.d_k + num_heads * self.d_k + num_heads * self.d_v, device=device, dtype=dtype)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.d_k, eps, device=device, dtype=dtype)
            self.k_norm = RMSNorm(self.d_k, eps, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    # x (..., seq_len, d_model)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        casual_mask = (torch.tril(torch.ones(seq_len, seq_len, device=self.device)) == 1)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=self.device)
        w_x = self.qkv_proj(x)  # (..., seq_len, hd_k + hd_k + hd_v) => [FLOPs] (b, n, d) @ (d, 3d) = 6bnd²
        wq_x = w_x[..., :self.num_heads * self.d_k]
        wk_x = w_x[..., self.num_heads * self.d_k:2 * self.num_heads * self.d_k]
        wv_x = w_x[..., 2 * self.num_heads * self.d_k:]
        wq_x_i = rearrange(wq_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        wk_x_i = rearrange(wk_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        if self.qk_norm:
            wq_x_i = self.q_norm(wq_x_i)
            wk_x_i = self.k_norm(wk_x_i)
        wv_x_i = rearrange(wv_x, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)  # (..., h, seq_len, d_v)
        if self.rope:
            wq_x_i = self.rope(wq_x_i, token_positions)
            wk_x_i = self.rope(wk_x_i, token_positions)
        result = scaled_dot_product_attention(wq_x_i, wk_x_i, wv_x_i, casual_mask)  # (..., h, seq_len, d_v) => [FLOPs] 4bn²d
        result = rearrange(result, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(result)  # => [FLOPs] (b, n, d) @ (d, d) = 2bnd²
