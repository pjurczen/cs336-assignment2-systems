from math import sqrt

import torch

from cs336_basics.embedding import Embedding, EmbeddingLookup
from cs336_basics.linear import Linear
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.transformer import TransformerBlock


class TransformerLM(torch.nn.Module):
    token_embeddings: EmbeddingLookup
    layers: torch.nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear
    logit_soft_cap: float | None

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int | None = None, eps: float = 1e-5,
                 theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, tie_embeddings: bool = False,
                 qk_norm: bool = False, logit_soft_cap: float | None = None):
        super().__init__()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        if tie_embeddings:
            self.token_embeddings = EmbeddingLookup(weight=self.lm_head.weight, scale_factor=sqrt(d_model))
        else:
            self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, eps, context_length, theta, device=device, dtype=dtype, qk_norm=qk_norm) for _ in range(num_layers))
        self.ln_final = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.logit_soft_cap = logit_soft_cap

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:  # => [FLOPs] 24 bnd²l + 4bn²dl + 2bnd * vocab_size / 8 bnd²l + 4bn²dl + 6 bnd d_ff l + 2bnd * vocab_size
        y = self.token_embeddings(x)
        for transformer in self.layers:  # => [FLOPs] 24 bnd²l + 4bn²dl / 8 bnd²l + 4bn²dl + 6 bnd d_ff l = (27'682'406'400 + 42'152'755'200) × 48 = 3'352'087'756'800
            y = transformer(y)  # => [FLOPs] 24 bnd² + 4bn²d / 8 bnd² + 4bn²d + 6 bnd d_ff
        y = self.ln_final(y)
        y = self.lm_head(y)  # => [FLOPs] (b, n, d) @ (d, vocab_size) = 2bnd * vocab_size = 2 × 1024 × 1600 × 50257 = 164'682'137'600
        if self.logit_soft_cap is not None:
            y = self.logit_soft_cap * torch.tanh(y / self.logit_soft_cap)
        return y
