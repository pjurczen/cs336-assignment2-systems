import math
from typing import Any

import torch
from torch.autograd.function import FunctionCtx

from cs336_systems.flash_attention import flash_bwd
from cs336_systems.flash_kernel import flash_fwd_kernel


class FlashAttention2TritonFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_casual: bool = False) -> torch.Tensor:
        batch_size = Q.shape[0]
        Q_TILE_SIZE, K_TILE_SIZE = 16, 16
        N_QUERIES, N_KEYS = Q.shape[-2], K.shape[-2]
        D = Q.shape[-1]
        scale = 1 / math.sqrt(D)

        Tq = math.ceil(N_QUERIES / Q_TILE_SIZE)

        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=Q.dtype)

        flash_fwd_kernel[(Tq, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_casual=is_casual
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_casual = is_casual

        return O

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Any):
        L, Q, K, V, O = ctx.saved_tensors
        is_casual = ctx.is_casual
        dO = grad_out

        flash_bwd_kernel = torch.compile(flash_bwd)
        return flash_bwd_kernel(Q, K, V, O, dO, L, is_casual)
