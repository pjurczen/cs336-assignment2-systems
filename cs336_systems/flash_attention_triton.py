import math
from typing import Any

import torch
from torch.autograd.function import FunctionCtx

from cs336_systems.flash_kernel import flash_fwd_kernel, flash_bwd_kv_kernel, flash_bwd_q_kernel


class FlashAttention2TritonFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_casual: bool = False) -> torch.Tensor:
        batch_size = Q.shape[0]
        N_QUERIES, N_KEYS = Q.shape[-2], K.shape[-2]
        D = Q.shape[-1]
        scale = 1 / math.sqrt(D)

        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=Q.dtype)

        grid = lambda meta: (math.ceil(N_QUERIES / meta["Q_TILE_SIZE"]), batch_size)
        flash_fwd_kernel[grid](
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
        d = Q.shape[-1]
        scale = 1 / math.sqrt(d)
        N_QUERIES, N_KEYS = Q.shape[-2], K.shape[-2]
        batch_size = Q.shape[0]

        D = torch.sum(O * dO, dim=-1)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        grid_kv = lambda meta: (math.ceil(N_KEYS / meta["K_TILE_SIZE"]), batch_size)
        flash_bwd_kv_kernel[grid_kv](
            Q, K, V, dO, L, D,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            d=d,
            is_casual=is_casual
        )

        grid_q = lambda meta: (math.ceil(N_QUERIES / meta["Q_TILE_SIZE"]), batch_size)
        flash_bwd_q_kernel[grid_q](
            Q, K, V, dO, L, D,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            d=d,
            is_casual=is_casual
        )

        return dQ, dK, dV, None
