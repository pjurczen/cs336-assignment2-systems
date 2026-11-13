import math
from typing import Any

import torch
from einops import einsum
from torch.autograd.function import FunctionCtx


class FlashAttention2Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        Q_TILE_SIZE, K_TILE_SIZE = 16, 16
        device, dtype = Q.device, Q.dtype
        q_tiles_count = math.ceil(Q.shape[-2] / Q_TILE_SIZE)  # (2)
        k_tiles_count = math.ceil(K.shape[-2] / K_TILE_SIZE)  # (3)
        d_k = Q.shape[-1]
        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=device, dtype=dtype)

        for i in range(q_tiles_count):  # (4)
            Q_i = Q[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE, :]  # (5)
            O_i = torch.zeros_like(Q_i)  # (6)
            l_i = torch.zeros(Q_i.shape[:-1], device=device, dtype=dtype)  # (6)
            m_i = torch.full(Q_i.shape[:-1], float('-inf'), device=device, dtype=dtype)  # (6)
            for j in range(k_tiles_count):  # (7)
                K_j = K[..., j * K_TILE_SIZE: (j + 1) * K_TILE_SIZE, :]  # (8)
                V_j = V[..., j * K_TILE_SIZE: (j + 1) * K_TILE_SIZE, :]  # (8)
                S_ij = einsum(Q_i, K_j, "... q d, ... k d -> ... q k") / math.sqrt(d_k)  # (9)
                rowmax_S_ij = S_ij.max(dim=-1).values  # (10)
                m_ij = torch.max(m_i, rowmax_S_ij)  # (10)
                P_ij = torch.exp(S_ij - m_ij[..., None])  # (11)
                rowsum_P_ij = P_ij.sum(dim=-1)  # (12)
                l_ij = (torch.exp(m_i - m_ij) * l_i) + rowsum_P_ij  # (12)
                Pij_Vj = einsum(P_ij, V_j, "... Bq Bk, ... Bk d -> ... Bq d")  # (13)
                O_i = torch.exp(m_i - m_ij)[..., None] * O_i + Pij_Vj  # (13)
                m_i = m_ij
                l_i = l_ij

            O_i = O_i / l_i[..., None]  # (15)
            L_i = m_i + l_i.log()  # (16)

            O[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE, :] = O_i  # (17)
            L[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE] = L_i  # (18)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Any):
        raise NotImplementedError
