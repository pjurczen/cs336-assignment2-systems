import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_casual: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (6)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (6)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)  # (6)

    Q_i = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)

    q_pos = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    k_arr = tl.arange(0, K_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE) (9)

        if is_casual:
            k_pos = j * K_TILE_SIZE + k_arr
            mask = k_pos[None, :] > q_pos[:, None]

            S_ij = tl.where(mask, S_ij - 1e6, S_ij)

        rowmax_S_ij = tl.max(S_ij, axis=1)  # (Q_TILE_SIZE, ) (10)
        m_ij = tl.maximum(m_i, rowmax_S_ij)  # (Q_TILE_SIZE, ) (10)
        P_ij = tl.exp(S_ij - tl.expand_dims(m_ij, 1))  # (Q_TILE_SIZE, K_TILE_SIZE) (11)
        rowsum_P_ij = tl.sum(P_ij, axis=-1)  # (Q_TILE_SIZE, ) (12)
        l_ij = (tl.exp(m_i - m_ij) * l_i) + rowsum_P_ij  # (Q_TILE_SIZE, ) (12)
        O_i = tl.expand_dims(tl.exp(m_i - m_ij), 1) * O_i  # (13)
        O_i = tl.dot(P_ij.to(V_j.dtype), V_j, acc=O_i)  # (13)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        m_i = m_ij
        l_i = l_ij

    O_i = O_i / tl.expand_dims(l_i, 1)  # (15)
    L_i = m_i + tl.log(l_i)  # (16)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))
