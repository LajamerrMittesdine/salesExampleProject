"""Naive recurrent KDA matching FLA ``naive_recurrent_kda``.

This is the clearest one-to-one mathematical form of Kimi Delta Attention.
Production inference uses chunk_kda / FlashKDA; see NUMERICS.md.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """L2 normalize over the last dimension (matches use_qk_l2norm_in_kernel intent)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True) + np.float32(eps))
    return (x / n).astype(np.float32)


def kda_recurrent(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float | None = None,
    initial_state: np.ndarray | None = None,
    l2norm_qk: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters (FLA naive_recurrent_kda)
    ------------------------------------
    q : (B, T, H, K)
    k : (B, T, H, K)
    v : (B, T, HV, V)  — for K3, HV == H and V == K == head_dim
    g : (B, T, HV, K)  — log-space decay (output of lowerbound gate)
    beta : (B, T, HV)
    scale : default 1/sqrt(K)
    initial_state : (B, HV, K, V) or None
    l2norm_qk : if True, L2-normalize q and k (K3 kernel flag)

    Returns
    -------
    o : (B, T, HV, V)
    S : (B, HV, K, V) final state

    Recurrence (per token t, after expanding q/k to HV):
        S = S * exp(g_t)[..., None]
        delta = v_t - sum_k (k_t[..., None] * S)
        S = S + einsum(beta * k, delta)
        o_t = einsum(q_t, S)
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    b, t, h, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    assert hv % h == 0
    group = hv // h
    if scale is None:
        scale = float(dk) ** -0.5

    if l2norm_qk:
        q = l2_normalize(q)
        k = l2_normalize(k)

    # Expand q/k heads to value heads and scale q
    if group != 1:
        q = np.repeat(q, group, axis=2)
        k = np.repeat(k, group, axis=2)
    q = q * np.float32(scale)

    s = np.zeros((b, hv, dk, dv), dtype=np.float32)
    if initial_state is not None:
        s = s + np.asarray(initial_state, dtype=np.float32)

    o = np.zeros((b, t, hv, dv), dtype=np.float32)
    for i in range(t):
        q_i = q[:, i]  # (B, HV, K)
        k_i = k[:, i]
        v_i = v[:, i]
        g_i = g[:, i]
        b_i = beta[:, i]  # (B, HV)

        # S = S * exp(g)[..., None]
        s = s * np.exp(g_i)[..., None]
        # k·S over key dim: (B, HV, V)
        k_s = np.sum(k_i[..., None] * s, axis=-2)
        delta = v_i - k_s
        # S += (beta * k) ⊗ delta
        bk = b_i[..., None] * k_i  # (B, HV, K)
        s = s + bk[..., None] * delta[..., None, :]
        # o = q · S
        o[:, i] = np.sum(q_i[..., None] * s, axis=-2)

    return o.astype(np.float32), s.astype(np.float32)
