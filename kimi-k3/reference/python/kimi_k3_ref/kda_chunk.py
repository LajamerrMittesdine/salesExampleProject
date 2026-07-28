"""Chunkwise KDA matching FLA ``naive_chunk_kda`` (educational float32).

Production prefill uses FLA/FlashKDA kernels; this module is the readable
chunk algebra from ``official/fla_kda/naive.py``. Cross-language ports use
``kda_recurrent`` (same recurrence, simpler control flow).
"""

from __future__ import annotations

import numpy as np

from .kda_recurrent import l2_normalize


def kda_chunk(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float | None = None,
    initial_state: np.ndarray | None = None,
    chunk_size: int = 4,
    l2norm_qk: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters match FLA ``naive_chunk_kda`` (numpy).

    q,k : (B, T, H, K)
    v   : (B, T, HV, V)
    g   : (B, T, HV, K)  log-space decays
    beta: (B, T, HV)
    T must be divisible by ``chunk_size``.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    bsz, seq_len, h, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    assert hv % h == 0
    group = hv // h
    bt = chunk_size
    assert seq_len % bt == 0, "T must be divisible by chunk_size"
    nt = seq_len // bt
    if scale is None:
        scale = float(dk) ** -0.5

    if l2norm_qk:
        q = l2_normalize(q)
        k = l2_normalize(k)

    def to_chunks(x: np.ndarray) -> np.ndarray:
        # (B, T, H, ...) -> (B, H, NT, BT, ...)
        rest = x.shape[3:]
        x = x.reshape(bsz, nt, bt, x.shape[2], *rest)
        return np.transpose(x, [0, 3, 1, 2] + list(range(4, x.ndim)))

    qh = to_chunks(q)
    kh = to_chunks(k)
    vh = to_chunks(v)
    gh = to_chunks(g)
    bh = to_chunks(beta[..., None])[..., 0]

    if group != 1:
        qh = np.repeat(qh, group, axis=1)
        kh = np.repeat(kh, group, axis=1)
    qh = qh * np.float32(scale)

    gh = np.cumsum(gh, axis=3)

    # A[..., i] = einsum(k * exp(g - g_i), k_i) * beta
    a = np.zeros((bsz, hv, nt, bt, bt), dtype=np.float32)
    for i in range(bt):
        k_i = kh[:, :, :, i, :]
        g_i = gh[:, :, :, i : i + 1, :]
        a[:, :, :, :, i] = np.sum(kh * np.exp(gh - g_i) * k_i[:, :, :, None, :], axis=-1)
    a = a * bh[..., None]

    tril_incl = np.triu(np.ones((bt, bt), dtype=bool), k=0)
    a = -np.where(tril_incl[None, None, None, :, :], 0.0, a)
    for i in range(1, bt):
        left = a[:, :, :, i, :, None]
        right = a[:, :, :, :, :i]
        a[:, :, :, i, :i] = a[:, :, :, i, :i] + np.sum(left * right, axis=-2)
    a = (a + np.eye(bt, dtype=np.float32)) * bh[..., None, :]

    # a: (B,HV,NT,BT,BT)  gek/vh: (B,HV,NT,BT,D)
    gek = np.exp(gh) * kh
    w = np.einsum("bhnij,bhnjd->bhnid", a, gek)
    u = np.einsum("bhnij,bhnjd->bhnid", a, vh)

    state = np.zeros((bsz, hv, dk, dv), dtype=np.float32)
    if initial_state is not None:
        state = state + np.asarray(initial_state, dtype=np.float32)

    out = np.zeros_like(vh)
    causal_strict = np.triu(np.ones((bt, bt), dtype=bool), k=1)
    for i in range(nt):
        q_i = qh[:, :, i]
        k_i = kh[:, :, i]
        u_i = u[:, :, i]
        g_i = gh[:, :, i]
        w_i = w[:, :, i]
        aqk = np.zeros((bsz, hv, bt, bt), dtype=np.float32)
        for j in range(bt):
            k_j = k_i[:, :, j, :]
            g_j = g_i[:, :, j : j + 1, :]
            aqk[:, :, :, j] = np.sum(
                q_i * np.exp(g_i - g_j) * k_j[:, :, None, :], axis=-1
            )
        aqk = np.where(causal_strict[None, None, :, :], 0.0, aqk)
        v_i = u_i - np.einsum("bhik,bhkv->bhiv", w_i, state)
        out[:, :, i] = np.einsum("bhik,bhkv->bhiv", q_i * np.exp(g_i), state) + np.einsum(
            "bhij,bhjv->bhiv", aqk, v_i
        )
        g_last = g_i[:, :, -1, :]
        state = state * np.exp(g_last)[..., None]
        decay = np.exp(g_last[:, :, None, :] - g_i) * k_i
        state = state + np.einsum("bhck,bhcv->bhkv", decay, v_i)

    out = np.transpose(out, (0, 2, 3, 1, 4)).reshape(bsz, seq_len, hv, dv)
    return out.astype(np.float32), state.astype(np.float32)
