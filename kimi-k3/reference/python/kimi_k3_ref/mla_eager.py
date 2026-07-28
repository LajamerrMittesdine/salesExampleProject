"""Eager Multi-Latent Attention pieces matching KimiMLAAttention (no FlashAttention).

Educational reduced-dimension form: callers supply already-projected Q/K/V tensors
shaped like the post-LoRA / post-split tensors in the official forward.
"""

from __future__ import annotations

import numpy as np

from .attn_res import stable_softmax


def mla_eager_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    scaling: float,
    causal: bool = True,
) -> np.ndarray:
    """
    Parameters
    ----------
    query : (B, H, Q, D)
    key   : (B, H, K, D)
    value : (B, H, K, Dv)
    scaling : 1/sqrt(q_head_dim) in official code

    Official eager_attention_forward (simplified, n_rep=1):
        scores = einsum(bhqd,bhkd->bhqk) * scaling
        probs = softmax(scores)
        out = einsum(bhqk,bhkd->bhqd)
    Then transpose to (B, Q, H, Dv) flattened by caller.
    """
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)

    # scores: (B, H, Q, K)
    scores = np.einsum("bhqd,bhkd->bhqk", query, key) * np.float32(scaling)
    if causal:
        q_len = scores.shape[-2]
        k_len = scores.shape[-1]
        # allow attending to last q_len keys when k_len >= q_len (cached decode style)
        mask = np.ones((q_len, k_len), dtype=bool)
        # standard causal: position i attends to keys <= i in the query window aligned to end
        offset = k_len - q_len
        for i in range(q_len):
            mask[i, offset + i + 1 :] = False
        scores = np.where(mask[None, None, :, :], scores, np.float32(-1e9))

    probs = stable_softmax(scores, axis=-1)
    out = np.einsum("bhqk,bhkd->bhqd", probs, value)
    # return (B, Q, H, Dv) like official .transpose(1,2)
    return np.transpose(out, (0, 2, 1, 3)).astype(np.float32)


def gated_mla_output(
    attn_output: np.ndarray,
    gate_logits: np.ndarray,
) -> np.ndarray:
    """
    Official:
        g = g_proj(hidden).sigmoid()
        attn_output = attn_output * g
    attn_output and gate_logits: (B, T, H*Dv)
    """
    attn_output = np.asarray(attn_output, dtype=np.float32)
    gate = 1.0 / (1.0 + np.exp(-np.asarray(gate_logits, dtype=np.float32)))
    return (attn_output * gate).astype(np.float32)
