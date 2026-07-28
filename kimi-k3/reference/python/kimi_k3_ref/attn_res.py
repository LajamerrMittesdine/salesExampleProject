"""Attention Residuals matching ``_apply_attn_res`` in modeling_kimi_linear.py."""

from __future__ import annotations

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (same definition used by all language ports)."""
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def apply_attn_res(
    prefix_sum: np.ndarray,
    block_residual: np.ndarray,
    proj_weight: np.ndarray,
    norm_weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Official ``_apply_attn_res``:

        v = cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # (N, num_blocks+1, H)
        v_float = v.float()
        variance = v_float.pow(2).mean(-1, keepdim=True)
        k = v_float * rsqrt(variance + eps)
        score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
        scores = (k * score_weight).sum(-1)
        probs = scores.softmax(-1).unsqueeze(1)
        hidden = matmul(probs, v_float).squeeze(1)

    Parameters
    ----------
    prefix_sum : (N, H)
    block_residual : (N, num_blocks, H)  — may have num_blocks=0
    proj_weight : (1, H)  — Linear(H, 1).weight
    norm_weight : (H,)
    """
    prefix_sum = np.asarray(prefix_sum, dtype=np.float32)
    block_residual = np.asarray(block_residual, dtype=np.float32)
    proj_weight = np.asarray(proj_weight, dtype=np.float32).reshape(-1)
    norm_weight = np.asarray(norm_weight, dtype=np.float32)

    # v: (N, B+1, H)
    v = np.concatenate([block_residual, prefix_sum[:, None, :]], axis=1)
    variance = np.mean(np.square(v), axis=-1, keepdims=True)
    k = v * np.reciprocal(np.sqrt(variance + np.float32(eps)))
    score_weight = norm_weight * proj_weight  # (H,)
    scores = np.sum(k * score_weight[None, None, :], axis=-1)  # (N, B+1)
    probs = stable_softmax(scores, axis=-1)  # (N, B+1)
    hidden = np.sum(probs[:, :, None] * v, axis=1)
    return hidden.astype(np.float32)
