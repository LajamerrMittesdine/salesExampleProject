"""MoE router matching ``KimiMoEGate`` (inference path) with deterministic top-k.

Official uses ``torch.topk(..., sorted=False)`` which is nondeterministic across
backends. This educational core uses a **deterministic** top-k:
for each row, select the ``top_k`` largest scores; ties broken by smaller index.
Returned indices are sorted ascending by expert id (not by score) so gather/scatter
order is stable across languages. Weights are gathered from the original sigmoid
scores (not the bias-adjusted scores), matching the official gate.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def deterministic_topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """
    scores: (N, E) float32
    returns: (N, top_k) int64 indices, sorted ascending per row
    """
    n, e = scores.shape
    # lexsort: primary key = -score (desc), secondary = index (asc for ties)
    idx = np.arange(e, dtype=np.int64)
    out = np.empty((n, top_k), dtype=np.int64)
    for i in range(n):
        order = np.lexsort((idx, -scores[i]))
        chosen = np.sort(order[:top_k])
        out[i] = chosen
    return out


def moe_gate(
    hidden_states: np.ndarray,
    weight: np.ndarray,
    e_score_correction_bias: np.ndarray,
    top_k: int,
    routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    hidden_states : (B, S, H) or (N, H)
    weight : (E, H) router weight
    e_score_correction_bias : (E,)
    top_k : experts per token (K3: 16)
    routed_scaling_factor : K3 default 1.0
    renormalize : K3 moe_renormalize=True

    Returns
    -------
    topk_idx : (N, top_k) int64
    topk_weight : (N, top_k) float32
    """
    hs = np.asarray(hidden_states, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(e_score_correction_bias, dtype=np.float32)

    if hs.ndim == 3:
        b, s, h = hs.shape
        hs = hs.reshape(b * s, h)
    elif hs.ndim != 2:
        raise ValueError("hidden_states must be (B,S,H) or (N,H)")

    # logits = hidden @ weight.T
    logits = hs @ w.T
    scores = _sigmoid(logits)
    scores_for_choice = scores + bias[None, :]
    topk_idx = deterministic_topk_indices(scores_for_choice, top_k)
    # gather original sigmoid scores
    topk_weight = np.take_along_axis(scores, topk_idx, axis=1)
    if top_k > 1 and renormalize:
        denom = topk_weight.sum(axis=1, keepdims=True) + np.float32(1e-20)
        topk_weight = topk_weight / denom
    topk_weight = (topk_weight * np.float32(routed_scaling_factor)).astype(np.float32)
    return topk_idx, topk_weight
