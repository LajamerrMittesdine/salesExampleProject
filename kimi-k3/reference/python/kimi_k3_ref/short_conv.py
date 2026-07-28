"""Causal short convolution + SiLU used before KDA q/k/v projections.

Educational implementation of FLA ``ShortConvolution`` for a single sequence
(kernel_size=4 in Kimi K3). Depthwise: each channel has its own kernel.
"""

from __future__ import annotations

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def short_conv1d_silu(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    x : (B, T, C) float32
    weight : (C, K) depthwise conv kernels (channel, kernel) — time-reversed
             so weight[..., 0] multiplies the oldest padded sample.
             Equivalent to nn.Conv1d groups=C with weight shape (C,1,K)
             flattened to (C,K).
    bias : optional (C,)

    Causal: left-pad with K-1 zeros, then for each t:
        y[t,c] = sum_{i=0}^{K-1} weight[c,i] * x_pad[t+i, c]
    Then SiLU.
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bsz, t, c = x.shape
    k = weight.shape[1]
    assert weight.shape[0] == c
    # pad left
    x_pad = np.pad(x, ((0, 0), (k - 1, 0), (0, 0)), mode="constant")
    y = np.zeros_like(x)
    for i in range(k):
        y += x_pad[:, i : i + t, :] * weight[None, None, :, i]
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)[None, None, :]
    return silu(y)
