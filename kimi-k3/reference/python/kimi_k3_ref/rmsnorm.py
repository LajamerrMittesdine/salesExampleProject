"""RMSNorm matching ``KimiRMSNorm`` in modeling_kimi_linear.py."""

from __future__ import annotations

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Parameters
    ----------
    x : float32 array, shape (..., hidden)
    weight : float32 array, shape (hidden,)
    eps : variance epsilon (official default rms_norm_eps=1e-5)

    Official:
        x = x.float()
        x = x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return weight * x.to(dtype)
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    y = x * np.reciprocal(np.sqrt(var + np.float32(eps)))
    return (weight * y).astype(np.float32)
