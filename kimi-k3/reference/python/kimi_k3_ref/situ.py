"""SiTU-GLU activation matching ``SituAndMul`` in modeling_kimi_linear.py."""

from __future__ import annotations

import numpy as np


def situ_and_mul(
    x: np.ndarray,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> np.ndarray:
    """
    Official SituAndMul:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        situ_a = beta * tanh(gate / beta) * sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * tanh(up / linear_beta)
        return situ_a * up

    Kimi K3 config: activation_situ_beta=4.0, activation_situ_linear_beta=25.0
    """
    x = np.asarray(x, dtype=np.float32)
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    beta_f = np.float32(beta)
    situ_a = beta_f * np.tanh(gate / beta_f) * (1.0 / (1.0 + np.exp(-gate)))
    if linear_beta is not None:
        lb = np.float32(linear_beta)
        up = lb * np.tanh(up / lb)
    return (situ_a * up).astype(np.float32)
