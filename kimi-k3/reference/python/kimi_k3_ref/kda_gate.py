"""KDA forget-gate matching FLA ``naive_kda_lowerbound_gate`` (Kimi K3 path).

K3 config: gate_lower_bound=-5.0, use_gate_in_kernel=True, safe_gate=True.
"""

from __future__ import annotations

import numpy as np


def kda_lowerbound_gate(
    g: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray | None = None,
    lower_bound: float = -5.0,
) -> np.ndarray:
    """
    Official (FLA gate.py):
        g = g + dt_bias.view(H, -1)   # if bias
        g = lower_bound * sigmoid(exp(A_log) * g)

    Parameters
    ----------
    g : (..., H, K)  pre-activation gate logits (from f_b_proj(f_a_proj(x)))
    a_log : (H,)
    dt_bias : (H, K) or (H*K,) flattened — K3 stores as projection_size = H*K
    lower_bound : scalar, K3 = -5.0

    Returns
    -------
    log-space decay values shaped like g  (actually the gated log-alpha used as g in KDA)
    Note: FLA naive_recurrent expects g already in log-space such that state *= exp(g).
    The lowerbound gate returns values in (lower_bound, 0) which are used as log-decay.
    """
    g = np.asarray(g, dtype=np.float32)
    a_log = np.asarray(a_log, dtype=np.float32)
    h = a_log.shape[0]
    k = g.shape[-1]
    if dt_bias is not None:
        dt = np.asarray(dt_bias, dtype=np.float32).reshape(h, k)
        g = g + dt
    # broadcast A_log over K
    scale = np.exp(a_log).reshape((1,) * (g.ndim - 2) + (h, 1))
    sig = 1.0 / (1.0 + np.exp(-(scale * g)))
    return (np.float32(lower_bound) * sig).astype(np.float32)
