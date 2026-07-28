"""Expert FFN + LatentMoE path matching KimiBlockSparseMLP / KimiSparseMoeBlock."""

from __future__ import annotations

import numpy as np

from .moe_gate import moe_gate
from .rmsnorm import rmsnorm
from .situ import situ_and_mul


def linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """y = x @ W.T  with W shaped (out, in), matching nn.Linear bias=False."""
    return (np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T).astype(
        np.float32
    )


def expert_ffn(
    hidden: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> np.ndarray:
    """
    Official KimiBlockSparseMLP with situ:
        gate_up = cat(w1(x), w3(x), dim=-1)
        y = situ(gate_up)
        return w2(y)
    """
    gate = linear(hidden, w1)
    up = linear(hidden, w3)
    gate_up = np.concatenate([gate, up], axis=-1)
    y = situ_and_mul(gate_up, beta=beta, linear_beta=linear_beta)
    return linear(y, w2)


def dense_mlp(
    hidden: np.ndarray,
    gate_proj: np.ndarray,
    up_proj: np.ndarray,
    down_proj: np.ndarray,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> np.ndarray:
    """Shared / dense MLP matching KimiMLP with situ."""
    gate = linear(hidden, gate_proj)
    up = linear(hidden, up_proj)
    gate_up = np.concatenate([gate, up], axis=-1)
    y = situ_and_mul(gate_up, beta=beta, linear_beta=linear_beta)
    return linear(y, down_proj)


def moe_infer(
    x: np.ndarray,
    topk_ids: np.ndarray,
    topk_weight: np.ndarray,
    expert_weights: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> np.ndarray:
    """
    Official moe_infer (simplified, EP size=1):
    - scatter tokens to experts by topk_ids
    - run each expert
    - weighted sum by topk_weight
    """
    x = np.asarray(x, dtype=np.float32)
    n, h = x.shape
    top_k = topk_ids.shape[1]
    # Accumulate weighted expert outputs per token
    out = np.zeros((n, h), dtype=np.float32)
    for t in range(n):
        acc = np.zeros((h,), dtype=np.float32)
        for k in range(top_k):
            e = int(topk_ids[t, k])
            w1, w2, w3 = expert_weights[e]
            y = expert_ffn(x[t : t + 1], w1, w2, w3, beta=beta, linear_beta=linear_beta)[0]
            acc += topk_weight[t, k] * y
        out[t] = acc
    return out


def latent_moe_forward(
    hidden_states: np.ndarray,
    router_weight: np.ndarray,
    router_bias: np.ndarray,
    top_k: int,
    expert_weights: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    routed_down: np.ndarray,
    routed_up: np.ndarray,
    routed_norm_weight: np.ndarray | None,
    shared_gate: np.ndarray,
    shared_up: np.ndarray,
    shared_down: np.ndarray,
    rms_eps: float = 1e-5,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
    routed_scaling_factor: float = 1.0,
) -> np.ndarray:
    """
    LatentMoE forward matching KimiSparseMoeBlock inference:
      identity = h
      route on h
      h' = down_proj(h)
      y = moe_infer(h', ...)
      y = rmsnorm(y); y = up_proj(y)
      y = y + shared_experts(identity)
    """
    identity = np.asarray(hidden_states, dtype=np.float32)
    orig_shape = identity.shape
    topk_idx, topk_weight = moe_gate(
        identity,
        router_weight,
        router_bias,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        renormalize=True,
    )
    flat = identity.reshape(-1, identity.shape[-1])
    latent = linear(flat, routed_down)
    y = moe_infer(latent, topk_idx, topk_weight, expert_weights, beta=beta, linear_beta=linear_beta)
    if routed_norm_weight is not None:
        y = rmsnorm(y, routed_norm_weight, eps=rms_eps)
    y = linear(y, routed_up)
    shared = dense_mlp(flat, shared_gate, shared_up, shared_down, beta=beta, linear_beta=linear_beta)
    y = y + shared
    return y.reshape(orig_shape).astype(np.float32)
