"""Educational decoder-block residual bookkeeping with AttnRes + attention + MoE.

Mirrors the control flow of ``KimiDecoderLayer._forward_attn_residual`` at reduced
dimensions. Attention and MoE are injected as callables / precomputed tensors so
ports can unit-test the residual algebra independently of full KDA/MLA.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .attn_res import apply_attn_res


def decoder_block_forward(
    hidden_states: np.ndarray,
    block_residual: np.ndarray,
    attn_fn: Callable[[np.ndarray], np.ndarray],
    mlp_fn: Callable[[np.ndarray], np.ndarray],
    input_norm_weight: np.ndarray,
    post_attn_norm_weight: np.ndarray,
    self_attn_res_proj: np.ndarray,
    self_attn_res_norm: np.ndarray,
    mlp_res_proj: np.ndarray,
    mlp_res_norm: np.ndarray,
    layer_idx: int,
    attn_res_block_size: int,
    rms_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (prefix_sum, block_residual) matching the official AttnRes layer return.

    Official sketch:
      prefix_sum = hidden_states
      if block_residual has blocks: hidden = apply_attn_res(prefix, block, attn_proj, attn_norm)
      if layer_idx % block_size == 0: append prefix to block_residual; prefix=None
      hidden = input_ln(hidden); hidden = attn(hidden)
      prefix = (prefix + hidden) or hidden
      hidden = apply_attn_res(prefix, block, mlp_proj, mlp_norm)
      hidden = post_ln(hidden); hidden = mlp(hidden)
      prefix = prefix + hidden
    """
    from .rmsnorm import rmsnorm

    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    block_residual = np.asarray(block_residual, dtype=np.float32)
    batch, seq, hidden = hidden_states.shape
    n = batch * seq
    prefix_sum = hidden_states

    if block_residual.shape[1] > 0:
        hidden_states = apply_attn_res(
            prefix_sum.reshape(n, hidden),
            block_residual,
            self_attn_res_proj,
            self_attn_res_norm,
            eps=rms_eps,
        ).reshape(batch, seq, hidden)

    if layer_idx % attn_res_block_size == 0:
        # append current prefix_sum as a new block vector
        new_block = prefix_sum.reshape(n, hidden)[:, None, :]
        block_residual = np.concatenate([block_residual, new_block], axis=1)
        prefix_sum = None

    hidden_states = rmsnorm(hidden_states, input_norm_weight, eps=rms_eps)
    hidden_states = attn_fn(hidden_states)

    if prefix_sum is not None:
        prefix_sum = prefix_sum + hidden_states
    else:
        prefix_sum = hidden_states

    hidden_states = apply_attn_res(
        prefix_sum.reshape(n, hidden),
        block_residual,
        mlp_res_proj,
        mlp_res_norm,
        eps=rms_eps,
    ).reshape(batch, seq, hidden)

    hidden_states = rmsnorm(hidden_states, post_attn_norm_weight, eps=rms_eps)
    hidden_states = mlp_fn(hidden_states)

    prefix_sum = prefix_sum + hidden_states
    return prefix_sum.astype(np.float32), block_residual.astype(np.float32)
