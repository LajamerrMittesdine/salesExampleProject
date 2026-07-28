"""Dependency-free float32 reference core for Kimi K3 educational ports.

These modules mirror the mathematical behavior of Moonshot AI's official
Hugging Face modeling code and FLA naive KDA helpers, without requiring
torch / transformers / fla at runtime.

Official sources live under ``kimi-k3/official/``.
"""

from .rmsnorm import rmsnorm
from .situ import situ_and_mul
from .moe_gate import moe_gate
from .expert_ffn import expert_ffn, latent_moe_forward
from .attn_res import apply_attn_res
from .short_conv import short_conv1d_silu
from .kda_gate import kda_lowerbound_gate
from .kda_recurrent import kda_recurrent, l2_normalize
from .mla_eager import mla_eager_attention, gated_mla_output
from .decoder_block import decoder_block_forward

__all__ = [
    "rmsnorm",
    "situ_and_mul",
    "moe_gate",
    "expert_ffn",
    "latent_moe_forward",
    "apply_attn_res",
    "short_conv1d_silu",
    "kda_lowerbound_gate",
    "kda_recurrent",
    "l2_normalize",
    "mla_eager_attention",
    "gated_mla_output",
    "decoder_block_forward",
]
