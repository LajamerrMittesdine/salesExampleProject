//! Float32 educational reference for Kimi K3 core ops.
//! Mirrors `kimi-k3/reference/python/kimi_k3_ref/` one-to-one.

pub mod attn_res;
pub mod decoder_block;
pub mod expert_ffn;
pub mod kda_gate;
pub mod kda_recurrent;
pub mod mla_eager;
pub mod moe_gate;
pub mod rmsnorm;
pub mod short_conv;
pub mod situ;

pub use attn_res::apply_attn_res;
pub use decoder_block::decoder_block_forward;
pub use expert_ffn::{expert_ffn, latent_moe_forward, linear};
pub use kda_gate::kda_lowerbound_gate;
pub use kda_recurrent::{kda_recurrent, l2_normalize};
pub use mla_eager::{gated_mla_output, mla_eager_attention};
pub use moe_gate::moe_gate;
pub use rmsnorm::rmsnorm;
pub use short_conv::short_conv1d_silu;
pub use situ::situ_and_mul;
