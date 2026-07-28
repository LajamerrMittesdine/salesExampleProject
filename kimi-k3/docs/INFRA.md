# Infrastructure Notes (FlashKDA, MoonEP, Quant, Deploy)

## FlashKDA

- Repo: https://github.com/MoonshotAI/FlashKDA (vendored README + `torch_ref.py`)
- CUTLASS kernels for Kimi Delta Attention on SM90+ (Hopper+)
- Auto-dispatched from `flash-linear-attention`’s `chunk_kda` when installed
- API: `flash_kda.fwd(q,k,v,g,beta,scale,out,A_log,dt_bias,lower_bound,...)`
- Requires head dim K=V=128 (matches K3)
- Opt out: `FLA_FLASH_KDA=0`
- Call chain in K3 HF code: `KimiDeltaAttention.forward` → `fla.ops.kda.chunk_kda` (prefill) or `fused_recurrent_kda` (decode) → optional FlashKDA backend
- Deeper reading: `docs/CODE_WALKTHROUGH.md` §5

Educational ports intentionally implement the **naive recurrence**, not the CUTLASS path.

## FLA dependency

Official HF modeling imports:

```python
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
```

Install: `pip install -U fla-core` / `flash-linear-attention` (see modeling file assert on transformers ≥ 4.56).

## MoonEP

- Perfectly balanced expert-parallel library for training at K3 sparsity
- Symmetric-memory weight buffers, static shapes, prefetch slots
- Vendored overview: `official/moonep/README.md`
- Not required to understand the **algorithmic** LatentMoE forward in the educational core

## Quantization-aware training

From HF model card / config:

- MXFP4 packed weights for most Linear layers
- Attention, shared experts, vision, lm_head often excluded (`quantization_config.ignore`)
- Activations targeted as MXFP8 in the training recipe

## Deployment hints (from Moonshot materials)

- Prefer large high-bandwidth domains (supernodes with ≥64 accelerators) for expert parallel inference efficiency
- KDA changes prefix-cache behavior vs pure MLA/Transformer KV caches — Moonshot notes a corresponding cache implementation alongside the release
- Official API uses Mooncake disaggregated inference with high cache-hit rates on coding workloads

## What this investigation tree does *not* run

- Full 2.8T weight load
- Multi-GPU EP training
- CUDA kernel builds for FlashKDA
