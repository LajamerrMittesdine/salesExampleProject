# C99 port

Idiomatic C99 educational port of `reference/python/kimi_k3_ref`.

```bash
make test
```

Public API: `include/kimi_k3_ref.h`  
Implementation: `src/ops.c`  
Goldens: `src/test_goldens.c` + `../fixtures/goldens_embedded.h`

| C API | Official Python |
|-------|-----------------|
| `k3_rmsnorm` | `KimiRMSNorm` |
| `k3_situ_and_mul` | `SituAndMul` |
| `k3_moe_gate` | `KimiMoEGate` (deterministic top-k) |
| `k3_expert_ffn` / `k3_latent_moe_forward` | `KimiBlockSparseMLP` / `KimiSparseMoeBlock` |
| `k3_apply_attn_res` | `_apply_attn_res` |
| `k3_kda_lowerbound_gate` / `k3_kda_recurrent` | FLA gate + `naive_recurrent_kda` |
| `k3_mla_eager_attention` | `eager_attention_forward` + output gate |
| `k3_decoder_block_forward` | `KimiDecoderLayer._forward_attn_residual` |
