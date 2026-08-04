# C++20 port

Idiomatic C++20 (`std::span`, `std::vector`, `std::optional`, namespace `kimi::k3`).

```bash
make test
```

Headers: `include/kimi_k3_ref.hpp`  
Implementation: `src/ops.cpp`

Symbols in `kimi::k3` mirror `reference/python/kimi_k3_ref` 1:1 (`rmsnorm`, `situ_and_mul`,
`moe_gate`, `expert_ffn`, `latent_moe_forward`, `apply_attn_res`, `short_conv1d_silu`,
`kda_lowerbound_gate`, `kda_recurrent`, `mla_eager_attention`, `decoder_block_forward`),
which themselves map to official HF/FLA symbols listed in `../docs/SOURCE_MAP.md`.
