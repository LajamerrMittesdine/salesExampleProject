# Source Map: Report / Feature → Official Code → Reference → Ports

| Concept | Tech report | Official Python | Reference (Python) | C / C++ / Rust / Go |
|---------|-------------|-----------------|--------------------|---------------------|
| SiTU-GLU | §2.3 / activation | `SituAndMul` in `modeling_kimi_linear.py` | `situ.py` | `situ.*` |
| RMSNorm | norms throughout | `KimiRMSNorm` | `rmsnorm.py` | `rmsnorm.*` |
| KDA recurrence | §2.1.1 eq (1) | FLA `naive_recurrent_kda`; module `KimiDeltaAttention` | `kda_recurrent.py` | `kda_recurrent.*` |
| KDA lower-bound gate | §2.1.1 | FLA `naive_kda_lowerbound_gate`; used via `chunk_kda(..., safe_gate=True, lower_bound=-5)` | `kda_gate.py` | `kda_gate.*` |
| Short conv + SiLU | §2.1.1 | FLA `ShortConvolution` in `KimiDeltaAttention` | `short_conv.py` | `short_conv.*` |
| Q/K L2Norm | §2.1.1 | `use_qk_l2norm_in_kernel=True` | `l2_normalize` in `kda_recurrent.py` | same |
| Chunk / Flash KDA | §2.1.1 / infra | `fla.ops.kda.chunk_kda`, FlashKDA `flash_kda.fwd` | `kda_chunk.py` (numpy; ports use recurrent) | — |
| Gated MLA | §2.1.2 | `KimiMLAAttention` | `mla_eager.py` | `mla_eager.*` |
| MLA output gate | §2.1.2 | `g_proj` + sigmoid | `gated_mla_output` | same |
| AttnRes apply | §2.2 | `_apply_attn_res` | `attn_res.py` | `attn_res.*` |
| AttnRes layer flow | §2.2 | `KimiDecoderLayer._forward_attn_residual` | `decoder_block.py` | `decoder_block.*` |
| MoE router | §2.3 | `KimiMoEGate` | `moe_gate.py` (deterministic top-k) | `moe_gate.*` |
| Expert FFN | §2.3 | `KimiBlockSparseMLP` | `expert_ffn.py` | `expert_ffn.*` |
| LatentMoE | §2.3 | `KimiSparseMoeBlock` | `latent_moe_forward` | same |
| Shared experts | §2.3 | `KimiMLP` as `shared_experts` | inside `latent_moe_forward` | same |
| Dynamic cache | decode | `KimiDynamicCache` | `docs/CODE_WALKTHROUGH.md` §2 | — |
| XTML chat/tools | product | `encoding_k3.py` | `docs/CODE_WALKTHROUGH.md` §3 | — |
| Image merge | §2.4 | `_merge_input_ids_with_image_features` | `docs/CODE_WALKTHROUGH.md` §4 | — |
| FlashKDA dispatch | infra | FLA `chunk_kda` → `flash_kda.fwd` | `docs/CODE_WALKTHROUGH.md` §5 | — |
| Backbone loop | §2 | `KimiLinearModel.forward` | — | — |
| Multimodal merge | §2.4 | `KimiK3ForConditionalGeneration` | official only | — |
| MoonViT-V2 | §2.4 | classes in `modeling_kimi_k3.py` | official only | — |
| Chat / tools XTML | product | `encoding_k3.py`, `tokenization_kimi.py` | official only | — |
| Config / schedule | tables | `configuration_kimi_k3.py`, `config.json` | fixtures use reduced dims | — |
| Expert parallel | infra | MoonEP | `official/moonep/README.md` | — |

## Layer schedule (1-based config indices)

```
1–3 KDA, 4 MLA, 5–7 KDA, 8 MLA, ... 89–91 KDA, 92 MLA, 93 MLA
```

`is_kda_layer(layer_idx)` ↔ `(layer_idx + 1) in linear_attn_config["kda_layers"]`.

## File index (official)

| Path | Role |
|------|------|
| `official/python/modeling_kimi_linear.py` | Backbone: KDA, MLA, MoE, AttnRes, SiTU |
| `official/python/modeling_kimi_k3.py` | Vision + multimodal causal LM |
| `official/python/configuration_kimi_k3.py` | Config classes |
| `official/python/encoding_k3.py` | Conversation / tool encoding |
| `official/python/tokenization_kimi.py` | TikToken tokenizer |
| `official/python/kimi_k3_processor.py` | Processor |
| `official/python/config.json` | Full hyperparams |
| `official/fla_kda/naive.py` | Mathematical KDA reference |
| `official/fla_kda/gate.py` | Gate kernels + naive gates |
| `official/flashkda/torch_ref.py` | Kernel-faithful torch reference |
| `official/reports/k3_tech_report.pdf` | Full technical report |
