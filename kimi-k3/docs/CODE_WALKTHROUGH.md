# Official Code Walkthrough (Phase 4)

Companion notes for reading the vendored Hugging Face / FLA / FlashKDA sources.
For equations and the educational ports, see `ARCHITECTURE.md` and `SOURCE_MAP.md`.

---

## 1. End-to-end token path

```text
messages / images
    │
    ├─ encoding_k3.build_chat_segments  →  EncodeSegment list (XTML)
    ├─ tokenization_kimi.encode          →  input_ids (+ specials)
    ├─ kimi_k3_processor / vision        →  pixel_values, grid_thws
    │
    ▼
KimiK3ForConditionalGeneration.forward
    ├─ vision_tower(pixel_values, grid_thws)     # MoonViT-V2
    ├─ mm projector (PatchMergerMLPV2)           # → text hidden 7168
    ├─ embed_tokens(input_ids)
    ├─ _merge_input_ids_with_image_features      # expand placeholders
    │
    ▼
KimiLinearForCausalLM / KimiLinearModel
    ├─ for layer in 0..92:
    │     AttnRes bookkeeping (block_residual)
    │     KDA  or  Gated MLA
    │     LatentMoE (or dense MLP on layer 0)
    ├─ output AttnRes + final RMSNorm
    └─ lm_head → logits
```

---

## 2. `KimiDynamicCache` (decode state)

**File:** `official/python/modeling_kimi_linear.py` → class `KimiDynamicCache`

K3 is hybrid, so the cache stores **different tensors per layer type**:

| Field | Used by | Meaning |
|-------|---------|---------|
| `layer_types[i]` | both | `"linear_attention"` (KDA) or `"full_attention"` (MLA) |
| `conv_states[i]` | KDA | `(q_conv, k_conv, v_conv)` ShortConvolution rolling buffers |
| `recurrent_states[i]` | KDA | recurrent matrix `S` (`[B,H,K,V]` layout after transpose flag) |
| `key_cache[i]` / `value_cache[i]` | MLA | concatenated KV along sequence dim |

**Prefill vs decode (KDA):**

- Prefill (`q_len > 1` or training): `mode = "chunk"` → `chunk_kda(...)`
- Single-token decode with cache: `mode = "fused_recurrent"` → `fused_recurrent_kda(...)`

`KimiDeltaAttention.forward` reads/writes `cache_params.conv_states[layer_idx]` and
`cache_params.recurrent_states[layer_idx]`.

`update()` only grows MLA KV caches. Sequence length for masks comes from an MLA
layer (`transformer_layers[0]`), not from KDA state size.

`reorder_cache(beam_idx)` index-selects MLA KV **and** KDA conv/recurrent states.

Educational ports do **not** implement this cache object; they expose the math of
one KDA step / AttnRes / MoE call. Use this section when reading official decode.

---

## 3. Chat / tool XTML encoding

**File:** `official/python/encoding_k3.py`

K3 chat is rendered as **XTML-like control markup**, not plain ChatML:

- Control tokens (encoded as specials): `<|open|>`, `<|close|>`, `<|sep|>`, `<|end_of_msg|>`
- Image placeholder string: `<|kimi_image_placeholder|>` (token id from config: `163605`)
- Thinking effort attributes: `low` / `high` / `max` (`_VALID_THINKING_EFFORTS`)

Core types:

- `EncodeSegment(text, allow_special)` — structural vs ordinary text
- `build_chat_segments(...)` — main entry used by the processor/tokenizer path
- Tool calls / results normalized via `normalize_conversation`, XTML tool declare/result helpers

**Reading tip:** start at `build_chat_segments`, then `_render_assistant_segments` and
`_render_tool_declare`. Token IDs are produced later by `tokenization_kimi.TikTokenTokenizer`
consuming those segments (`allow_special` decides special-token encoding).

---

## 4. Image placeholder merge

**File:** `official/python/modeling_kimi_k3.py` → `_merge_input_ids_with_image_features`

After vision features are extracted and projected to text width:

1. Count how many tokens each `media_placeholder_token_id` occupies (`feature_lengths`).
2. Build a new sequence length = sum of occupations (text tokens stay length 1; each
   image placeholder expands to `num_image_tokens`).
3. Scatter text embeddings into their new positions; fill remaining slots with image
   features; rebuild attention mask / position ids; zero pad positions.

This is why one placeholder id in `input_ids` can expand to hundreds of visual tokens
in `inputs_embeds` before the language backbone runs.

Related helpers:

- `_extract_image_features` → `vision_tower` then list of per-image feature tensors
- `kimi_k3_vision_processing.py` / `media_utils.py` — resize, NaViT-style patchify, prompts

---

## 5. FlashKDA dispatch from HF / FLA

**Official call site:** `KimiDeltaAttention.forward` → `chunk_kda` / `fused_recurrent_kda`
from `fla.ops.kda`.

**FlashKDA integration** (see `official/flashkda/README.md`):

1. Install FlashKDA + `flash-linear-attention >= 0.5`.
2. Under `torch.inference_mode()`, FLA’s `chunk_kda` may auto-dispatch to
   `flash_kda.fwd` when shapes/dtypes/arch match (SM90+, head dim 128, etc.).
3. Opt out: env `FLA_FLASH_KDA=0`.
4. Kernel-faithful torch reference for correctness tests: `official/flashkda/torch_ref.py`
   (uses `tanh.approx` sigmoid, fp16acc GEMM, warp-style L2Norm) — **not** the same as
   our educational float32 recurrent reference (`NUMERICS.md`).

API sketch:

```text
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None)
```

K3 enables `use_gate_in_kernel`, `use_qk_l2norm_in_kernel`, `use_beta_sigmoid_in_kernel`,
`safe_gate` with `lower_bound=-5.0`, `transpose_state_layout=True`.

---

## 6. MoonEP (expert parallel) — conceptual

**Vendored overview:** `official/moonep/README.md`  
**Summary in:** `INFRA.md`

At K3 sparsity (896 experts, top-16), training/inference needs balanced EP:

- Contiguous symmetric-memory weight buffer `[E+B, H, H']` per projection
- Planner produces `cu_seqlens` selecting active expert rows
- Prefetch slots `B` hold remote experts; training typically `B = E/R`
- Dispatch → local GEMM → combine; grad reduce for prefetch slots in training

The educational LatentMoE forward (`reference/.../expert_ffn.py`) is the **single-rank
algorithmic** view (`ep_size=1`) matching `KimiSparseMoeBlock.moe_infer`.

---

## 7. Suggested reading order in `official/python/`

1. `config.json` + `configuration_kimi_k3.py` — shapes and layer schedule  
2. `SituAndMul`, `KimiRMSNorm`, `KimiMoEGate`, `KimiSparseMoeBlock`  
3. `KimiDeltaAttention`, then FLA `official/fla_kda/naive.py`  
4. `KimiMLAAttention` + `_apply_attn_res` + `KimiDecoderLayer`  
5. `KimiLinearModel.forward` (AttnRes loop)  
6. `KimiK3ForConditionalGeneration` (vision + merge)  
7. `encoding_k3.py` + `tokenization_kimi.py` (chat surface)

Then jump to the language port that matches how you learn best.
