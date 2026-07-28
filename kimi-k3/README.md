# Kimi K3 Investigation Tree

Educational, source-faithful study of **Moonshot AI’s Kimi K3** (2.8T MoE, KDA + AttnRes + Stable LatentMoE + MoonViT-V2).

This folder vendors the **official Hugging Face Python modeling code**, synthesizes how it works, and provides **one-to-one float32 ports** of the algorithmic core into **C, C++, Rust, and Go**, checked against shared golden fixtures.

> Model weights are **not** included. Production GPU kernels (FlashKDA / FlashAttention-2 / MXFP4) are documented, not reimplemented.

## Quick start

```bash
# Verify every language against the same goldens
bash kimi-k3/scripts/verify_all.sh
```

## Learning path

1. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
2. Use [`docs/SOURCE_MAP.md`](docs/SOURCE_MAP.md) to jump from equations → official symbols → ports
3. Read official backbone: [`official/python/modeling_kimi_linear.py`](official/python/modeling_kimi_linear.py)
4. Read multimodal wrapper: [`official/python/modeling_kimi_k3.py`](official/python/modeling_kimi_k3.py)
5. Compare with educational core: [`reference/python/kimi_k3_ref/`](reference/python/kimi_k3_ref/)
6. Pick a language: [`c/`](c/), [`cpp/`](cpp/), [`rust/`](rust/), [`go/`](go/)

## Layout

| Path | Contents |
|------|----------|
| `official/python/` | Vendored HF `custom_code` (full modeling stack) |
| `official/reports/k3_tech_report.pdf` | Official technical report |
| `official/fla_kda/` | FLA naive KDA + gate references |
| `official/flashkda/` | FlashKDA README + torch reference |
| `official/moonep/` | Expert-parallel overview |
| `docs/` | Architecture, source map, numerics, infra |
| `reference/python/` | Dependency-free NumPy float32 core |
| `fixtures/goldens.json` | Shared golden tensors (seed 20260728) |
| `c/`, `cpp/`, `rust/`, `go/` | Idiomatic 1:1 ports of the reference core |

## Accuracy contract

Cross-language float32 parity on goldens:

- Elementwise / norms / AttnRes: `atol=rtol=1e-5`
- Matmul-heavy (KDA, FFN, MoE, MLA): `atol=rtol=1e-4`

See [`docs/NUMERICS.md`](docs/NUMERICS.md) for differences vs production bf16 / FlashKDA / unsorted `topk`.

## License / attribution

See [`LICENSE`](LICENSE) (Kimi K3 License) and [`NOTICE.md`](NOTICE.md).
