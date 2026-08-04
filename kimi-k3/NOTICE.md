# Notices and Attribution

This `kimi-k3/` tree investigates **Kimi K3** from Moonshot AI for educational purposes.

## Moonshot AI — Kimi K3

- Copyright (c) 2026 Moonshot AI
- Licensed under the **Kimi K3 License** (see `LICENSE` and `official/python/LICENSE`)
- Official model page: https://huggingface.co/moonshotai/Kimi-K3
- Official repository: https://github.com/MoonshotAI/Kimi-K3
- Technical blog: https://www.kimi.com/blog/kimi-k3

Vendored files under `official/python/` and `official/reports/` are copied from those sources.

## DeepSeek-AI — Multi-head Latent Attention / MoE patterns

Portions of `official/python/modeling_kimi_linear.py` are adapted from DeepSeek-V3
(`modeling_deepseek.py`) and are licensed under the **Apache License, Version 2.0**,
as stated in that file’s header. They have been extensively modified for Kimi Linear / K3.

## flash-linear-attention (FLA)

- Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li, and contributors
- MIT License
- https://github.com/fla-org/flash-linear-attention
- Vendored educational copies: `official/fla_kda/`

## FlashKDA

- Copyright (c) Moonshot AI
- MIT License
- https://github.com/MoonshotAI/FlashKDA
- Vendored: `official/flashkda/torch_ref.py`, README

## MoonEP

- https://github.com/MoonshotAI/MoonEP
- Documentation excerpt under `official/moonep/`

## Educational ports

The `reference/`, `c/`, `cpp/`, `rust/`, and `go/` directories implement a **float32
mathematical reference** of core Kimi K3 algorithms for learning. They are not the
production inference stack and are not bit-identical to GPU kernels (FlashKDA,
FlashAttention-2) or MXFP4 quantized weights.
