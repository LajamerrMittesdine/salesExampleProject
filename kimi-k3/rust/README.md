# Rust port

Idiomatic Rust crate `kimi-k3-ref` mirroring the Python reference modules.

```bash
cargo test
```

Sources under `src/` (one module per op); integration tests in `tests/goldens.rs` load
`../fixtures/goldens.json`. Module names match `reference/python/kimi_k3_ref/*.py` and the
official mapping in `../docs/SOURCE_MAP.md`.
