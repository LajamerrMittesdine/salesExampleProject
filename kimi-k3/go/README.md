# Go port

Idiomatic Go module with package `ref`.

```bash
go test ./ref/ -count=1
```

Ops live in `ref/ops.go` (`RMSNorm`, `SituAndMul`, `MoEGate`, `KDARecurrent`, …);
goldens in `ref/goldens_test.go`. See `../docs/SOURCE_MAP.md` for official HF/FLA counterparts.
