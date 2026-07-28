#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Python reference =="
python3 scripts/generate_goldens.py >/dev/null
python3 -m pytest -q reference/python/tests

echo "== Rust =="
(cd rust && cargo test --quiet)

echo "== Go =="
(cd go && go test ./ref/ -count=1)

echo "== C++ =="
(cd cpp && make -s test)

echo "== C =="
(cd c && make -s test)

echo
echo "All language golden suites passed."
