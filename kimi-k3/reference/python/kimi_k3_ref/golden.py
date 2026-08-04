"""Helpers to load / compare shared golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures"
GOLDENS_PATH = FIXTURES_DIR / "goldens.json"


def load_goldens(path: Path | None = None) -> dict[str, Any]:
    p = path or GOLDENS_PATH
    return json.loads(p.read_text())


def as_f32(case: dict[str, Any], key: str, shape: list[int] | tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(case[key], dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def as_i64(case: dict[str, Any], key: str, shape: list[int] | tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(case[key], dtype=np.int64)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def assert_close(
    got: np.ndarray,
    want: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    name: str = "",
) -> None:
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol, err_msg=name)
