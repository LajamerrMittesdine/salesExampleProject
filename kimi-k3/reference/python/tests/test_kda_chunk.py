"""Chunk KDA should match recurrent KDA on the same inputs (float32)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kimi_k3_ref import kda_chunk, kda_lowerbound_gate, kda_recurrent


def test_chunk_matches_recurrent():
    rng = np.random.default_rng(7)
    b, t, h, d = 1, 8, 2, 4
    q = rng.standard_normal((b, t, h, d), dtype=np.float32)
    k = rng.standard_normal((b, t, h, d), dtype=np.float32)
    v = rng.standard_normal((b, t, h, d), dtype=np.float32)
    g_raw = rng.standard_normal((b, t, h, d), dtype=np.float32)
    a_log = rng.uniform(0.5, 2.0, size=(h,)).astype(np.float32)
    dt = rng.standard_normal((h, d), dtype=np.float32) * 0.05
    g = kda_lowerbound_gate(g_raw, a_log, dt, lower_bound=-5.0)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((b, t, h), dtype=np.float32)))).astype(
        np.float32
    )

    o_r, s_r = kda_recurrent(q, k, v, g, beta, l2norm_qk=False)
    o_c, s_c = kda_chunk(q, k, v, g, beta, chunk_size=4, l2norm_qk=False)
    np.testing.assert_allclose(o_c, o_r, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(s_c, s_r, atol=1e-4, rtol=1e-4)
