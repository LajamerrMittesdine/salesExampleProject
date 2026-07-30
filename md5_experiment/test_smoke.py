"""Lightweight smoke tests for the MD5 experiment pipeline."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from md5_experiment.config import ExperimentConfig
from md5_experiment.data import (
    build_experiment_data,
    encode_plaintext_onehot,
    enumerate_plaintexts,
    md5_bits,
    md5_hex,
)
from md5_experiment.models import build_forward_model, build_inverse_model


class DataTests(unittest.TestCase):
    def test_md5_matches_stdlib(self):
        self.assertEqual(md5_hex("000"), hashlib.md5(b"000").hexdigest())

    def test_md5_bits_length(self):
        self.assertEqual(md5_bits("42").shape, (128,))

    def test_enumerate_size(self):
        plain = enumerate_plaintexts("01", 3)
        self.assertEqual(len(plain), 8)
        self.assertEqual(plain[0], "000")

    def test_split_no_leakage(self):
        cfg = ExperimentConfig(alphabet="0123456789", plaintext_length=2, seed=0)
        data = build_experiment_data(cfg)
        train, val, test = (
            set(data.train.plaintexts),
            set(data.val.plaintexts),
            set(data.test.plaintexts),
        )
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(len(train | val | test), cfg.space_size)

    def test_onehot_roundtrip_shape(self):
        v = encode_plaintext_onehot("09", "0123456789")
        self.assertEqual(v.shape, (20,))
        self.assertEqual(float(v.sum()), 2.0)


class ModelTests(unittest.TestCase):
    def test_forward_shape(self):
        m = build_forward_model(40, 128, (32,), 0.0)
        y = m(torch.zeros(2, 40))
        self.assertEqual(tuple(y.shape), (2, 128))
        self.assertTrue(torch.all((y >= 0) & (y <= 1)))

    def test_inverse_shape(self):
        m = build_inverse_model(128, 4, 10, (32,), 0.0)
        y = m(torch.zeros(2, 128))
        self.assertEqual(tuple(y.shape), (2, 40))


class MemorizationSanity(unittest.TestCase):
    def test_tiny_inverse_can_overfit(self):
        """Pipeline sanity: a small MLP must be able to memorize a few pairs."""
        cfg = ExperimentConfig(
            alphabet="0123456789",
            plaintext_length=3,
            hidden_dims=(256, 256),
            dropout=0.0,
            weight_decay=0.0,
            learning_rate=3e-3,
            seed=1,
        )
        data = build_experiment_data(cfg)
        x = torch.from_numpy(data.train.y_bits[:32])
        y = torch.from_numpy(data.train.y_chars[:32])
        model = build_inverse_model(128, 3, 10, cfg.hidden_dims, 0.0)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(120):
            opt.zero_grad()
            logits = model(x).view(-1, 3, 10)
            loss = loss_fn(logits.reshape(-1, 10), y.reshape(-1))
            loss.backward()
            opt.step()
        pred = model(x).view(-1, 3, 10).argmax(dim=-1)
        exact = float((pred == y).all(dim=1).float().mean())
        self.assertGreaterEqual(exact, 0.99)


if __name__ == "__main__":
    unittest.main()
