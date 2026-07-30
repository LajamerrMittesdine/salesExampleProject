# MD5 Learnability Experiment

A small scientific experiment: can a neural network **learn the structure of MD5**
well enough to generalize beyond memorized `(plaintext → hash)` pairs — and, if
so, does that yield any useful prior for recovering inputs?

MD5 is cryptographically one-way. This project does **not** claim a break. It
asks a narrower, measurable question under a **tiny enumerable plaintext space**.

## What we measure

| Probe | Question |
| --- | --- |
| **Memorization probe** | Can the net fit a tiny subset of pairs as a lookup table? Does that transfer? |
| **Forward model** | Given a novel plaintext, can the net predict MD5 bits better than chance (~50%)? |
| **Avalanche probe** | When one input symbol flips, does the model flip ~50% of hash bits like real MD5? |
| **Inverse model** | Given a novel hash, can the net recover the held-out plaintext? |
| **Search acceleration** | Does the inverse model rank the true plaintext far above a random guess in the space? |

**Generalization** means strong held-out (test) performance.
**Memorization** means high train / near-chance test — the usual outcome for MD5.

## Setup

```bash
pip install -r md5_experiment/requirements.txt
```

## Run

Full default experiment (4-digit plaintexts, space size 10 000):

```bash
python -m md5_experiment.run_experiment
```

Quick smoke test:

```bash
python -m md5_experiment.run_experiment --quick
```

Useful flags: `--epochs`, `--length`, `--alphabet`, `--output-dir`.

## Design choices

- **Constrained inputs**: fixed-length digit strings (default length 4). The full
  space is enumerable, so train/val/test can be split by plaintext identity with
  no leakage from rainbow tables or web scrapes.
- **Representations**: plaintext as position-wise one-hots; MD5 as a 128-bit vector.
- **Models**: plain MLPs — enough capacity to memorize a small table, which makes
  a memorization-vs-generalization contrast meaningful.
- **Metrics**: bit/char accuracy, exact match, avalanche statistics, and mean
  rank of the true preimage under the model’s scoring.

## Interpreting results

- Test bit accuracy ≈ 0.5 and inverse exact-match ≈ `1/space_size` ⇒ no useful
  learned structure.
- High train exact-match + near-chance test ⇒ memorization only.
- Mean preimage rank much lower than the random mean rank ⇒ the model would
  accelerate search *within this toy space* (still not a general MD5 break).

Artifacts land in `md5_experiment/artifacts/` (`report.json`, `summary.txt`, curves).

## Scientific scope

This studies **learnability / generalization of a one-way function under severe
input constraints**. It is not a cryptanalytic attack on unconstrained MD5, nor
a password-cracking tool.
