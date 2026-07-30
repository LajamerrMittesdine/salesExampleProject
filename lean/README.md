# EvenPrime (Lean 4 companion)

Machine-checked layer for the paper
*On the Uniqueness of the Even Prime* (`../paper/unique_even_prime.tex`).

## Claim

**TargetClaim** (uniqueness-only):

```text
∀ p : Nat, IsPrime p → IsEven p → p = 2
```

**UniqueEvenPrime** (existence + uniqueness):

```text
IsPrime 2 ∧ IsEven 2 ∧ TargetClaim
```

also packaged as `∃! p, IsPrime p ∧ IsEven p`.

## Trust boundary

| Paper | Lean |
|---|---|
| Peano axioms → recursion → `+`/`·` → order | kernel `Nat` (assumed) |
| Divisibility / primes / uniqueness | proved here (stdlib only, no Mathlib) |

Lean does **not** reconstruct ℕ from Peano. It checks the uniqueness argument
relative to Lean’s `Nat`, including Proof I / Proof II and the Spec certificate.

## Build & test

```bash
lake build
lake test          # runs even_prime_tests
# or:
lake exe even_prime_tests
```

Toolchain: see `lean-toolchain` (Lean 4.16.0).

## Module map

| Module | Role |
|---|---|
| `Defs` | `IsPrime`, `IsComposite`, `IsEven`, `IsIrreducible` |
| `Basic` | toolkit, `isPrime_two`, examples, characterizations |
| `AboveTwo` | Proof I (definitional), Proof II (factorization) |
| `BelowTwo` | no prime `< 2` |
| `Unique` | grand packaging, `∃!`, expansive ↔ unique |
| `Spec` | trusted reading surface + certificate |
| `Tests` | Bool smoke tests (not re-exported by the lib root) |

## Paper ↔ Lean names

| Paper | Lean |
|---|---|
| `def:prime` | `IsPrime` |
| `def:composite` | `IsComposite` |
| `def:even` | `IsEven` |
| `prop:two` | `isPrime_two`, `isEven_two` |
| Proof I | `eq_two_of_isPrime_of_isEven` |
| Proof II | `not_isPrime_of_gt_two_of_isEven_factorization` / `isComposite_of_gt_two_of_isEven_factorization` |
| `thm:below` | `not_isPrime_of_lt_two` |
| `thm:grand`(A) | `UniqueEvenPrime` |
| `thm:grand`(B) | `EvenPrimeSplit` |
| `thm:grand`(C) | `EveryEvenPrimeEqTwo` / `TargetClaim` |
| `thm:grand`(D) | `NoEvenPrimeNeTwo` |
| Cor. unique | `existsUnique_even_prime` |
