/-!
# Definitions (paper §§primes)

Public predicates used throughout the uniqueness development.
Aligned with `paper/unique_even_prime.tex` (v3).
-/

namespace EvenPrime

/-- Prime (divisor form): `n > 1` and every divisor is `1` or `n`.
Paper: `def:prime`. -/
def IsPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d ∣ n → d = 1 ∨ d = n

/-- Composite: `n > 1` and not prime. Paper: `def:composite`. -/
def IsComposite (n : Nat) : Prop :=
  n > 1 ∧ ¬ IsPrime n

/-- Even: divisible by 2. Paper: `def:even`. -/
def IsEven (n : Nat) : Prop :=
  2 ∣ n

/-- Irreducible in `(ℕ≥1, ·)`: `n > 1` and every factorization has a factor `1`.
Paper: `rem:irreducible`. On `ℕ` this coincides with `IsPrime`. -/
def IsIrreducible (n : Nat) : Prop :=
  n > 1 ∧ ∀ a b : Nat, a * b = n → a = 1 ∨ b = 1

end EvenPrime
