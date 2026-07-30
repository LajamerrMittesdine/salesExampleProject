import EvenPrime.Prime
import EvenPrime.Proofs

/-!
# No prime (hence no even prime) strictly less than 2
-/

namespace EvenPrime
namespace BelowTwo

def NoPrimeBelowTwo : Prop :=
  ∀ n : Nat, n < 2 → ¬ IsPrime n

def NoEvenPrimeBelowTwo : Prop :=
  ∀ n : Nat, n < 2 → IsEven n → ¬ IsPrime n

theorem no_prime_lt_two : NoPrimeBelowTwo := by
  intro n hn hp
  have : n > 1 := hp.1
  omega

theorem no_even_prime_lt_two : NoEvenPrimeBelowTwo := by
  intro n hn _he hp
  exact no_prime_lt_two n hn hp

/-- Exhaustion on `{0,1}` (presentation variant). -/
theorem no_even_prime_lt_two_by_exhaustion : NoEvenPrimeBelowTwo := by
  intro n hn _he hp
  have hcases : n = 0 ∨ n = 1 := (lt_two_iff n).1 hn
  cases hcases with
  | inl h0 => subst h0; exact zero_not_prime hp
  | inr h1 => subst h1; exact one_not_prime hp

#check (no_even_prime_lt_two : NoEvenPrimeBelowTwo)

end BelowTwo
end EvenPrime
