import EvenPrime.Basic

/-!
# No prime less than two (paper `thm:below`)
-/

namespace EvenPrime

/-- Paper `thm:below`. -/
theorem not_isPrime_of_lt_two {n : Nat} (hn : n < 2) : ¬ IsPrime n := by
  intro hp
  have : 1 < n := hp.1
  omega

/-- A fortiori form. -/
theorem not_isPrime_of_lt_two_of_isEven {n : Nat}
    (hn : n < 2) (_he : IsEven n) : ¬ IsPrime n :=
  not_isPrime_of_lt_two hn

/-- Exhaustion presentation on `{0,1}`. -/
example {n : Nat} (hn : n < 2) (_he : IsEven n) : ¬ IsPrime n := by
  have hcases : n = 0 ∨ n = 1 := (lt_two_iff).1 hn
  cases hcases with
  | inl h0 => subst h0; exact not_isPrime_zero
  | inr h1 => subst h1; exact not_isPrime_one

end EvenPrime
