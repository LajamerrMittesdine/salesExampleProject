import EvenPrime.Basic

/-!
# Even numbers greater than two (paper `thm:above`)

Two primary proofs:
* **Proof I** — definitional (divisor clause on 2)
* **Proof II** — factorization / reducibility
-/

namespace EvenPrime

/-! ## Proof I — definitional -/

/-- Canonical uniqueness form. Paper Proof I / `cor:unique-form`.
Lean name: `eq_two_of_isPrime_of_isEven`. -/
theorem eq_two_of_isPrime_of_isEven {p : Nat}
    (hp : IsPrime p) (he : IsEven p) : p = 2 := by
  have hdiv := hp.2 2 he
  cases hdiv with
  | inl h => exact absurd h (by decide)
  | inr h => exact h.symm

/-- Paper `thm:above` (not-prime form), via Proof I. -/
theorem not_isPrime_of_gt_two_of_isEven {n : Nat}
    (hn : n > 2) (he : IsEven n) : ¬ IsPrime n := by
  intro hp
  have : n = 2 := eq_two_of_isPrime_of_isEven hp he
  omega

/-- Paper `thm:above` (composite form), via Proof I. -/
theorem isComposite_of_gt_two_of_isEven {n : Nat}
    (hn : n > 2) (he : IsEven n) : IsComposite n :=
  ⟨by omega, not_isPrime_of_gt_two_of_isEven hn he⟩

/-! ## Proof II — factorization -/

/-- Paper Proof II: even `n > 2` has a proper factor, hence not prime. -/
theorem not_isPrime_of_gt_two_of_isEven_factorization {n : Nat}
    (hn : n > 2) (he : IsEven n) : ¬ IsPrime n := by
  intro hp
  obtain ⟨k, hk, hk_gt, hk_lt⟩ := exists_eq_two_mul_of_gt_two_of_isEven hn he
  have hk_dvd : k ∣ n := ⟨2, by rw [hk, Nat.mul_comm]⟩
  have := hp.2 k hk_dvd
  cases this with
  | inl h => omega
  | inr h => omega

/-- Paper Proof II concludes composite. -/
theorem isComposite_of_gt_two_of_isEven_factorization {n : Nat}
    (hn : n > 2) (he : IsEven n) : IsComposite n := by
  obtain ⟨k, hk, hk_gt, hk_lt⟩ := exists_eq_two_mul_of_gt_two_of_isEven hn he
  refine (isComposite_iff_proper_factors n).2 ⟨by omega, ?_⟩
  exact ⟨2, k, hk, by decide, by omega, hk_gt, hk_lt⟩

theorem eq_two_of_isPrime_of_isEven_factorization {p : Nat}
    (hp : IsPrime p) (he : IsEven p) : p = 2 := by
  by_cases h : p > 2
  · exact absurd hp (not_isPrime_of_gt_two_of_isEven_factorization h he)
  · have : 1 < p := hp.1
    omega

/-! ## Equivalence of the two formulations (uniqueness-only) -/

/-- Paper `lem:above-iff-unique`. Neither side asserts that 2 is prime. -/
theorem not_isPrime_of_gt_two_of_isEven_iff_eq_two_of_isPrime_of_isEven :
    (∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n) ↔
      (∀ p : Nat, IsPrime p → IsEven p → p = 2) := by
  constructor
  · intro h p hp he
    by_cases ht : p > 2
    · exact absurd hp (h p ht he)
    · have : 1 < p := hp.1
      omega
  · intro h n hn he hp
    have : n = 2 := h n hp he
    omega

/-! ## Presentation variants (not independent; not part of the public spine) -/

/-- Contradiction packaging of Proof I. -/
private theorem eq_two_of_isPrime_of_isEven_contradiction {p : Nat}
    (hp : IsPrime p) (he : IsEven p) (hne : p ≠ 2) : False :=
  hne (eq_two_of_isPrime_of_isEven hp he)

/-- Via `prime ∣ prime`; depends on `isPrime_two` (extra vs Proof I). -/
private theorem eq_two_of_isPrime_of_isEven_via_dvd {p : Nat}
    (hp : IsPrime p) (he : IsEven p) : p = 2 :=
  (eq_of_isPrime_of_dvd isPrime_two hp he).symm

end EvenPrime
