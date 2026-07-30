import EvenPrime.Prime

/-!
# Two primary proofs that every even `n > 2` is not prime

Following the v2 paper: there are two genuine arithmetic ideas.
Presentation variants are recorded separately and are **not** claimed
to be independent.
-/

namespace EvenPrime

/-! ## Primary Proof I — definitional (divisor clause on 2) -/

/-- If `p` is prime and even, then `p = 2`. -/
theorem even_prime_eq_two {p : Nat} (hp : IsPrime p) (he : IsEven p) : p = 2 := by
  have h2 : 2 ∣ p := he
  have hdiv := hp.2 2 h2
  cases hdiv with
  | inl h => exact absurd h (by decide)
  | inr h => exact h.symm

/-- Every even natural greater than 2 fails to be prime. -/
theorem even_gt_two_not_prime {n : Nat} (hn : n > 2) (he : IsEven n) : ¬ IsPrime n := by
  intro hp
  have : n = 2 := even_prime_eq_two hp he
  omega

/-- Even `n > 2` is composite. -/
theorem even_gt_two_composite {n : Nat} (hn : n > 2) (he : IsEven n) : IsComposite n :=
  ⟨by omega, even_gt_two_not_prime hn he⟩

/-! ## Primary Proof II — factorization / reducibility -/

/-- Factorization proof: even `n > 2` has a proper factor, hence is not prime. -/
theorem even_gt_two_not_prime_by_factorization {n : Nat}
    (hn : n > 2) (he : IsEven n) : ¬ IsPrime n := by
  intro hp
  obtain ⟨k, hk, hk_gt, hk_lt⟩ := halve_even_gt_two hn he
  have hk_dvd : k ∣ n := ⟨2, by rw [hk, Nat.mul_comm]⟩
  have := hp.2 k hk_dvd
  cases this with
  | inl h => omega
  | inr h => omega

theorem even_prime_eq_two_by_factorization {p : Nat}
    (hp : IsPrime p) (he : IsEven p) : p = 2 := by
  by_cases h : p > 2
  · exact absurd hp (even_gt_two_not_prime_by_factorization h he)
  · have : p > 1 := hp.1
    omega

/-! ## Equivalence of the two target formulations -/

/-- The “above 2” form ↔ the uniqueness form (no existence claim). -/
theorem above_iff_unique :
    (∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n) ↔
      (∀ p : Nat, IsPrime p → IsEven p → p = 2) := by
  constructor
  · intro h p hp he
    by_cases ht : p > 2
    · exact absurd hp (h p ht he)
    · have : p > 1 := hp.1
      omega
  · intro h n hn he hp
    have : n = 2 := h n hp he
    omega

/-! ## Presentation variants (not independent proofs) -/

/-- Variant of Proof I in contradiction style. -/
theorem even_prime_ne_two_false {p : Nat}
    (hp : IsPrime p) (he : IsEven p) (hne : p ≠ 2) : False :=
  hne (even_prime_eq_two hp he)

/-- Variant of Proof I via `prime ∣ prime ⇒ equal`. -/
theorem even_prime_eq_two_via_prime_dvd {p : Nat}
    (hp : IsPrime p) (he : IsEven p) : p = 2 :=
  ((prime_dvd_prime_iff_eq two_is_prime hp).1 he).symm

/-- Well-ordering exists, but applying it to even primes > 2 does **not**
yield an independent arithmetic argument: the contradiction is exactly
Proof II on any witness (minimality is idle). -/
theorem exists_least {P : Nat → Prop} (h : ∃ n, P n) :
    ∃ m, P m ∧ ∀ k < m, ¬ P k := by
  classical
  obtain ⟨n0, hn0⟩ := h
  have : ∀ n, P n → ∃ m, P m ∧ ∀ k < m, ¬ P k := by
    intro n
    induction n using Nat.strongRecOn with
    | ind n ih =>
      intro hn
      by_cases hsmall : ∃ k < n, P k
      · obtain ⟨k, hklt, hkP⟩ := hsmall
        exact ih k hklt hkP
      · refine ⟨n, hn, ?_⟩
        intro k hk hkP
        exact hsmall ⟨k, hk, hkP⟩
  exact this n0 hn0

/-- Nonexistence of even primes > 2, via Proof II (WOP packaging is optional). -/
theorem no_even_prime_gt_two : ¬ ∃ n : Nat, IsEven n ∧ n > 2 ∧ IsPrime n := by
  intro ⟨n, he, hn, hp⟩
  exact even_gt_two_not_prime_by_factorization hn he hp

/-! ## Canonical uniqueness export (Proof I) -/

/-- Canonical statement: every even prime equals 2. -/
theorem unique_even_prime {p : Nat} (hp : IsPrime p) (he : IsEven p) : p = 2 :=
  even_prime_eq_two hp he

/-! ## Regression examples -/

example : IsPrime 2 ∧ IsEven 2 := two_is_even_prime
example : IsEven 0 := zero_is_even
example : ¬ IsEven 1 := one_not_even
example : IsPrime 3 := three_is_prime
example : IsComposite 4 := four_is_composite
example : ¬ IsPrime 4 := even_gt_two_not_prime (by decide) ⟨2, rfl⟩
example : ¬ IsPrime 6 := even_gt_two_not_prime_by_factorization (by decide) ⟨3, rfl⟩

/-- The two primary proofs both establish the same proposition. -/
example {p : Nat} (hp : IsPrime p) (he : IsEven p) :
    p = 2 ∧ p = 2 :=
  ⟨even_prime_eq_two hp he, even_prime_eq_two_by_factorization hp he⟩

end EvenPrime
