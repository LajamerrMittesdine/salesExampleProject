import EvenPrime.Unique
import EvenPrime.AboveTwo
import EvenPrime.Basic

/-!
# Trusted reading surface (paper Spec)

Single vocabulary for the informal claim. All names here are the
English-facing API; proofs live in `Unique` / `AboveTwo`.
-/

namespace EvenPrime

/-- **Trusted reading surface (uniqueness-only).**
English: every even prime natural number equals 2.
Paper: `thm:grand`(C). Does *not* by itself assert that 2 is prime. -/
abbrev TargetClaim : Prop := EveryEvenPrimeEqTwo

/-- Existence + uniqueness. Paper: `thm:grand`(A) / Cor. unique. -/
abbrev UniqueEvenPrimeClaim : Prop := UniqueEvenPrime

/-- Negative uniqueness-only. Paper: `thm:grand`(D). -/
abbrev NoOtherEvenPrime : Prop := NoEvenPrimeNeTwo

/-- Above-two form (not-prime). Equivalent to `TargetClaim`. -/
abbrev EvensAboveTwoNotPrime : Prop :=
  ∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n

/-- Above-two form (composite). -/
abbrev EvensAboveTwoComposite : Prop :=
  ∀ n : Nat, n > 2 → IsEven n → IsComposite n

theorem targetClaim_proofI : TargetClaim :=
  everyEvenPrimeEqTwo

theorem targetClaim_proofII : TargetClaim :=
  fun _p hp he => eq_two_of_isPrime_of_isEven_factorization hp he

theorem targetClaim_iff_noOtherEvenPrime : TargetClaim ↔ NoOtherEvenPrime :=
  everyEvenPrimeEqTwo_iff_noEvenPrimeNeTwo

theorem targetClaim_iff_evensAboveTwoNotPrime :
    TargetClaim ↔ EvensAboveTwoNotPrime :=
  everyEvenPrimeEqTwo_iff_not_isPrime_of_gt_two_of_isEven

theorem evensAboveTwoComposite_proofI : EvensAboveTwoComposite :=
  fun _n hn he => isComposite_of_gt_two_of_isEven hn he

theorem evensAboveTwoComposite_proofII : EvensAboveTwoComposite :=
  fun _n hn he => isComposite_of_gt_two_of_isEven_factorization hn he

theorem uniqueEvenPrimeClaim_holds : UniqueEvenPrimeClaim :=
  uniqueEvenPrime

/-! ## Anti-triviality -/

theorem not_forall_eq_three_of_isPrime_of_isEven :
    ¬ ∀ p : Nat, IsPrime p → IsEven p → p = 3 := by
  intro h
  exact absurd (h 2 isPrime_two isEven_two) (by decide)

theorem exists_isPrime_isEven : ∃ p : Nat, IsPrime p ∧ IsEven p :=
  ⟨2, isPrime_two, isEven_two⟩

theorem exists_isPrime_not_isEven : ∃ p : Nat, IsPrime p ∧ ¬ IsEven p := by
  refine ⟨3, isPrime_three, ?_⟩
  intro he
  exact absurd ((isEven_iff_mod_two 3).1 he) (by decide)

theorem exists_isEven_isComposite : ∃ n : Nat, IsEven n ∧ IsComposite n :=
  ⟨4, ⟨2, rfl⟩, isComposite_four⟩

/-- Spec certificate: uniqueness, audits, anti-triviality, `∃!`. -/
theorem spec_certificate :
    TargetClaim
    ∧ UniqueEvenPrimeClaim
    ∧ ExistsUniqueEvenPrime
    ∧ (TargetClaim ↔ NoOtherEvenPrime)
    ∧ (TargetClaim ↔ EvensAboveTwoNotPrime)
    ∧ EvensAboveTwoComposite
    ∧ (∀ n, IsEven n ↔ n % 2 = 0)
    ∧ (∀ n, IsPrime n ↔ IsIrreducible n)
    ∧ (∀ n, IsComposite n ↔
        n > 1 ∧ ∃ d k, n = d * k ∧ 1 < d ∧ d < n ∧ 1 < k ∧ k < n)
    ∧ IsEven 0
    ∧ ¬ IsEven 1
    ∧ (∃ p, IsPrime p ∧ IsEven p)
    ∧ (∃ p, IsPrime p ∧ ¬ IsEven p)
    ∧ ¬ ∀ p, IsPrime p → IsEven p → p = 3 :=
  ⟨ targetClaim_proofI
  , uniqueEvenPrimeClaim_holds
  , existsUnique_even_prime
  , targetClaim_iff_noOtherEvenPrime
  , targetClaim_iff_evensAboveTwoNotPrime
  , evensAboveTwoComposite_proofII
  , isEven_iff_mod_two
  , isPrime_iff_isIrreducible
  , isComposite_iff_proper_factors
  , isEven_zero
  , not_isEven_one
  , exists_isPrime_isEven
  , exists_isPrime_not_isEven
  , not_forall_eq_three_of_isPrime_of_isEven ⟩

end EvenPrime
