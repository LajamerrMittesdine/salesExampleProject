import EvenPrime.AboveTwo
import EvenPrime.BelowTwo

/-!
# Uniqueness packaging (paper `thm:grand`)

Separates uniqueness-only from existence+uniqueness.
The reverse direction of the expansive ↔ unique iff derives both sides
from the uniqueness hypothesis alone (no smuggled BelowTwo proof).
-/

namespace EvenPrime

/-- Uniqueness-only. Paper `thm:grand`(C) / Spec `TargetClaim`. -/
def EveryEvenPrimeEqTwo : Prop :=
  ∀ p : Nat, IsPrime p → IsEven p → p = 2

/-- Existence + uniqueness. Paper `thm:grand`(A). -/
def UniqueEvenPrime : Prop :=
  IsPrime 2 ∧ IsEven 2 ∧ EveryEvenPrimeEqTwo

/-- Negative uniqueness-only. Paper `thm:grand`(D). -/
def NoEvenPrimeNeTwo : Prop :=
  ¬ ∃ p : Nat, IsPrime p ∧ IsEven p ∧ p ≠ 2

/-- Expansive split. Paper `thm:grand`(B). -/
def EvenPrimeSplit : Prop :=
  (∀ n : Nat, n < 2 → IsEven n → ¬ IsPrime n)
  ∧ (∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n)
  ∧ IsPrime 2
  ∧ IsEven 2

/-! ## The picture holds -/

theorem evenPrimeSplit_holds : EvenPrimeSplit :=
  ⟨ fun _n hn he => not_isPrime_of_lt_two_of_isEven hn he
  , fun _n hn he => not_isPrime_of_gt_two_of_isEven hn he
  , isPrime_two
  , isEven_two ⟩

/-! ## Uniqueness-only equivalences -/

theorem everyEvenPrimeEqTwo_iff_noEvenPrimeNeTwo :
    EveryEvenPrimeEqTwo ↔ NoEvenPrimeNeTwo := by
  constructor
  · intro H ⟨p, hp, he, hne⟩
    exact hne (H p hp he)
  · intro H p hp he
    by_cases hne : p = 2
    · exact hne
    · exact False.elim (H ⟨p, hp, he, hne⟩)

theorem everyEvenPrimeEqTwo_iff_not_isPrime_of_gt_two_of_isEven :
    EveryEvenPrimeEqTwo ↔
      (∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n) :=
  not_isPrime_of_gt_two_of_isEven_iff_eq_two_of_isPrime_of_isEven.symm

/-! ## From split to uniqueness -/

theorem everyEvenPrimeEqTwo_of_evenPrimeSplit
    (H : EvenPrimeSplit) : EveryEvenPrimeEqTwo := by
  obtain ⟨hBelow, hAbove, _, _⟩ := H
  intro p hp he
  by_cases hlt : p < 2
  · exact False.elim (hBelow p hlt he hp)
  · by_cases hgt : p > 2
    · exact False.elim (hAbove p hgt he hp)
    · omega

theorem everyEvenPrimeEqTwo : EveryEvenPrimeEqTwo :=
  everyEvenPrimeEqTwo_of_evenPrimeSplit evenPrimeSplit_holds

theorem uniqueEvenPrime : UniqueEvenPrime :=
  ⟨isPrime_two, isEven_two, everyEvenPrimeEqTwo⟩

/-- Existence and uniqueness packaging (stdlib has no unique-exists sugar).
Paper Cor. unique. -/
def ExistsUniqueEvenPrime : Prop :=
  ∃ p : Nat, (IsPrime p ∧ IsEven p) ∧ ∀ y : Nat, IsPrime y → IsEven y → y = p

theorem existsUnique_even_prime : ExistsUniqueEvenPrime :=
  ⟨2, ⟨isPrime_two, isEven_two⟩, fun y hy_p hy_e => everyEvenPrimeEqTwo y hy_p hy_e⟩

/-! ## Expansive ↔ existence+uniqueness (fixed reverse) -/

theorem evenPrimeSplit_iff_uniqueEvenPrime :
    EvenPrimeSplit ↔ UniqueEvenPrime := by
  constructor
  · intro H
    exact ⟨H.2.2.1, H.2.2.2, everyEvenPrimeEqTwo_of_evenPrimeSplit H⟩
  · intro ⟨h2p, h2e, H⟩
    -- Derive both sides from H alone (do not import BelowTwo/AboveTwo proofs).
    refine ⟨?below, ?above, h2p, h2e⟩
    · intro n hn he hp
      have : n = 2 := H n hp he
      omega
    · intro n hn he hp
      have : n = 2 := H n hp he
      omega

theorem uniqueEvenPrime_of_everyEvenPrimeEqTwo_of_witness
    (H : EveryEvenPrimeEqTwo) : UniqueEvenPrime :=
  ⟨isPrime_two, isEven_two, H⟩

end EvenPrime
