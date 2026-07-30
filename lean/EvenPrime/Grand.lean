import EvenPrime.Prime
import EvenPrime.Proofs
import EvenPrime.BelowTwo

/-!
# Grand uniqueness theorem (v2: equivalences repaired)

Uniqueness-only forms are equivalent to each other.
Existence+uniqueness requires the witness that 2 is an even prime.
-/

namespace EvenPrime
namespace Grand

/-- Expansive picture: both sides + witness. -/
def ExpansiveEvenPrimePicture : Prop :=
  BelowTwo.NoEvenPrimeBelowTwo
  ∧ (∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n)
  ∧ IsPrime 2
  ∧ IsEven 2

/-- Uniqueness-only claim (no existence). -/
def EveryEvenPrimeIsTwo : Prop :=
  ∀ p : Nat, IsPrime p → IsEven p → p = 2

/-- Existence + uniqueness. -/
def UniqueEvenPrimeIsTwo : Prop :=
  IsPrime 2 ∧ IsEven 2 ∧ EveryEvenPrimeIsTwo

/-- Negative packaging of uniqueness-only. -/
def NoEvenPrimeOtherThanTwo : Prop :=
  ¬ ∃ p : Nat, IsPrime p ∧ IsEven p ∧ p ≠ 2

/-! ## The expansive picture holds -/

theorem expansive_holds : ExpansiveEvenPrimePicture :=
  ⟨ BelowTwo.no_even_prime_lt_two
  , fun _n hn he => even_gt_two_not_prime hn he
  , two_is_prime
  , two_is_even ⟩

/-! ## Uniqueness-only equivalences -/

theorem unique_iff_no_other :
    EveryEvenPrimeIsTwo ↔ NoEvenPrimeOtherThanTwo := by
  constructor
  · intro H ⟨p, hp, he, hne⟩
    exact hne (H p hp he)
  · intro H p hp he
    by_cases hne : p = 2
    · exact hne
    · exact False.elim (H ⟨p, hp, he, hne⟩)

theorem unique_iff_above :
    EveryEvenPrimeIsTwo ↔ (∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n) :=
  above_iff_unique.symm

/-! ## From expansive picture to uniqueness -/

theorem every_even_prime_is_two_of_expansive
    (H : ExpansiveEvenPrimePicture) : EveryEvenPrimeIsTwo := by
  obtain ⟨hBelow, hAbove, _, _⟩ := H
  intro p hp he
  by_cases hlt : p < 2
  · exact False.elim (hBelow p hlt he hp)
  · by_cases hgt : p > 2
    · exact False.elim (hAbove p hgt he hp)
    · omega

theorem every_even_prime_is_two : EveryEvenPrimeIsTwo :=
  every_even_prime_is_two_of_expansive expansive_holds

theorem unique_even_prime_is_two : UniqueEvenPrimeIsTwo :=
  ⟨two_is_prime, two_is_even, every_even_prime_is_two⟩

/-! ## Expansive ↔ existence+uniqueness (not ↔ uniqueness-only alone) -/

theorem expansive_iff_unique_with_witness :
    ExpansiveEvenPrimePicture ↔ UniqueEvenPrimeIsTwo := by
  constructor
  · intro H
    exact ⟨H.2.2.1, H.2.2.2, every_even_prime_is_two_of_expansive H⟩
  · intro ⟨h2p, h2e, H⟩
    exact ⟨ BelowTwo.no_even_prime_lt_two
          , fun n hn he hp => by
              have : n = 2 := H n hp he
              omega
          , h2p
          , h2e ⟩

/-- Recovering the expansive picture from uniqueness-only needs the witness. -/
theorem expansive_of_unique_plus_witness
    (H : EveryEvenPrimeIsTwo) (h2p : IsPrime 2) (h2e : IsEven 2) :
    ExpansiveEvenPrimePicture :=
  (expansive_iff_unique_with_witness).2 ⟨h2p, h2e, H⟩

theorem grand_certificate :
    ExpansiveEvenPrimePicture
    ∧ UniqueEvenPrimeIsTwo
    ∧ EveryEvenPrimeIsTwo
    ∧ NoEvenPrimeOtherThanTwo
    ∧ (ExpansiveEvenPrimePicture ↔ UniqueEvenPrimeIsTwo)
    ∧ (EveryEvenPrimeIsTwo ↔ NoEvenPrimeOtherThanTwo)
    ∧ (EveryEvenPrimeIsTwo ↔ (∀ n, n > 2 → IsEven n → ¬ IsPrime n)) :=
  ⟨ expansive_holds
  , unique_even_prime_is_two
  , every_even_prime_is_two
  , unique_iff_no_other.1 every_even_prime_is_two
  , expansive_iff_unique_with_witness
  , unique_iff_no_other
  , unique_iff_above ⟩

#check (every_even_prime_is_two : EveryEvenPrimeIsTwo)
#check grand_certificate

end Grand
end EvenPrime
