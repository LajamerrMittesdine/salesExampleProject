import EvenPrime.Prime
import EvenPrime.Proofs
import EvenPrime.Grand

/-!
# Specification certificate (v2)

Trusted reading surface + audits + anti-triviality.
Equivalences carefully separate uniqueness-only from existence+uniqueness.
-/

namespace EvenPrime
namespace Spec

/-- **Trusted reading surface.**
English: “Every even prime natural number equals 2.” (uniqueness-only) -/
def TargetClaim : Prop :=
  ∀ p : Nat, IsPrime p → IsEven p → p = 2

def UniqueEvenPrime : Prop :=
  IsPrime 2 ∧ IsEven 2 ∧ TargetClaim

def NoOtherEvenPrime : Prop :=
  ¬ ∃ p : Nat, IsPrime p ∧ IsEven p ∧ p ≠ 2

def EvensAboveTwoComposite : Prop :=
  ∀ n : Nat, n > 2 → IsEven n → ¬ IsPrime n

theorem TargetClaim_iff_NoOtherEvenPrime : TargetClaim ↔ NoOtherEvenPrime :=
  Grand.unique_iff_no_other

theorem TargetClaim_iff_EvensAboveTwoComposite :
    TargetClaim ↔ EvensAboveTwoComposite :=
  Grand.unique_iff_above

/-- Existence+uniqueness is TargetClaim plus the witness for 2. -/
theorem UniqueEvenPrime_iff :
    UniqueEvenPrime ↔ (IsPrime 2 ∧ IsEven 2 ∧ TargetClaim) := Iff.rfl

theorem proofI_proves_TargetClaim : TargetClaim :=
  fun _p hp he => even_prime_eq_two hp he

theorem proofII_proves_TargetClaim : TargetClaim :=
  fun _p hp he => even_prime_eq_two_by_factorization hp he

theorem TargetClaim_with_witness : UniqueEvenPrime :=
  ⟨two_is_prime, two_is_even, proofI_proves_TargetClaim⟩

/-! ## Anti-triviality -/

theorem not_all_even_primes_are_three :
    ¬ ∀ p : Nat, IsPrime p → IsEven p → p = 3 := by
  intro h
  have : 2 = 3 := h 2 two_is_prime two_is_even
  exact absurd this (by decide)

theorem exists_even_prime : ∃ p : Nat, IsPrime p ∧ IsEven p :=
  ⟨2, two_is_prime, two_is_even⟩

theorem exists_odd_prime : ∃ p : Nat, IsPrime p ∧ ¬ IsEven p := by
  refine ⟨3, three_is_prime, ?_⟩
  intro he
  have : 3 % 2 = 0 := (isEven_iff_mod_two 3).1 he
  exact absurd this (by decide)

theorem exists_even_composite : ∃ n : Nat, IsEven n ∧ IsComposite n :=
  ⟨4, ⟨2, rfl⟩, four_is_composite⟩

theorem spec_certificate :
    TargetClaim
    ∧ UniqueEvenPrime
    ∧ (TargetClaim ↔ NoOtherEvenPrime)
    ∧ (TargetClaim ↔ EvensAboveTwoComposite)
    ∧ (∀ n, IsEven n ↔ n % 2 = 0)
    ∧ (∀ n, IsPrime n ↔ IsIrreducible n)
    ∧ (∀ n, IsComposite n ↔
        n > 1 ∧ ∃ d k, n = d * k ∧ 1 < d ∧ d < n ∧ 1 < k ∧ k < n)
    ∧ (∃ p, IsPrime p ∧ IsEven p)
    ∧ (∃ p, IsPrime p ∧ ¬ IsEven p)
    ∧ IsEven 0
    ∧ ¬ IsEven 1
    ∧ ¬ ∀ p, IsPrime p → IsEven p → p = 3 :=
  ⟨ proofI_proves_TargetClaim
  , TargetClaim_with_witness
  , TargetClaim_iff_NoOtherEvenPrime
  , TargetClaim_iff_EvensAboveTwoComposite
  , isEven_iff_mod_two
  , isPrime_iff_isIrreducible
  , isComposite_iff_proper_factors
  , exists_even_prime
  , exists_odd_prime
  , zero_is_even
  , one_not_even
  , not_all_even_primes_are_three ⟩

#check TargetClaim
#check (proofI_proves_TargetClaim : TargetClaim)
#check (proofII_proves_TargetClaim : TargetClaim)
#check spec_certificate

end Spec
end EvenPrime
