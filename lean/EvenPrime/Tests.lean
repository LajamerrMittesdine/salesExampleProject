import EvenPrime.Spec

/-!
# Concrete tests and regression examples (not part of the library API)
-/

namespace EvenPrime

/-- Computational check approximating `IsPrime` (no `d = 0` escape). -/
def checkPrime (n : Nat) : Bool :=
  decide (n > 1) &&
    (List.range (n + 1)).all fun d =>
      !(decide (d ∣ n)) || d == 1 || d == n

def checkEven (n : Nat) : Bool :=
  n % 2 == 0

/-- Soundness of the Bool checker on the tested shape. -/
theorem checkEven_iff_isEven (n : Nat) : checkEven n = true ↔ IsEven n := by
  simp [checkEven, isEven_iff_mod_two]

def runTests : Option String := Id.run do
  let mut failures : List String := []
  if !(checkEven 0) then failures := "0 should be even" :: failures
  if checkEven 1 then failures := "1 should be odd" :: failures
  if !(checkEven 2) then failures := "2 should be even" :: failures
  if !(checkPrime 2) then failures := "2 should be prime" :: failures
  if !(checkPrime 3) then failures := "3 should be prime" :: failures
  if checkPrime 1 then failures := "1 should not be prime" :: failures
  if checkPrime 4 then failures := "4 should not be prime" :: failures
  for n in [4, 6, 8, 10, 12, 14, 16, 18, 20, 100, 1000] do
    if !(checkEven n) then failures := s!"{n} should be even" :: failures
    if checkPrime n then failures := s!"{n} should not be prime" :: failures
  for n in [3, 5, 7, 11, 13] do
    if checkEven n then failures := s!"{n} should be odd" :: failures
    if !(checkPrime n) then failures := s!"{n} should be prime" :: failures
  match failures with
  | [] => none
  | fs => some (String.intercalate "; " fs.reverse)

/-- Fail elaboration if concrete tests fail. -/
def assertTests : IO Unit := do
  match runTests with
  | none => pure ()
  | some msg => throw <| IO.userError s!"EvenPrime tests failed: {msg}"

#eval assertTests

example : eq_two_of_isPrime_of_isEven isPrime_two isEven_two = rfl := rfl
example : eq_two_of_isPrime_of_isEven_factorization isPrime_two isEven_two = rfl := rfl
example : IsComposite 4 := isComposite_four
example : IsEven 0 := isEven_zero

end EvenPrime
