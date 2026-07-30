import EvenPrime.Prime
import EvenPrime.Proofs
import EvenPrime.Spec
import EvenPrime.Grand

namespace EvenPrime

def checkPrime (n : Nat) : Bool :=
  decide (n > 1) &&
    (List.range (n + 1)).all fun d =>
      !decide (d ∣ n) || d == 0 || d == 1 || d == n

def checkEven (n : Nat) : Bool :=
  n % 2 == 0

def runTests : Option String := Id.run do
  let mut failures : List String := []
  if !(checkEven 0) then failures := "0 should be even" :: failures
  if checkEven 1 then failures := "1 should be odd" :: failures
  if !(checkEven 2) then failures := "2 should be even" :: failures
  if !(checkPrime 2) then failures := "2 should be prime" :: failures
  if !(checkPrime 3) then failures := "3 should be prime" :: failures
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

#eval runTests

theorem all_primary_agree {p : Nat} (hp : IsPrime p) (he : IsEven p) : p = 2 :=
  even_prime_eq_two hp he

example : even_prime_eq_two two_is_prime two_is_even = rfl := rfl
example : even_prime_eq_two_by_factorization two_is_prime two_is_even = rfl := rfl

end EvenPrime
