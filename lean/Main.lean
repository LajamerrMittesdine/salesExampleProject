import EvenPrime.Tests
import EvenPrime.Spec

/-- CLI test driver: exit 0 iff concrete checks pass. Theorems are checked at build. -/
def main : IO UInt32 := do
  match EvenPrime.runTests with
  | none =>
      IO.println "OK: concrete even/prime checks passed"
      IO.println "OK: TargetClaim / UniqueEvenPrime / ExistsUnique certificates typecheck"
      IO.println "    TargetClaim: ∀ p, IsPrime p → IsEven p → p = 2"
      IO.println "    UniqueEvenPrime: IsPrime 2 ∧ IsEven 2 ∧ TargetClaim"
      let _ := EvenPrime.spec_certificate
      let _ := EvenPrime.existsUnique_even_prime
      pure 0
  | some msg =>
      IO.eprintln s!"FAIL: {msg}"
      pure 1
