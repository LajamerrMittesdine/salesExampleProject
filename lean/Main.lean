import EvenPrime.Tests
import EvenPrime.Spec
import EvenPrime.Grand

def main : IO UInt32 := do
  match EvenPrime.runTests with
  | none =>
      IO.println "OK: concrete even/prime checks passed"
      IO.println "OK: two primary uniqueness proofs typecheck"
      IO.println "OK: spec + grand certificates typecheck"
      IO.println "    TargetClaim: ∀ p, IsPrime p → IsEven p → p = 2"
      let _ := EvenPrime.Spec.spec_certificate
      let _ := EvenPrime.Grand.grand_certificate
      pure 0
  | some msg =>
      IO.eprintln s!"FAIL: {msg}"
      pure 1
