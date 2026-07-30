import Lake
open Lake DSL

package «EvenPrime»

lean_lib «EvenPrime»

@[default_target, test_driver]
lean_exe «even_prime_tests» where
  root := `Main
