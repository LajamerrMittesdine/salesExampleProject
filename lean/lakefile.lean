import Lake
open Lake DSL

package «EvenPrime»

@[default_target]
lean_lib «EvenPrime»

@[default_target]
lean_exe «even_prime_tests» where
  root := `Main
