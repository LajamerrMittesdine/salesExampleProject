/-!
# Core definitions and preparatory lemmas

Aligned with the v2 paper: prime / composite / even, with the
cancellation-style and factorization lemmas made first-class.
-/

namespace EvenPrime

/-! ## Definitions -/

/-- `n` is prime (divisor form / irreducible in `(ℕ, ·)`):
greater than 1, and every divisor is `1` or `n`. -/
def IsPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d ∣ n → d = 1 ∨ d = n

/-- `n` is composite: greater than 1 and not prime. -/
def IsComposite (n : Nat) : Prop :=
  n > 1 ∧ ¬ IsPrime n

/-- `n` is even: divisible by 2. -/
def IsEven (n : Nat) : Prop :=
  2 ∣ n

/-- Irreducible form: `n > 1` and every factorization `n = a * b` has a unit factor.
On `ℕ` the only unit is `1`, so this says `a = 1 ∨ b = 1`. -/
def IsIrreducible (n : Nat) : Prop :=
  n > 1 ∧ ∀ a b : Nat, a * b = n → a = 1 ∨ b = 1

/-! ## Small arithmetic facts (named, not buried) -/

theorem ne_zero_iff_ge_one {n : Nat} : n ≠ 0 ↔ n ≥ 1 := by omega

theorem pos_of_ne_zero {n : Nat} (h : n ≠ 0) : n ≥ 1 :=
  (ne_zero_iff_ge_one).1 h

theorem lt_two_iff : ∀ n : Nat, n < 2 ↔ n = 0 ∨ n = 1 := by
  intro n; omega

theorem gt_one_iff_ge_two {n : Nat} : n > 1 ↔ n ≥ 2 := by omega

/-! ## Evenness audits -/

theorem isEven_iff_two_dvd (n : Nat) : IsEven n ↔ 2 ∣ n := Iff.rfl

theorem isEven_iff_mod_two (n : Nat) : IsEven n ↔ n % 2 = 0 := by
  constructor
  · intro ⟨k, hk⟩
    simp [hk]
  · intro h
    refine ⟨n / 2, ?_⟩
    have := Nat.div_add_mod n 2
    omega

/-- Zero is even (often overlooked edge case). -/
theorem zero_is_even : IsEven 0 := ⟨0, by decide⟩

theorem one_not_even : ¬ IsEven 1 := by
  intro h
  have : 1 % 2 = 0 := (isEven_iff_mod_two 1).1 h
  exact absurd this (by decide)

/-! ## Divisors of two; two is an even prime -/

theorem dvd_two {d : Nat} (hd : d ∣ 2) : d = 1 ∨ d = 2 := by
  have hle : d ≤ 2 := Nat.le_of_dvd (by decide : 0 < 2) hd
  have hpos : 0 < d := Nat.pos_of_dvd_of_pos hd (by decide : 0 < 2)
  match d, hpos, hle with
  | 0, h0, _ => exact absurd h0 (Nat.not_lt_zero 0)
  | 1, _, _ => exact Or.inl rfl
  | 2, _, _ => exact Or.inr rfl
  | n + 3, _, _ => omega

theorem two_is_prime : IsPrime 2 :=
  ⟨by decide, fun d hd => dvd_two hd⟩

theorem two_is_even : IsEven 2 := ⟨1, rfl⟩

theorem two_is_even_prime : IsPrime 2 ∧ IsEven 2 :=
  ⟨two_is_prime, two_is_even⟩

/-! ## Zero and one are not prime; examples 3 and 4 -/

theorem zero_not_prime : ¬ IsPrime 0 := by
  intro h; exact absurd h.1 (by decide)

theorem one_not_prime : ¬ IsPrime 1 := by
  intro h; exact absurd h.1 (by decide)

theorem three_is_prime : IsPrime 3 := by
  refine ⟨by decide, ?_⟩
  intro d hd
  have hpos : 0 < d := Nat.pos_of_dvd_of_pos hd (by decide)
  have hle : d ≤ 3 := Nat.le_of_dvd (by decide) hd
  match d, hpos, hle with
  | 0, h0, _ => exact absurd h0 (Nat.not_lt_zero 0)
  | 1, _, _ => exact Or.inl rfl
  | 2, _, _ => exact absurd hd (by decide : ¬ 2 ∣ 3)
  | 3, _, _ => exact Or.inr rfl
  | n + 4, _, _ => omega

theorem four_not_prime : ¬ IsPrime 4 := by
  intro hp
  have h2 : 2 ∣ 4 := ⟨2, rfl⟩
  have := hp.2 2 h2
  cases this with
  | inl h => exact absurd h (by decide)
  | inr h => exact absurd h (by decide)

theorem four_is_composite : IsComposite 4 :=
  ⟨by decide, four_not_prime⟩

/-! ## Characterization lemmas -/

theorem isPrime_iff_no_proper_divisor (n : Nat) :
    IsPrime n ↔ n > 1 ∧ ∀ d : Nat, 1 < d → d < n → ¬ d ∣ n := by
  constructor
  · intro ⟨hgt, hdiv⟩
    refine ⟨hgt, ?_⟩
    intro d hd1 hdn hdd
    have := hdiv d hdd
    cases this <;> omega
  · intro ⟨hgt, hnone⟩
    refine ⟨hgt, ?_⟩
    intro d hd
    have hpos : 0 < d := Nat.pos_of_dvd_of_pos hd (Nat.zero_lt_of_lt hgt)
    have hle : d ≤ n := Nat.le_of_dvd (Nat.zero_lt_of_lt hgt) hd
    by_cases h1 : d = 1
    · exact Or.inl h1
    · by_cases hnEq : d = n
      · exact Or.inr hnEq
      · have hd1 : 1 < d := by omega
        have hdn : d < n := by omega
        exact absurd hd (hnone d hd1 hdn)

/-- On `ℕ`, prime ↔ irreducible. -/
theorem isPrime_iff_isIrreducible (n : Nat) : IsPrime n ↔ IsIrreducible n := by
  constructor
  · intro ⟨hgt, hdiv⟩
    refine ⟨hgt, ?_⟩
    intro a b hab
    have ha : a ∣ n := ⟨b, hab.symm⟩
    have := hdiv a ha
    cases this with
    | inl h => exact Or.inl h
    | inr ha_eq =>
        have hb : b = 1 := by
          have hmul : n * b = n * 1 := by
            calc
              n * b = a * b := by rw [ha_eq]
              _ = n := hab
              _ = n * 1 := by rw [Nat.mul_one]
          exact Nat.eq_of_mul_eq_mul_left (by omega : 0 < n) hmul
        exact Or.inr hb
  · intro ⟨hgt, hirr⟩
    refine ⟨hgt, ?_⟩
    intro d hd
    obtain ⟨k, hk⟩ := hd
    have := hirr d k (Eq.symm hk)
    cases this with
    | inl h => exact Or.inl h
    | inr h =>
        have : d = n := by
          calc
            d = d * 1 := (Nat.mul_one d).symm
            _ = d * k := by rw [h]
            _ = n := hk.symm
        exact Or.inr this

/-- If `n` is not prime and `n > 1`, it has a proper factor. -/
theorem exists_proper_factor_of_not_prime {n : Nat}
    (hgt : n > 1) (hnp : ¬ IsPrime n) :
    ∃ d k : Nat, n = d * k ∧ 1 < d ∧ d < n ∧ 1 < k ∧ k < n := by
  classical
  have hfail : ¬ ∀ d : Nat, d ∣ n → d = 1 ∨ d = n := by
    intro hall
    exact hnp ⟨hgt, hall⟩
  have hex : ∃ d : Nat, d ∣ n ∧ ¬ (d = 1 ∨ d = n) :=
    Classical.byContradiction fun hnone =>
      hfail fun d hd => by
        by_cases h1 : d = 1
        · exact Or.inl h1
        · by_cases hnEq : d = n
          · exact Or.inr hnEq
          · exact False.elim (hnone ⟨d, hd, fun h => h.elim h1 hnEq⟩)
  obtain ⟨d, hd, hne⟩ := hex
  have hne1 : d ≠ 1 := fun h => hne (Or.inl h)
  have hnen : d ≠ n := fun h => hne (Or.inr h)
  have hpos : 0 < d := Nat.pos_of_dvd_of_pos hd (Nat.zero_lt_of_lt hgt)
  have hle : d ≤ n := Nat.le_of_dvd (Nat.zero_lt_of_lt hgt) hd
  have hd1 : 1 < d := by omega
  have hdn : d < n := by omega
  obtain ⟨k, hk⟩ := hd
  -- hk : n = d * k
  subst hk
  -- now n is d * k
  have hk0 : k ≠ 0 := by
    intro hz
    -- d * 0 = 0, but n > 1
    simp [hz] at hgt
  have hk1ne : k ≠ 1 := by
    intro h1
    -- d < d * 1 = d is absurd
    simp [h1] at hdn
  have hk1 : 1 < k := by
    have : k ≥ 1 := pos_of_ne_zero hk0
    omega
  have hkn : k < d * k := by
    have hge : 2 * k ≤ d * k := Nat.mul_le_mul_right k (Nat.succ_le_of_lt hd1)
    omega
  exact ⟨d, k, rfl, hd1, hdn, hk1, hkn⟩

/-- Composite ↔ proper factorization. -/
theorem isComposite_iff_proper_factors (n : Nat) :
    IsComposite n ↔
      n > 1 ∧ ∃ d k : Nat, n = d * k ∧ 1 < d ∧ d < n ∧ 1 < k ∧ k < n := by
  constructor
  · intro ⟨hgt, hnp⟩
    exact ⟨hgt, exists_proper_factor_of_not_prime hgt hnp⟩
  · intro ⟨hgt, d, k, hn, hd1, hdn, hk1, hkn⟩
    refine ⟨hgt, ?_⟩
    intro hp
    have hdd : d ∣ n := ⟨k, hn⟩
    have := hp.2 d hdd
    cases this <;> omega

/-! ## Prime divides prime -/

theorem prime_dvd_prime_iff_eq {p q : Nat}
    (hp : IsPrime p) (hq : IsPrime q) : p ∣ q ↔ p = q := by
  constructor
  · intro h
    have := hq.2 p h
    cases this with
    | inl h1 =>
        have : p > 1 := hp.1
        omega
    | inr heq => exact heq
  · intro h
    exact h ▸ Nat.dvd_refl q

/-- Halving lemma: even n > 2 factors as 2*k with 1 < k < n. -/
theorem halve_even_gt_two {n : Nat} (hn : n > 2) (he : IsEven n) :
    ∃ k : Nat, n = 2 * k ∧ 1 < k ∧ k < n := by
  obtain ⟨k, hk⟩ := he
  refine ⟨k, hk, ?_, ?_⟩
  · have : 2 * k > 2 := by simpa [hk] using hn
    omega
  · have : n = 2 * k := hk
    omega

theorem prime_ge_two {n : Nat} (hp : IsPrime n) : n ≥ 2 := by
  have : n > 1 := hp.1
  omega

end EvenPrime
