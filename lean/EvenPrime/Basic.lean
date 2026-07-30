import EvenPrime.Defs

/-!
# Basic lemmas (paper §§arith–prep examples)

Named toolkit facts used by the uniqueness proofs. Avoids burying
cancellation-style reasoning inside larger arguments.
-/

namespace EvenPrime

/-! ## Small order facts -/

theorem one_le_iff_ne_zero {n : Nat} : 1 ≤ n ↔ n ≠ 0 := by omega

theorem one_lt_iff_two_le {n : Nat} : 1 < n ↔ 2 ≤ n := by omega

theorem lt_two_iff {n : Nat} : n < 2 ↔ n = 0 ∨ n = 1 := by omega

theorem two_le_of_isPrime {n : Nat} (hp : IsPrime n) : 2 ≤ n := by
  have : 1 < n := hp.1
  omega

/-! ## Units and multiplicative facts -/

/-- The only unit in `(ℕ, ·)` is `1`. Paper: units lemma for irreducibility. -/
theorem mul_eq_one_iff {a b : Nat} : a * b = 1 ↔ a = 1 ∧ b = 1 := by
  constructor
  · intro h
    have hb0 : b ≠ 0 := by intro hb; simp [hb] at h
    have ha0 : a ≠ 0 := by intro ha; simp [ha] at h
    have ha_le : a ≤ 1 := by
      have hle : a * 1 ≤ a * b := Nat.mul_le_mul_left a (one_le_iff_ne_zero.2 hb0)
      have hle' : a ≤ a * b := by simpa [Nat.mul_one] using hle
      omega
    have hb_le : b ≤ 1 := by
      have hle : 1 * b ≤ a * b := Nat.mul_le_mul_right b (one_le_iff_ne_zero.2 ha0)
      have hle' : b ≤ a * b := by simpa [Nat.one_mul] using hle
      omega
    omega
  · intro ⟨ha, hb⟩
    simp [ha, hb]

/-! ## Evenness -/

theorem isEven_iff_two_dvd (n : Nat) : IsEven n ↔ 2 ∣ n := Iff.rfl

theorem isEven_iff_mod_two (n : Nat) : IsEven n ↔ n % 2 = 0 := by
  constructor
  · intro ⟨k, hk⟩; simp [hk]
  · intro h
    refine ⟨n / 2, ?_⟩
    have := Nat.div_add_mod n 2
    omega

theorem isEven_zero : IsEven 0 := ⟨0, by decide⟩

theorem not_isEven_one : ¬ IsEven 1 := by
  intro h
  have : 1 % 2 = 0 := (isEven_iff_mod_two 1).1 h
  exact absurd this (by decide)

theorem isEven_two : IsEven 2 := ⟨1, rfl⟩

/-! ## Divisors of two; two is prime -/

theorem eq_one_or_eq_two_of_dvd_two {d : Nat} (hd : d ∣ 2) : d = 1 ∨ d = 2 := by
  have hle : d ≤ 2 := Nat.le_of_dvd (by decide : 0 < 2) hd
  have hpos : 0 < d := Nat.pos_of_dvd_of_pos hd (by decide : 0 < 2)
  match d, hpos, hle with
  | 0, h0, _ => exact absurd h0 (Nat.not_lt_zero 0)
  | 1, _, _ => exact Or.inl rfl
  | 2, _, _ => exact Or.inr rfl
  | n + 3, _, _ => omega

/-- Paper: `prop:two` (primality half). -/
theorem isPrime_two : IsPrime 2 :=
  ⟨by decide, fun d hd => eq_one_or_eq_two_of_dvd_two hd⟩

theorem isPrime_two_and_isEven_two : IsPrime 2 ∧ IsEven 2 :=
  ⟨isPrime_two, isEven_two⟩

/-! ## Zero / one / three / four -/

theorem not_isPrime_zero : ¬ IsPrime 0 := fun h => absurd h.1 (by decide)

theorem not_isPrime_one : ¬ IsPrime 1 := fun h => absurd h.1 (by decide)

theorem isPrime_three : IsPrime 3 := by
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

theorem not_isPrime_four : ¬ IsPrime 4 := by
  intro hp
  have := hp.2 2 ⟨2, rfl⟩
  cases this <;> simp_all

theorem isComposite_four : IsComposite 4 := ⟨by decide, not_isPrime_four⟩

/-! ## Characterizations -/

theorem isPrime_iff_no_proper_divisor (n : Nat) :
    IsPrime n ↔ n > 1 ∧ ∀ d : Nat, 1 < d → d < n → ¬ d ∣ n := by
  constructor
  · intro ⟨hgt, hdiv⟩
    refine ⟨hgt, fun d hd1 hdn hdd => ?_⟩
    have := hdiv d hdd
    cases this <;> omega
  · intro ⟨hgt, hnone⟩
    refine ⟨hgt, fun d hd => ?_⟩
    have hpos : 0 < d := Nat.pos_of_dvd_of_pos hd (Nat.zero_lt_of_lt hgt)
    have hle : d ≤ n := Nat.le_of_dvd (Nat.zero_lt_of_lt hgt) hd
    by_cases h1 : d = 1
    · exact Or.inl h1
    · by_cases hnEq : d = n
      · exact Or.inr hnEq
      · exact absurd hd (hnone d (by omega) (by omega))

/-- Paper units + definitions ⇒ prime ↔ irreducible. -/
theorem isPrime_iff_isIrreducible (n : Nat) : IsPrime n ↔ IsIrreducible n := by
  constructor
  · intro ⟨hgt, hdiv⟩
    refine ⟨hgt, fun a b hab => ?_⟩
    have ha : a ∣ n := ⟨b, hab.symm⟩
    match hdiv a ha with
    | Or.inl h => exact Or.inl h
    | Or.inr ha_eq =>
        have hmul : n * b = n * 1 := by
          calc
            n * b = a * b := by rw [ha_eq]
            _ = n := hab
            _ = n * 1 := (Nat.mul_one n).symm
        exact Or.inr (Nat.eq_of_mul_eq_mul_left (by omega : 0 < n) hmul)
  · intro ⟨hgt, hirr⟩
    refine ⟨hgt, fun d hd => ?_⟩
    obtain ⟨k, hk⟩ := hd
    match hirr d k hk.symm with
    | Or.inl h => exact Or.inl h
    | Or.inr h =>
        have : d = n := by
          calc
            d = d * 1 := (Nat.mul_one d).symm
            _ = d * k := by rw [h]
            _ = n := hk.symm
        exact Or.inr this

/-! ## Proper factors (decidable search; no `Classical.choice`) -/

/-- If `n > 1` is not prime, it has a proper factor. Paper: `lem:composite-factors`. -/
theorem exists_proper_factor_of_not_prime {n : Nat}
    (hgt : n > 1) (hnp : ¬ IsPrime n) :
    ∃ d k : Nat, n = d * k ∧ 1 < d ∧ d < n ∧ 1 < k ∧ k < n := by
  -- Candidates: d ∈ (1, n) with d ∣ n
  let ds := (List.range n).filter fun d => decide (1 < d ∧ d ∣ n)
  have hne : ds ≠ [] := by
    intro hempty
    apply hnp
    refine ⟨hgt, ?_⟩
    intro d hd
    have hle : d ≤ n := Nat.le_of_dvd (Nat.zero_lt_of_lt hgt) hd
    have hpos : 0 < d := Nat.pos_of_dvd_of_pos hd (Nat.zero_lt_of_lt hgt)
    by_cases h1 : d = 1
    · exact Or.inl h1
    · by_cases hnEq : d = n
      · exact Or.inr hnEq
      · have hlt : d < n := Nat.lt_of_le_of_ne hle hnEq
        have hd1 : 1 < d := by omega
        have hmem : d ∈ ds := by
          simp [ds, List.mem_filter, List.mem_range, hlt, hd, hd1]
        simp [hempty] at hmem
  have ⟨d, hdmem⟩ : ∃ d, d ∈ ds := List.exists_mem_of_ne_nil ds hne
  have hprops : 1 < d ∧ d ∣ n ∧ d < n := by
    simp [ds, List.mem_filter, List.mem_range] at hdmem
    exact ⟨hdmem.2.1, hdmem.2.2, hdmem.1⟩
  obtain ⟨hd1, hd, hdn⟩ := hprops
  obtain ⟨k, hk⟩ := hd
  subst hk
  have hk1ne : k ≠ 1 := by
    intro h1; simp [h1] at hdn
  have hk0 : k ≠ 0 := by
    intro hz; simp [hz] at hgt
  have hk1 : 1 < k := by
    have : 1 ≤ k := (one_le_iff_ne_zero).2 hk0
    omega
  have hkn : k < d * k := by
    have : 2 * k ≤ d * k := Nat.mul_le_mul_right k (Nat.succ_le_of_lt hd1)
    omega
  exact ⟨d, k, rfl, hd1, hdn, hk1, hkn⟩

theorem isComposite_iff_proper_factors (n : Nat) :
    IsComposite n ↔
      n > 1 ∧ ∃ d k : Nat, n = d * k ∧ 1 < d ∧ d < n ∧ 1 < k ∧ k < n := by
  constructor
  · intro ⟨hgt, hnp⟩
    exact ⟨hgt, exists_proper_factor_of_not_prime hgt hnp⟩
  · intro ⟨hgt, d, k, hn, hd1, hdn, hk1, hkn⟩
    refine ⟨hgt, fun hp => ?_⟩
    have := hp.2 d ⟨k, hn⟩
    cases this <;> omega

/-! ## Prime divides prime; halving -/

theorem eq_of_isPrime_of_dvd {p q : Nat}
    (hp : IsPrime p) (hq : IsPrime q) (h : p ∣ q) : p = q := by
  have := hq.2 p h
  cases this with
  | inl h1 =>
      have : 1 < p := hp.1
      omega
  | inr heq => exact heq

theorem isPrime_dvd_isPrime_iff_eq {p q : Nat}
    (hp : IsPrime p) (hq : IsPrime q) : p ∣ q ↔ p = q :=
  ⟨eq_of_isPrime_of_dvd hp hq, fun h => h ▸ Nat.dvd_refl q⟩

/-- Paper: `lem:halve`. -/
theorem exists_eq_two_mul_of_gt_two_of_isEven {n : Nat}
    (hn : n > 2) (he : IsEven n) :
    ∃ k : Nat, n = 2 * k ∧ 1 < k ∧ k < n := by
  obtain ⟨k, hk⟩ := he
  refine ⟨k, hk, ?_, ?_⟩
  · have : 2 * k > 2 := by simpa [hk] using hn
    omega
  · omega

end EvenPrime
