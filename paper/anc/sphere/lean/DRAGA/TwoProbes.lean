import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum

/-! Identifiability of normalized kernel sums under common stream continuations.
    Theorems about exact observations are pointwise in queries. -/
noncomputable section

namespace DRAGA

structure Stats (Q I : Type*) where
  mass : Q → ℝ
  num : Q → I → ℝ

def update {K Q I : Type*} (w : Q → K → ℝ) (s : Stats Q I)
    (tok : K × (I → ℝ)) : Stats Q I where
  mass q := s.mass q + w q tok.1
  num q i := s.num q i + w q tok.1 * tok.2 i

def readout {Q I : Type*} (s : Stats Q I) (q : Q) (i : I) : ℝ :=
  s.num q i / s.mass q

def FutureEq {K Q I : Type*} (w : Q → K → ℝ) (allowed : (I → ℝ) → Prop)
    (s t : Stats Q I) : Prop :=
  ∀ (xs : List (K × (I → ℝ))), xs ≠ [] → (∀ tok ∈ xs, allowed tok.2) →
    ∀ q i, readout (xs.foldl (update w) s) q i = readout (xs.foldl (update w) t) q i

theorem recover_mass {d d' b x x' : ℝ} (hb : 0 < b)
    (hd : d + b ≠ 0) (hd' : d' + b ≠ 0)
    (h0 : x / (d+b) = x' / (d'+b))
    (h1 : (x+b) / (d+b) = (x'+b) / (d'+b)) : d = d' := by
  have h0' := (div_eq_div_iff hd hd').mp h0
  have h1' := (div_eq_div_iff hd hd').mp h1
  have h : b * (d - d') = 0 := by nlinarith only [h0', h1']
  have hdiff := (mul_eq_zero.mp h).resolve_left (ne_of_gt hb)
  linarith

/-- A zero probe and any admissible vector with one coordinate equal to one
    distinguish the complete positive-mass and numerator profiles. -/
theorem futureEq_iff {K Q I : Type*} (w : Q → K → ℝ)
    (allowed : (I → ℝ) → Prop) (k : K) (i₀ : I) (e : I → ℝ)
    (he : e i₀ = 1) (hzero : allowed (fun _ => 0)) (hone : allowed e)
    (hw : ∀ q, 0 < w q k) (s t : Stats Q I)
    (hs : ∀ q, 0 ≤ s.mass q) (ht : ∀ q, 0 ≤ t.mass q) :
    FutureEq w allowed s t ↔ s = t := by
  constructor
  · intro hf
    have obs (v : I → ℝ) (hv : allowed v) (q : Q) (i : I) :
        (s.num q i + w q k * v i) / (s.mass q + w q k) =
        (t.num q i + w q k * v i) / (t.mass q + w q k) := by
      have h := hf [(k, v)] (by simp) (by simpa using hv) q i
      simpa [readout, update] using h
    have z (q : Q) (i : I) :
        s.num q i / (s.mass q + w q k) = t.num q i / (t.mass q + w q k) := by
      simpa using obs (fun _ => 0) hzero q i
    have ds (q : Q) : s.mass q + w q k ≠ 0 := ne_of_gt (add_pos_of_nonneg_of_pos (hs q) (hw q))
    have dt (q : Q) : t.mass q + w q k ≠ 0 := ne_of_gt (add_pos_of_nonneg_of_pos (ht q) (hw q))
    have m (q : Q) : s.mass q = t.mass q :=
      recover_mass (hw q) (ds q) (dt q) (z q i₀) (by simpa [he] using obs e hone q i₀)
    have n (q : Q) (i : I) : s.num q i = t.num q i := by
      have h := (div_eq_div_iff (ds q) (dt q)).mp (z q i)
      rw [m q] at h
      exact mul_right_cancel₀ (dt q) h
    cases s
    cases t
    congr
    · exact funext m
    · exact funext (fun q => funext (n q))
  · rintro rfl
    intro xs _ _ q i
    rfl

/-- Every deterministic exact streaming encoding refines the profile equivalence
    on its histories. No linearity or continuity hypothesis is needed here. -/
theorem exact_state_refines_stats {K Q I H S : Type*} (w : Q → K → ℝ)
    (allowed : (I → ℝ) → Prop) (k : K) (i₀ : I) (e : I → ℝ)
    (he : e i₀ = 1) (hzero : allowed (fun _ => 0)) (hone : allowed e)
    (hw : ∀ q, 0 < w q k) (profile : H → Stats Q I) (encode : H → S)
    (decode : S → List (K × (I → ℝ)) → Q → I → ℝ)
    (hm : ∀ h q, 0 ≤ (profile h).mass q)
    (exactRead : ∀ h xs, xs ≠ [] → (∀ tok ∈ xs, allowed tok.2) → ∀ q i,
      decode (encode h) xs q i = readout (xs.foldl (update w) (profile h)) q i)
    (h₁ h₂ : H) (hc : encode h₁ = encode h₂) :
    profile h₁ = profile h₂ := by
  apply (futureEq_iff w allowed k i₀ e he hzero hone hw _ _ (hm h₁) (hm h₂)).mp
  intro xs hn ha q i
  rw [← exactRead h₁ xs hn ha q i, ← exactRead h₂ xs hn ha q i, hc]

/-- Separation in one output coordinate when a common suffix has sign s. -/
def gap (A b z x e s : ℝ) : ℝ :=
  (x + s*b*e)/(A+z) - (-x + s*b*e)/(A-z)

theorem scaled_gap (A b z x e s : ℝ) (hp : A+z ≠ 0) (hm : A-z ≠ 0) :
    ((A+z)*(A-z)) * gap A b z x e s = 2*(A*x-s*b*z*e) := by
  unfold gap
  field_simp [hp, hm]
  ring

theorem two_sign_energy (A b z x e : ℝ) (hp : A+z ≠ 0) (hm : A-z ≠ 0) :
    ((A+z)*(A-z))^2 * ((gap A b z x e 1)^2 + (gap A b z x e (-1))^2) =
      8*(A^2*x^2 + b^2*z^2*e^2) := by
  calc
    _ = (((A+z)*(A-z))*gap A b z x e 1)^2 +
        (((A+z)*(A-z))*gap A b z x e (-1))^2 := by ring
    _ = _ := by rw [scaled_gap _ _ _ _ _ _ hp hm, scaled_gap _ _ _ _ _ _ hp hm]; ring

open scoped BigOperators

/-- Summing the two squared Euclidean gaps removes the mixed term in every
    value dimension. The suffix vector has Euclidean norm one. -/
theorem vector_two_sign_energy {I : Type*} [Fintype I] (A b z : ℝ) (x e : I → ℝ)
    (he : ∑ i, (e i)^2 = 1) (hp : A+z ≠ 0) (hm : A-z ≠ 0) :
    ((A+z)*(A-z))^2 *
      (∑ i, ((gap A b z (x i) (e i) 1)^2 + (gap A b z (x i) (e i) (-1))^2)) =
      8*(A^2*(∑ i, (x i)^2) + b^2*z^2) := by
  calc
    _ = ∑ i, ((A+z)*(A-z))^2 *
            ((gap A b z (x i) (e i) 1)^2 + (gap A b z (x i) (e i) (-1))^2) := by
        rw [Finset.mul_sum]
    _ = ∑ i, 8*(A^2*(x i)^2 + b^2*z^2*(e i)^2) := by
        exact Finset.sum_congr rfl (fun i _ => two_sign_energy A b z (x i) (e i) hp hm)
    _ = _ := by rw [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, he, mul_one]

/-- At least one of the two allowed suffix signs witnesses the full joint
    numerator/mass energy, even if the two contributions align. -/
theorem vector_probe_witness {I : Type*} [Fintype I] (A b z : ℝ) (x e : I → ℝ)
    (he : ∑ i, (e i)^2 = 1) (hp : A+z ≠ 0) (hm : A-z ≠ 0) :
    ∃ s : ℝ, (s = 1 ∨ s = -1) ∧
      4*(A^2*(∑ i, (x i)^2) + b^2*z^2) ≤
        ((A+z)*(A-z))^2 * ∑ i, (gap A b z (x i) (e i) s)^2 := by
  have energy := vector_two_sign_energy A b z x e he hp hm
  rw [Finset.sum_add_distrib, mul_add] at energy
  by_cases h : 4*(A^2*(∑ i, (x i)^2) + b^2*z^2) ≤
      ((A+z)*(A-z))^2 * ∑ i, (gap A b z (x i) (e i) 1)^2
  · exact ⟨1, Or.inl rfl, h⟩
  · exact ⟨-1, Or.inr rfl, by linarith only [energy, lt_of_not_ge h]⟩

/-- At beta = log 2 the aligned/anti-aligned token weights are 2 and 1/2.
    A perpendicular key of value one gives the two fractions below. -/
theorem four_token_witness :
    (1:ℝ)/(3*2+1/2+1) = 2/15 ∧
    (1:ℝ)/(2+3*(1/2)+1) = 2/9 ∧ (2:ℝ)/9-2/15=4/45 := by norm_num

/-- An ordinary stream instance at the query x, with three possible keys.
    Weight 1 is the perpendicular key used in the common continuation. -/
inductive Ray | aligned | opposite | perpendicular

def witnessKernel (_ : Unit) : Ray → ℝ
  | .aligned => 2
  | .opposite => 1/2
  | .perpendicular => 1

def sampleToken (k : Ray) (v : ℝ) : Ray × (Unit → ℝ) := (k, fun _ => v)

def fourPrefix (a b : Ray) : List (Ray × (Unit → ℝ)) :=
  [sampleToken a 0, sampleToken a 0, sampleToken a 0, sampleToken b 0]

def exampleRead (xs : List (Ray × (Unit → ℝ))) : ℝ :=
  let empty : Stats Unit Unit := ⟨fun _ => 0, fun _ _ => 0⟩
  readout (xs.foldl (update witnessKernel) empty) () ()

theorem ordinary_stream_witness :
    let left := fourPrefix .aligned .opposite
    let right := fourPrefix .opposite .aligned
    let suffix := [sampleToken .perpendicular 1]
    exampleRead left = 0 ∧ exampleRead right = 0 ∧
      exampleRead (left ++ suffix) = 2/15 ∧ exampleRead (right ++ suffix) = 2/9 := by
  norm_num [exampleRead, fourPrefix, sampleToken, readout, update, witnessKernel, List.foldl]

/-- A deterministic state collision is impossible whenever one allowable
    common future separates the targets by more than twice the tolerance.
    Its output may be a vector in any type with the specified triangle bound. -/
theorem collision_obstruction {H F S O : Type*} (state : H → S)
    (target : H → F → O) (predict : S → F → O) (dist : O → O → ℝ)
    (htri : ∀ x y z, dist x z ≤ dist x y + dist z y)
    (ε : ℝ) (hacc : ∀ h f, dist (target h f) (predict (state h) f) ≤ ε)
    (h h' : H) (f : F) (hsep : 2*ε < dist (target h f) (target h' f)) :
    state h ≠ state h' := by
  intro hc
  have ha := hacc h f
  have hb := hacc h' f
  rw [← hc] at hb
  have ht := htri (target h f) (predict (state h) f) (target h' f)
  linarith

end DRAGA

#print axioms DRAGA.futureEq_iff
#print axioms DRAGA.vector_two_sign_energy
#print axioms DRAGA.four_token_witness
#print axioms DRAGA.collision_obstruction
#print axioms DRAGA.vector_probe_witness
#print axioms DRAGA.ordinary_stream_witness
#print axioms DRAGA.exact_state_refines_stats
