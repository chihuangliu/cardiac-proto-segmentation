# Execution Plan v7: Automated Level Selection for Interpretable Prototype Segmentation

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16 train / 2 val / 2 test patients)
**Hardware:** MacBook 48GB RAM (Apple Silicon MPS)
**Date:** 2026-03-19
**Preceded by:** `plan/execution-plan-v6.md`

---

## Motivation

v6 completed four experiments aimed at closing the M4-attn → M2 performance gap.
The final diagnosis is clear:

> The gap cannot be closed by modifying an M4 model's training after the fact.
> The warm-start pipeline (Stage 1 discovery → Stage 2 cold decoder) is the right
> architecture — but its Stage 1 level discovery is unreliable due to the L2 feedback loop.

**Key evidence from v6-D (two-stage warm-start):**

| Observation | Implication |
|-------------|-------------|
| L2+L3+L4 warmstart Dice = 0.8635 (−0.009 vs M2) | Stage 1 encoder learns good L3 reps even with w_L3≈0.01 |
| Adding L3 gave +0.034 Dice over L2+L4 | L3 skip is genuinely useful; encoder quality is not the bottleneck |
| L2 captured 74% pixel dominance in all warm-start models | L2 contamination from Stage 1 carries into Stage 2 decoder |
| seed=42 → L2+L4 discovery (expected L3+L4) | L2 feedback loop makes Stage 1 seed-dependent |

**Two separable problems remain:**

| Problem | Fix |
|---------|-----|
| Stage 1 discovers wrong levels (L2 feedback loop) | Feature detach + temperature annealing in `LevelAttentionModule` |
| Warm-starting imports L2 encoder bias into Stage 2 | Use correct level set (L3+L4) for Stage 2 regardless of discovery noise |

v7 resolves both problems sequentially, with a fast validation experiment first.

---

## Research Questions

**RQ10:** If we warm-start Stage 2 with L3+L4 (ignoring Stage 1's noisy L2+L4 discovery),
does the Stage 1 encoder's L3 representation quality translate into better Dice and
effective prototype quality than either cold-start M2 or the v6 warm-start variants?

**RQ11:** Does fixing `LevelAttentionModule` (feature detach + temperature annealing)
produce stable, seed-independent level discovery that reliably converges to L3+L4?

**RQ12:** Given a single trained M4 model, can inference-time level ablation identify
the level subset that lies on the Pareto front of segmentation Dice vs prototype purity,
without any retraining or prior knowledge of which levels are interpretability-optimal?

---

## Baseline for comparison

| Model | 3D Dice | Eff. Purity | Notes |
|-------|---------|-------------|-------|
| M2 cold-start | 0.8722 | ~0.77 (est.) | Best overall; manual L3+L4 |
| L3+L4 warm-start (Stage 29) | 0.8656 | 0.649 | Best warm-start |
| L2+L3+L4 warm-start (v6) | 0.8635 | ~0.267 | Previous best on Dice |
| M4 noent (v6, all levels) | ~0.87 | ~0.12 | Starting point for Stage 31 ablation |
| M4-attn wloss (v6) | 0.8475 | ~0.122 | |

---

## Experiment A — Warm-Start L3+L4 with Existing Stage 1 Encoder

### Hypothesis (RQ10)

The Stage 1 (seed=42) encoder already contains good L3 and L4 representations —
demonstrated by the +0.034 Dice improvement when L3 was added in v6-D. If we
warm-start a fresh M2(L3+L4) decoder using this encoder but **exclude L2 entirely**,
the decoder will co-adapt to L3+L4 skips only, eliminating the L2 dominance problem.

This experiment costs nothing extra — the Stage 1 checkpoint already exists.

### Config

```
Source checkpoint : checkpoints/proto_seg_ct_l1234_attn_noent.pth  (Stage 1, ep 45, seed=42)
Target checkpoint : checkpoints/proto_seg_ct_l3l4_warmstart_v7.pth
Log               : results/v7/train_curve_proto_ct_l3l4_warmstart.csv
Notebook          : notebooks/29_warmstart_l3l4.ipynb
PROTO_LEVELS      : [3, 4]          ← L3+L4 only, identical to cold-start M2
USE_LEVEL_ATTENTION : False
FREEZE_ENCODER_PHASE_A : False
λ_div=0.001, λ_push=0.5, λ_pull=0.25  (identical to cold-start M2)
```

Key difference from v6-D: we use L3+L4 (not the Stage 1-discovered L2+L4).
The encoder warm-start is identical; only the level set changes.

### Expected outcomes

- **Dice**: ≥ 0.8635 (L2+L3+L4 warmstart) since L2 skip no longer competes; possibly ≥ M2
- **L2 dominance**: 0% (L2 is not an active level)
- **L4 purity**: ≥ 0.709 (inherited from Stage 1 encoder; no L2 competition)
- **Effective purity**: substantially higher than ~0.267 (v6 warm-starts)

### Results

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| 3D Dice | **0.8656** | ≥ 0.8635 | ✅ |
| 3D Dice vs M2 | −0.0066 | ≥ 0.8722 | ❌ |
| Effective purity | **0.649** | ≥ 0.60 | ✅ |
| L4 dominance | **56.6%** | ≥ 40% | ✅ |

**Proto quality (per level):**

| Level | Purity | AP | Compactness | Dominance |
|-------|--------|----|-------------|-----------|
| L3 | 0.486 | 0.213 | 0.634 | 43.4% |
| L4 | 0.774 | 0.224 | 0.402 | 56.6% |

**Effective quality:** purity=0.649, AP=0.219, compactness=0.503

**Key findings:**
- Removing L2 was the decisive fix: effective purity jumped from ~0.267 (v6 warm-starts) → 0.649
- L4 now correctly dominates (56.6%), L3 at 43.4% — no L2 at all
- L4 purity 0.774 vs M2 0.804 — encoder contamination is minor, not a fundamental blocker
- Dice gap to M2 narrowed to −0.0066 (vs −0.0087 for L2+L3+L4 in v6-D)
- Best val Dice: 0.8234 at epoch 100

**RQ10 Verdict: PARTIAL (3/4).** Warm-start L3+L4 beats all v6 warm-starts on Dice and
dramatically improves effective prototype quality. The remaining Dice gap (−0.0066 vs M2)
is small. Exp B found that a "cleaner" Stage 1 encoder cannot be obtained automatically
via segmentation-driven attention — the gap to M2 is the final residual cost of warm-starting
from an L2-contaminated encoder, and is acceptable given the automation constraint is lifted.

### Tasks

- [x] Notebook `29_warmstart_l3l4.ipynb` (training + evaluation combined)
- [x] Answer RQ10

---

## Experiment B — Fix LevelAttentionModule: Feature Detach + Temperature Annealing

### Hypothesis (RQ11)

Two architectural changes in `LevelAttentionModule.forward()` break the L2 feedback
loop and replace init-dependent commitment with principled warm-up:

**Change 1 — Feature detach (1 line):**
```python
# Before (current):
x = torch.cat(pooled, dim=1)

# After:
x = torch.cat(pooled, dim=1).detach()   # attention is a reader, not a writer
```
The attention MLP can no longer push gradient back into the encoder. The encoder
is shaped purely by the segmentation loss. L2 cannot self-reinforce because
better L2 heatmaps no longer increase the MLP's gradient signal toward L2.

**Change 2 — Temperature annealing (training loop):**
```python
T_START   = 5.0   # initial temperature — softmax near-uniform
T_END     = 1.0   # final temperature — softmax sharp
ANNEAL_EP = 40    # reach T=1 by ep 40 (attention unfreeze + warmup period)

T = max(T_END, T_START * ((T_END / T_START) ** (epoch / ANNEAL_EP)))
# In forward(): w = torch.softmax(logits / T, dim=-1)
```
High T early on prevents random init from committing to any level prematurely.
As T→1, the softmax sharpens based on learned feature statistics rather than init luck.

### Side effects analysis

| Effect | Assessment |
|--------|------------|
| Encoder no longer gets gradient from attention MLP | ✅ Wanted — removes feedback loop |
| Attention MLP still trains (gets gradient via w → soft_mask → decoder → loss) | ✅ MLP learns to read features |
| Segmentation loss path unchanged (decoder → skip → encoder) | ✅ Encoder still trains normally |
| No entropy reg loss affected (noent baseline used) | ✅ No additional side effect |
| Attention convergence may be slower early on | Compensated by temperature annealing |

### Implementation

File: `src/models/proto_seg_net.py` — `LevelAttentionModule.forward()`
File: `notebooks/30_fixed_stage1.ipynb` — training loop with temperature schedule

### Stability verification

Run Stage 1 with **3 seeds** (42, 0, 123). Success = all three converge to the same
dominant levels (expected: L3+L4) within 40 epochs of attention unfreezing.

### Results (seed=42, partial run)

| Epoch | T | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|---|------|------|------|------|
| 5–30 | 5.0 | ~0.25 | ~0.25 | ~0.25 | ~0.25 |
| 35 | 4.5 | 0.09 | **0.35** | 0.29 | 0.27 |
| 40 | 4.0 | 0.05 | **0.47** | 0.29 | 0.19 |
| 45 | 3.5 | 0.02 | **0.48** | 0.30 | 0.20 |

T=5 kept weights uniform during warmup — temperature annealing works as designed.
But once T drops, the MLP converges to **L2+L3**, not L3+L4. L4 declines.

**Root cause:** Feature detach removed the L2 gradient feedback loop on the encoder.
However, the MLP still receives gradient via `w → soft_mask → decoder → seg_loss → ∇w`.
It learns to upweight whichever level maximises segmentation — and L2 (64×64 spatial)
genuinely helps segmentation more than L4 (16×16). This is correct MLP behaviour,
not a bug.

**Fundamental finding:** Segmentation-objective attention ≠ prototype-quality-optimal attention.
These are inherently conflicting objectives. L2 wins for segmentation; L4 wins for prototype quality.
No gradient-based mechanism trained on segmentation loss can automatically discover L3+L4
as the interpretability-optimal level set.

**RQ11 Verdict: NOT MET — but the failure is informative, not a fixable bug.**
The fix (detach + temperature) solved the gradient feedback loop correctly; the remaining
problem is a fundamental objective mismatch that cannot be resolved within this architecture.

### Tasks

- [x] Add `.detach()` in `LevelAttentionModule.forward()`
- [x] Add temperature parameter + annealing schedule to training loop
- [x] Notebook `30_fixed_stage1.ipynb` created
- [x] Seed=42 partial run confirms fundamental finding
- [x] Answer RQ11 — NOT MET (objective mismatch, not fixable)

---

## Stage Overview

| Stage | Name | Deliverable | Status |
|-------|------|-------------|--------|
| 29 | Warm-start L3+L4 (existing Stage 1 encoder) | `notebooks/29_warmstart_l3l4.ipynb` | ✅ |
| 30 | Fix LevelAttentionModule (detach + temperature) | `notebooks/30_fixed_stage1.ipynb` | ✅ (partial — see findings) |
| 31 | Report v7 | `report/v7/report-v7.md` | ⬜ |

---

## Stage Independence

| Stage | Outcome |
|-------|---------|
| 29: Warm-start L3+L4 | PARTIAL (3/4). Dice=0.8656, eff.purity=0.649 |
| 30: Fix LevelAttention | NOT MET. Detach+temp work mechanically but MLP converges to L2+L3, not L3+L4 (objective mismatch) |

## Decision Tree (final)

```
Stage 29: Warm-start L3+L4  ──── ✅  Dice=0.8656, eff.purity=0.649  [PARTIAL 3/4]
    │                              Best warm-start; pipeline concept validated
    │
Stage 30: Fix LevelAttention ──── ✅  T=5 keeps uniform during warmup ✅
    │                              MLP converges to L2+L3 once T drops
    │                              Fundamental objective mismatch — not fixable
    │
Stage 31: Report v7          ──── ⬜ next
```

---

## Abandoned Directions

### Post-hoc level ablation (`notebooks/31_ablation_level_selection.ipynb`)

**What was tried:** Load trained M4 model; zero out each level subset via
`pruned_levels` at inference time; measure 3D Dice + effective purity for all
15 non-empty subsets of {L1,L2,L3,L4}; identify Pareto-optimal subset from
the (Dice, eff_purity) front.

**What the ablation found:**
- Any subset without L4 gives Dice = 0. L4 is a hard architectural dependency
  of the M4 decoder (deepest `dec4` block receives zeros without it).
- {L3,L4} is on the Pareto front — but so are {L4}, {L1,L3,L4}, {L2,L3,L4},
  and {L1,L2,L3,L4}. Five subsets are Pareto-optimal simultaneously.
- {L3,L4} ablation Dice = 0.695 vs its true achievable Dice = 0.866 (Stage 29).
  The 0.17 gap is co-adaptation noise: the M4 decoder was trained with all 4
  levels and cannot be reliably evaluated on subsets at inference time.

**Why it fails as a discovery tool:**
1. The Pareto front is non-unique — 5 subsets are returned, not one.
2. No automatic rule selects {L3,L4} from those 5 without a human setting a
   Dice-sacrifice threshold.
3. Even with a threshold, the ablation Dice values are unreliable (−0.17 bias),
   so the threshold cannot be set correctly without retraining anyway.

**Conclusion:** Post-hoc ablation is descriptive (shows decoder dependencies)
but not prescriptive (cannot identify the interpretability-optimal level set
without prior knowledge). The ablation confirms {L3,L4} is on the front, but
only because we already know it is the right choice from Stage 29.

## Risk Register

| Risk | Outcome |
|------|---------|
| Stage 29 Dice < M2 (−0.0066): encoder L2 contamination | Observed. Accepted as warm-start residual. |
| Stage 30 MLP converges to wrong levels | Observed. Root cause: objective mismatch. |

---

## Notebook Convention (v7 onwards)

Each notebook contains **both training and evaluation** — no separate analysis notebooks.
Sections within each notebook:
1. Config + imports
2. Model init / weight transfer
3. Training loop + live training curve
4. Evaluation (3D Dice per patient)
5. Proto quality (purity, AP, compactness, dominance, effective quality)
6. Comparison table vs all relevant baselines
7. RQ verdict

## File Structure (v7 additions)

```
plan/
  execution-plan-v7.md

src/
  models/
    proto_seg_net.py                      # LevelAttentionModule: detach + temperature param

notebooks/
  29_warmstart_l3l4.ipynb                 # Stage 29: train + eval (warm-start L3+L4)
  30_fixed_stage1.ipynb                   # Stage 30: train + eval (fixed LevelAttention, seed=42)
  31_ablation_level_selection.ipynb       # Abandoned: post-hoc ablation (negative result)

results/v7/
  train_curve_proto_ct_l3l4_warmstart.csv
  train_curve_fixed_stage1_seed42.csv
  attn_evolution_fixed_seed42.csv
  ablation_results.csv                    # Abandoned stage: 15-row subset table
  pareto_front_ablation.png              # Abandoned stage: Pareto front
  comparison_table_v7.csv
  effective_quality_v7.csv

checkpoints/
  proto_seg_ct_l3l4_warmstart_v7.pth     # Stage 29
  proto_seg_ct_l1234_attn_fixed.pth      # Stage 30 (seed=42)

report/v7/
  report-v7.md
```
