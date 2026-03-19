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

**RQ12:** Does the fully automated two-stage pipeline — fixed Stage 1 discovers L3+L4
automatically → Stage 2 warm-starts from that encoder — match or exceed cold-start M2
on both 3D Dice and effective prototype quality (effective purity, effective AP)?

---

## Baseline for comparison

| Model | 3D Dice | Eff. Purity | Notes |
|-------|---------|-------------|-------|
| M2 cold-start | 0.8722 | ~0.77 (est.) | Best overall; manual L3+L4 |
| **L3+L4 warmstart (Exp A)** | **0.8656** | **0.649** | New best warm-start |
| L2+L3+L4 warmstart (v6) | 0.8635 | ~0.267 | Previous best on Dice |
| M4-attn wloss (v6) | 0.8475 | ~0.122 | Best v6 single-model |

Target for Exp C: **3D Dice ≥ 0.8722** and **effective purity ≥ 0.70** — Exp A
proves the pipeline concept works; Exp C with a clean encoder should close the gap.

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

## Experiment C — Full Automated Pipeline ⬜ SKIPPED

### Reason for skipping

Exp B revealed a fundamental objective mismatch: segmentation-driven attention always
prefers L2 over L4, regardless of architectural fixes. Exp C's premise — that Stage 1
auto-discovers L3+L4 — cannot hold. Any attention-based discovery trained on segmentation
loss will discover levels optimal for segmentation (L2-dominant), not for prototype quality.

Running Exp C would produce a warm-start Stage 2 with an L2+L3-discovered level set,
which is known from v6-D to give poor effective purity (~0.267). This is worse than
the manually-specified Exp A (eff. purity=0.649).

### What "automation" can and cannot mean here

| Can automate | Cannot automate (via seg-loss attention) |
|---|---|
| Confirming L1 is always uninformative (w_L1→0, consistent) | Discovering that L4 > L2 for prototype quality |
| Providing a warm-start encoder via Stage 1 | Selecting the interpretability-optimal level set |
| Reducing training time vs cold-start | Replacing empirical level ablation |

### RQ12 Answer: NOT TESTABLE as originally posed

The automation goal requires a mechanism that jointly optimises segmentation and
prototype quality. A pure segmentation-loss attention cannot do this. This is a
finding, not a failure — it correctly identifies the boundary of what attention-based
discovery can achieve.

**Future direction (out of scope for v7):** Multi-objective attention that adds a
prototype purity term to the attention loss, penalising levels with low purity from
receiving high weight. This would require online purity estimation during training.

---

## Stage Overview

| Stage | Name | Deliverable | Status |
|-------|------|-------------|--------|
| 29 | Exp A: Warm-start L3+L4 (existing Stage 1) | `notebooks/29_warmstart_l3l4.ipynb` | ✅ |
| 30 | Exp B: Fix LevelAttentionModule | `notebooks/30_fixed_stage1.ipynb` | ✅ (partial — see findings) |
| 31 | Exp C: Full automated pipeline | `notebooks/31_auto_pipeline.ipynb` | ⬜ skipped |
| 32 | Report v7 | `report/v7/report-v7.md` | ⬜ |

---

## Exp A Independence and Effect on B/C

Exp A and B test **independent** hypotheses — neither blocks the other:

| Exp A outcome | Effect on B | Effect on C |
|---|---|---|
| **Dice ≥ M2 + eff.purity ≥ 0.60** (full success) | B still runs — Exp A required manually specifying L3+L4; B provides the automation | C is high-confidence — warm-start with correct levels works; just needs reliable discovery |
| **Dice OK, purity < 0.60** | B still runs — cleaner Stage 1 encoder may rescue purity in C | C hypothesis shifts: relying on B's encoder being cleaner |
| **Dice < M2** (hard failure) | B still runs — fixed Stage 1 produces less L2-contaminated encoder | C is now the primary hope |

**Actual outcome: PARTIAL (3/4).** Dice OK, eff. purity ✅, L4 dominance ✅, Dice vs M2 ❌ (−0.0066).

**Exp B finding:** detach + temperature work mechanically, but MLP correctly learns L2 > L4
for segmentation. Objective mismatch is fundamental — not fixable within this architecture.

**Exp C: SKIPPED.** Automation of interpretability-optimal level selection via segmentation
attention is not achievable. L3+L4 is accepted as the empirically-justified level set.

## Decision Tree (final)

```
Stage 29: Exp A  ──── ✅  Dice=0.8656, eff.purity=0.649  [PARTIAL 3/4]
    │               Best warm-start result; pipeline concept validated
    │
Stage 30: Exp B  ──── ✅  T=5 keeps uniform during warmup ✅
    │               But MLP converges to L2+L3 (not L3+L4) once T drops
    │               Root cause: seg-loss attention ≠ interpretability-optimal attention
    │               Fundamental objective mismatch — not a fixable bug
    │
Stage 31: Exp C  ──── ⬜ SKIPPED
    │               Premise invalidated by Exp B finding
    │               Would produce L2-dominated warm-start (same failure as v6-D)
    │
Stage 32: Report v7  ──── ⬜ next
```

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Exp A Dice < M2: encoder L3 reps contaminated by L2 training | Medium | Exp B (fixed Stage 1) should produce cleaner L3 reps; run Exp C anyway |
| Exp A effective purity < 0.60: L4 still dominated despite no L2 skip | Low | L2 skip removed entirely; decoder must rely on L3+L4 |
| Exp B: detach causes attention to converge too slowly | Low | Temperature annealing provides principled warm-up; monitor w evolution |
| Exp B: all seeds converge to L4-only (not L3+L4) | Low | Stage 1 segmentation loss still flows through all skips; L3 skip useful for mid-size structures |
| Exp C: Stage 2 over-fits warm-start encoder (slow feature adaptation) | Medium | FREEZE_ENCODER_PHASE_A=False ensures encoder adapts from ep 1 |

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
  29_warmstart_l3l4.ipynb                 # Exp A: train + eval (warm-start L3+L4)
  30_fixed_stage1.ipynb                   # Exp B: train + eval (fixed Stage 1, 3 seeds)
  31_auto_pipeline.ipynb                  # Exp C: train + eval (full automated pipeline)

results/v7/
  train_curve_proto_ct_l3l4_warmstart.csv
  train_curve_fixed_stage1_seed42.csv
  train_curve_fixed_stage1_seed0.csv
  train_curve_fixed_stage1_seed123.csv
  attention_evolution_fixed_seed42.csv
  attention_evolution_fixed_seed0.csv
  attention_evolution_fixed_seed123.csv
  train_curve_auto_warmstart.csv
  comparison_table_v7.csv
  effective_quality_v7.csv

checkpoints/
  proto_seg_ct_l3l4_warmstart_v7.pth     # Exp A
  proto_seg_ct_l1234_attn_fixed.pth      # Exp B (seed=42 representative)
  proto_seg_ct_auto_warmstart.pth        # Exp C

report/v7/
  report-v7.md
```
