# Execution Plan v8: Two-Phase Pipeline with Anatomical Locality Constraint

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16/2/2) + MR (16/2/2, optional)
**Hardware:** MacBook 48GB RAM (Apple Silicon MPS)
**Date:** 2026-03-20
**Preceded by:** `plan/execution-plan-v7.md`

---

## Motivation

v7 produced two negative results and one positive result:

| Finding | Implication |
|---------|-------------|
| Attention-based level selection fails (objective mismatch) | Cannot use seg-loss to discover interpretability-optimal levels |
| Post-hoc ablation fails (non-unique Pareto + co-adaptation) | Cannot identify optimal subset from a single model at inference time |
| Manual L3+L4 with warm-start works (Dice=0.8656, eff.purity=0.649) | The level choice is principled and evidenced across 3 independent experiments |

**v7's core gap:** L3+L4 was selected manually. The pipeline has no data-driven justification for the level choice that would generalise to a new dataset or modality.

**v8 goal:** Replace the manual level selection with a **data-driven, two-phase pipeline** and improve prototype anatomical precision with an **Anatomical Locality Constraint (ALC)** loss.

---

## Why Plain M4 (No Attention) for Stage 1

In v7, Stage 29 used `proto_seg_ct_l1234_attn_noent.pth` as the diagnostic model — a
convenience choice because the checkpoint already existed. Using an attention model for
diagnosis introduces a subtle circularity:

- The L2 feedback loop (attention → encoder gradient) may artificially inflate L2 purity
- The diagnostic purity values are then partially a product of the attention mechanism
  rather than the pure segmentation objective

A plain M4 (no attention, no feedback loop) gives a cleaner purity signal: each level's
prototype quality reflects only what a segmentation-trained encoder can represent at that
scale, with no attention-mediated reinforcement.

---

## Pipeline Overview

```
Phase 1 — Diagnose
    Train plain M4 (no attention, all 4 levels)
    Compute per-level purity on training set
    Apply max-gap filter → auto-select surviving levels

Phase 2 — Refine
    Warm-start M2(surviving levels) from Phase 1 encoder
    Add ALC loss → improve anatomical prototype precision
    Evaluate: Dice, purity, centroid deviation
```

### Max-Gap Filter (data-driven, no hardcoded threshold)

```python
purities = sorted(purity_per_level.items(), key=lambda x: x[1])
# [(L1, 0.084), (L2, 0.195), (L3, 0.613), (L4, 0.689)]

gaps = [(purities[i+1][1] - purities[i][1], i) for i in range(len(purities)-1)]
cut = max(gaps, key=lambda x: x[0])[1]         # index of largest gap

low_purity  = [purities[i][0] for i in range(cut+1)]        # discard
high_purity = [purities[i][0] for i in range(cut+1, len(purities))]  # keep
```

CT expectation: gap(L2→L3) = 0.418 >> gap(L3→L4) = 0.076 → L3+L4 selected automatically.

---

## Research Questions

**RQ13:** Does the max-gap filter on plain M4 purity correctly and automatically select
L3+L4 for CT, without any hardcoded threshold?

**RQ14:** Does ALC improve prototype anatomical precision (purity ↑, centroid deviation ↓)
within the M2(L3+L4) architecture without significant Dice cost (ΔDice ≤ 0.01)?

**RQ15 (optional):** Does the full two-phase pipeline generalise to MR — i.e., does a
plain MR M4 model also show a clear purity gap that the max-gap filter resolves correctly?

---

## Stage 32 — M4 Diagnostic (CT)

### Task

Verify that a plain M4 model's per-level purity distribution has a clear max-gap that
the filter resolves to L3+L4. Use an existing checkpoint if suitable; retrain otherwise.

### Checkpoint verification

Candidate: `checkpoints/proto_seg_ct_pp2.pth` (ep=90, val_dice=0.8238, levels=None, attn=None)

Acceptance criteria for reuse:
- 3D test Dice ≥ 0.82
- All 4 proto levels present (levels=None = [1,2,3,4])
- No attention module

If not accepted: retrain plain M4 with standard config (same hyperparameters as cold-start
M2 but with PROTO_LEVELS=[1,2,3,4], USE_LEVEL_ATTENTION=False).

### Deliverable

- Per-level purity table: L1, L2, L3, L4
- Max-gap filter output: which levels survive
- Confirm: surviving levels == [3, 4]

### Config

```
Checkpoint (candidate) : checkpoints/proto_seg_ct_pp2.pth
Notebook               : notebooks/32_m4_diagnostic.ipynb
Output                 : results/v8/purity_m4_plain_ct.csv
```

### Tasks

- [ ] Load candidate M4, compute 3D Dice (verify ≥ 0.82)
- [ ] Compute purity per level (train set)
- [ ] Run max-gap filter, confirm L3+L4 selected
- [ ] Answer RQ13

---

## Stage 33 — Implement ALC Loss

### What is ALC

**Anatomical Locality Constraint:** each prototype's activation centroid should lie near
the expected anatomical location of the class it represents. For cardiac CT/MR, structural
positions are consistent across patients.

```
L_ALC = Σ_{k ∈ FG, m, l ∈ active_levels}  || centroid(A_{k,m,l}) - μ_k ||²

centroid(A) = Σ_{x,y} A(x,y) · (x,y) / Σ A(x,y)    ← differentiable (soft-argmax)
μ_k         = mean centroid of class k over training set (precomputed, fixed)
```

ALC is applied only to active proto levels (L3+L4), not to all encoder levels.

### Key design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Apply to which levels? | Active proto levels only (L3+L4) | Shallow levels are not semantic; ALC would be misleading |
| Normalise coordinates? | [0,1] relative to image size | Scale-invariant; same μ_k works for CT and MR |
| Phase? | Phase B onwards (after prototype projection) | Phase A prototypes are random; ALC before projection adds noise |
| λ_ALC | 0.01–0.1 (tune) | Start small; Dice should not drop >0.005 |

### Implementation

New file: `src/losses/alc_loss.py`

```python
def compute_anatomical_priors(train_loader, n_classes, device="cpu"):
    """Precompute mean centroid μ_k for each class from training labels."""
    ...
    return mu  # (K, 2) — normalised (y, x) ∈ [0, 1]

def alc_loss(heatmaps, mu, active_levels, foreground):
    """
    heatmaps : {level: (B, K, M, H_l, W_l)}
    mu       : (K, 2)  precomputed anatomical priors
    Returns scalar loss.
    """
    ...
```

Integrate into `ProtoSegLoss` as optional `lambda_alc` term.

### Tasks

- [ ] `src/losses/alc_loss.py` — ALC implementation
- [ ] Precompute μ_k for CT training set → `results/v8/anatomical_priors_ct.csv`
- [ ] Unit test: ALC loss decreases when centroids are close to μ_k
- [ ] Integrate into ProtoSegLoss (backward compatible: lambda_alc=0.0 by default)

---

## Stage 34 — M2(L3+L4) + ALC (CT)

### Hypothesis (RQ14)

ALC forces L3+L4 prototype activations to remain anatomically anchored. This should:
- Reduce centroid drift (prototypes migrating to background during training)
- Increase purity (spatially precise activation → less class overlap)
- Have minimal Dice cost (ALC is a prototype-level constraint, not an encoder constraint)

### Config

```
Stage 1 encoder  : checkpoints/proto_seg_ct_pp2.pth  (plain M4, Phase 1 of pipeline)
PROTO_LEVELS     : [3, 4]  (from max-gap filter output)
lambda_alc       : 0.05  (tune if needed)
Notebook         : notebooks/34_m2_alc.ipynb
Output ckpt      : checkpoints/proto_seg_ct_l3l4_alc.pth
Log              : results/v8/train_curve_proto_ct_l3l4_alc.csv
```

### Comparison baseline

Stage 29 (same architecture, no ALC):
- 3D Dice = 0.8656, eff.purity = 0.649, L4 purity = 0.774, L3 purity = 0.486

### Success criteria

| Metric | Stage 29 (no ALC) | Target (with ALC) |
|--------|-------------------|-------------------|
| 3D Dice | 0.8656 | ≥ 0.855 (ΔDice ≤ 0.01) |
| L3 purity | 0.486 | ≥ 0.55 |
| L4 purity | 0.774 | ≥ 0.80 |
| Effective purity | 0.649 | ≥ 0.70 |
| Mean centroid deviation | — | < 20px (new metric) |

### New metric: centroid deviation

```python
centroid_deviation_l = mean over (k, m, batch) of || centroid(A_{k,m,l}) - μ_k ||
```

Computed on test set. Lower = prototypes more anatomically anchored.

### Tasks

- [x] Notebook `34_m2_alc.ipynb` (training + evaluation)
- [x] Compute centroid deviation pre/post ALC
- [x] Proto quality: purity, AP, compactness, effective purity, centroid deviation
- [x] Comparison table vs Stage 29
- [x] Answer RQ14 — NOT MET (1/5). See findings below.

### Stage 34 Results (ALC on L3+L4, λ=0.05)

| Metric | Stage 29 | Stage 34 | Target |
|--------|----------|----------|--------|
| 3D Dice | 0.8656 | **0.8478** | ≥ 0.855 ❌ |
| L3 purity | 0.486 | **0.646** | ≥ 0.55 ✅ |
| L4 purity | 0.774 | **0.670** | ≥ 0.80 ❌ |
| Effective purity | 0.649 | **0.661** | ≥ 0.70 ❌ |
| Mean centroid deviation | — | **37.0px** | < 20px ❌ |

**Root cause:** ALC is level-selective in its effect.
- L3 (32×32, spatial level): purity +0.160 — anchoring is compatible with spatial features
- L4 (16×16, semantic level): purity −0.104 — anchoring conflicts with semantic features
- Applying ALC at L4 also increases the effective constraint on the segmentation loss,
  causing the Dice drop (−0.018, concentrated in harder patient ct_1019)

**Fix → Stage 34b:** Apply ALC to L3 only (`alc_levels=[3]`).

---

## Stage 34b — M2(L3+L4) + ALC on L3 only (CT)

**Motivation:** Stage 34 showed ALC helps L3 but hurts L4 and Dice.
Removing the L4 constraint should preserve the L3 purity gain while recovering L4 purity and Dice.

```
Stage 1 encoder : checkpoints/proto_seg_ct_pp2.pth
PROTO_LEVELS    : [3, 4]   (model architecture unchanged)
ALC_LEVELS      : [3]      (ALC applied to L3 only)
lambda_alc      : 0.05
Notebook        : notebooks/34b_m2_alc_l3only.ipynb
Output ckpt     : checkpoints/proto_seg_ct_l3l4_alc_l3only.pth
Log             : results/v8/train_curve_proto_ct_l3l4_alc_l3only.csv
```

**Success criteria (adjusted):**

| Metric | Stage 29 | Stage 34 | Target (34b) |
|--------|----------|----------|--------------|
| 3D Dice | 0.8656 | 0.8478 | ≥ 0.855 |
| L3 purity | 0.486 | 0.646 | ≥ 0.60 |
| L4 purity | 0.774 | 0.670 | ≥ 0.75 |
| Effective purity | 0.649 | 0.661 | ≥ 0.68 |
| L3 centroid deviation | — | 37.2px | < 30px |

### Tasks

- [ ] Notebook `34b_m2_alc_l3only.ipynb` (training + evaluation)
- [ ] Verify L4 purity recovers without ALC pressure
- [ ] Verify Dice recovers toward Stage 29
- [ ] Answer RQ14

---

## Stage 35 — MR Validation (optional, time-permitting)

### Purpose

If the pipeline only works on CT (where L3/L4 purity advantage is already known), it is
not generalisable. MR validation tests whether:
1. Plain MR M4 also shows a clear purity gap (max-gap filter gives a non-trivial answer)
2. ALC anatomical priors transfer to MR (or need separate μ_k computation)

### Config

```
MR M4 candidate : checkpoints/proto_seg_mr.pth  (ep=80, val=0.7459, levels=None)
                  (verify or retrain if Dice < 0.75)
μ_k (MR)        : results/v8/anatomical_priors_mr.csv  (recompute from MR train set)
Notebook        : notebooks/35_mr_pipeline.ipynb
```

Note: MR anatomy is consistent with CT in location (same structures, similar centroids
in normalised image space), but contrast differs. Separate μ_k recommended.

### Tasks

- [ ] Verify MR M4 checkpoint; retrain if needed
- [ ] Compute MR purity per level; run max-gap filter
- [ ] Precompute μ_k for MR
- [ ] Train MR M2(selected levels) + ALC
- [ ] Answer RQ15

---

## Stage 36 — Report v8

`report/v8/report-v8.md`

Sections:
1. Introduction — v7 findings motivating v8
2. Max-gap filter — method, CT result, RQ13 verdict
3. ALC — method, implementation, training dynamics
4. CT results — comparison table Stage 29 vs Stage 34
5. MR results (if Stage 35 completed)
6. Full pipeline summary — two-phase end-to-end
7. Limitations and future directions

---

## Stage Overview

| Stage | Name | Deliverable | Status |
|-------|------|-------------|--------|
| 32 | M4 Diagnostic (CT) | `notebooks/32_m4_diagnostic.ipynb` | ⬜ |
| 33 | ALC Implementation | `src/losses/alc_loss.py` | ✅ |
| 34 | M2(L3+L4) + ALC L3+L4 | `notebooks/34_m2_alc.ipynb` | ✅ (NOT MET — ALC hurts L4) |
| 34b | M2(L3+L4) + ALC L3 only | `notebooks/34b_m2_alc_l3only.ipynb` | ⬜ |
| 35 | MR Validation | `notebooks/35_mr_pipeline.ipynb` | ⬜ optional |
| 36 | Report v8 | `report/v8/report-v8.md` | ⬜ |

---

## Decision Tree

```
Stage 32: M4 Diagnostic ──── does max-gap filter select L3+L4?
    │   YES (expected) ────→ proceed to Stage 33
    │   NO              ────→ investigate: purity gap unclear → reconsider pipeline
    │
Stage 33: ALC impl      ──── unit tests pass?
    │   YES             ────→ Stage 34
    │   NO              ────→ debug ALC gradient / coordinate normalisation
    │
Stage 34: M2+ALC (CT)   ──── purity ↑, Dice stable?
    │   YES (full)      ────→ RQ14 MET → Stage 35 (if time) → Stage 36
    │   YES (partial)   ────→ RQ14 PARTIAL → Stage 36 with honest reporting
    │   NO (Dice drop)  ────→ reduce λ_ALC and retrain
    │
Stage 35: MR (optional) ──── max-gap filter generalises?
    │   YES             ────→ RQ15 MET → strong generalisability claim
    │   NO              ────→ document as limitation; pipeline is CT-specific
    │
Stage 36: Report v8
```

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `proto_seg_ct_pp2.pth` too weak (3D Dice < 0.82) | Low | Retrain plain M4 (known config, ~3h) |
| Max-gap filter selects wrong levels (gap not at L2→L3) | Very low | Gap is 0.418 vs next largest 0.076 — very robust |
| ALC hurts Dice (λ_ALC too large) | Medium | Start λ_ALC=0.01; sweep to 0.1 |
| ALC doesn't improve purity (prototypes already anchored) | Medium | Stage 29 L3 purity=0.486 has room to improve; L4=0.774 may not |
| MR purity gap unclear (no clean separation) | Medium | Document as CT-specific finding; don't over-claim |

---

## File Structure (v8 additions)

```
plan/
  execution-plan-v8.md

src/
  losses/
    alc_loss.py                            # ALC: anatomical_priors + alc_loss()

notebooks/
  32_m4_diagnostic.ipynb                   # verify M4, run max-gap filter
  34_m2_alc.ipynb                          # Stage 34: train + eval
  35_mr_pipeline.ipynb                     # Stage 35: MR (optional)

results/v8/
  anatomical_priors_ct.csv                 # μ_k for CT (K×2)
  anatomical_priors_mr.csv                 # μ_k for MR (K×2, optional)
  purity_m4_plain_ct.csv                   # Stage 32: per-level purity
  train_curve_proto_ct_l3l4_alc.csv        # Stage 34: training log
  comparison_table_v8.csv                  # Stage 34: vs Stage 29
  centroid_deviation_ct.csv               # Stage 34: new metric

checkpoints/
  proto_seg_ct_l3l4_alc.pth               # Stage 34
  proto_seg_mr_l3l4_alc.pth               # Stage 35 (optional)

report/v8/
  report-v8.md
```
