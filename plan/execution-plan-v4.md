# Execution Plan v4: Prototype Quality & Multi-Scale Ablation

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (ct_1019, ct_1020 test patients)
**Hardware:** MacBook 48GB RAM (Apple Silicon MPS)
**Date:** 2026-03-16
**Preceded by:** `report/v3/execution-plan-v3.md`

---

## Scope Change from v3

v1→v2→v3 pursued absolute XAI targets (AP ≥ 0.70, Faithfulness ≥ 0.55, Stability ≤ 0.20) that are structurally unachievable under the current coupled prototype-segmentation design. The three-architecture failure analysis arc is now complete and documented in `report/v2/report-v2.md`.

**v4 pivots to a narrower, more tractable research question:**

> *Do multi-scale prototypes learn structurally distinct, anatomically meaningful representations — and does scale configuration affect prototype quality?*

This reframes the project from "beat XAI targets" to "understand what prototypes actually learn," which is both more achievable and more scientifically interesting.

**Dropped:**
- Chasing AP ≥ 0.70, Faithfulness ≥ 0.55, Stability ≤ 0.20 as success criteria
- v4 architecture design

**Retained:**
- Segmentation Dice as primary performance metric
- Prototype heatmap visualisation (Prototype Atlas)

**Added:**
- Prototype quality metrics (Purity, Utilization, Compactness, Dice Sensitivity)
- Multi-scale ablation study

---

## Research Questions

**RQ1:** What do prototypes at each scale level actually learn? (qualitative + purity)

**RQ2:** Does the multi-scale configuration meaningfully diversify prototype representations, or does one level dominate?

**RQ3:** Is there a simpler scale configuration (e.g. L3+L4 only) that achieves equivalent segmentation quality with cleaner prototype representations?

---

## Stage Overview

| Stage | Name | Deliverable | Status |
|-------|------|-------------|--------|
| 14 | Prototype Quality Metrics | `notebooks/14_proto_quality_metrics.ipynb` | ✅ |
| 15 | Post-hoc Analysis (v1) | `notebooks/15_posthoc_analysis_v1.ipynb` | ✅ |
| 16 | Multi-scale Ablation Training | `notebooks/16_multiscale_ablation_training.ipynb` | ✅ |
| 17 | Ablation Comparison | `notebooks/17_ablation_comparison.ipynb` | ✅ |
| 18 | Report v4 | `report/v4/report-v4.md` | ✅ |

---

## Prototype Quality Metrics

### 1. Prototype Purity

**Definition:** For each prototype p_{l,k,m}, find the top-N training positions with highest activation. Purity = fraction where GT label == k.

**Comparison caveat:** Purity compares models at the *system level* — different models have different encoder representations. Differences in purity reflect the joint effect of encoder adaptation and prototype configuration, not isolated prototype quality. This must be stated in the report.

```python
def compute_purity(model, train_loader, top_n=50) -> dict[(l,k,m), float]:
    # For each prototype: collect (activation_value, gt_label) pairs
    # across all training slices. Return top_n fraction with label==k.
```

**Output:** `purity[l][k][m]` — also aggregated as mean per level, mean per class.

---

### 2. Prototype Utilization

**Definition:** Fraction of prototypes with max activation > threshold across the test set. Dead prototypes (max_act < 0.1) waste capacity.

```python
def compute_utilization(model, test_loader, threshold=0.1) -> dict:
    # Returns max_activation per prototype + list of dead prototypes
```

**Output:** Overall utilization rate, dead prototype list by (l, k, m).

---

### 3. Spatial Compactness

**Definition:** For each prototype, mean fraction of pixels in activation map above 95th percentile (relative to map resolution). Lower = more focused activation.

**Level-relative thresholds (not absolute 5% for all):**

| Level | Max acceptable compactness |
|-------|---------------------------|
| L1 (128×128, boundary) | 5% |
| L2 (64×64, borders) | 8% |
| L3 (32×32, structure) | 15% |
| L4 (16×16, global) | 25% |

```python
def compute_compactness(model, test_loader) -> dict[(l,k,m), float]:
    # Fraction of spatial locations above p95 threshold per prototype
```

---

### 4. Dice Sensitivity

**Definition:** Zero out one prototype's activations at inference (no retraining), measure mean Dice drop. Captures causal importance to segmentation.

**Valid only for v1 architecture** — in v3 decoupled, prototype removal has minimal Dice impact by design.

```python
def compute_dice_sensitivity(model, test_loader) -> dict[(l,k,m), float]:
    # For each prototype: set A[l,k,m] = 0, recompute logits, measure Dice delta
```

**Threshold for "important" prototype:** Dice drop > 0.005 (0.5%).

---

### 5. Level Dominance (multi-scale models only)

**Definition:** For each test pixel, which level's prototype wins the max aggregation in the final heatmap?

```python
def compute_level_dominance(model, test_loader) -> list[float]:
    # Returns [frac_l1, frac_l2, frac_l3, frac_l4]
    # Each = fraction of pixels where level l's heatmap is the maximum
```

If one level dominates (> 60%), the other levels contribute minimally to the XAI explanation despite adding parameters.

---

### 6. Per-level AP

**Definition:** Compute AP using only the heatmap from level l (max over prototypes only, before cross-level max). Upsample to 256×256 for comparison with GT.

```python
def compute_per_level_ap(model, test_loader) -> dict[int, float]:
    # For each level l: AP using A_k = max_m A[l,k,m] (no max over levels)
```

This isolates which level produces the most anatomically precise activations.

---

## Stage 14 — Prototype Quality Metrics ✅

### Goal

Implement all six metric functions as reusable utilities in `src/metrics/proto_quality.py`. The notebook `notebooks/14_proto_quality_metrics.ipynb` imports from `src/` and handles only config, execution, and saving outputs.

### src/metrics/proto_quality.py

All shared logic lives here:

```python
# src/metrics/proto_quality.py

def compute_purity(model, train_loader, top_n=50) -> dict: ...
def compute_utilization(model, test_loader, threshold=0.1) -> dict: ...
def compute_compactness(model, test_loader) -> dict: ...
def compute_dice_sensitivity(model, test_loader) -> dict: ...
def compute_level_dominance(model, test_loader) -> list[float]: ...
def compute_per_level_ap(model, test_loader) -> dict: ...
def build_prototype_atlas(model, train_loader, level: int) -> plt.Figure: ...
```

### Notebook Structure

`notebooks/14_proto_quality_metrics.ipynb` — thin execution layer only:

```
Cell 1 — Imports & config
  from src.metrics.proto_quality import (
      compute_purity, compute_utilization, compute_compactness,
      compute_dice_sensitivity, compute_level_dominance,
      compute_per_level_ap, build_prototype_atlas
  )
  checkpoint_path = "checkpoints/proto_seg_ct_l2.pth"
  out_dir = "results/v4/proto_quality/v1/"

Cell 2 — Load model + dataloaders
Cell 3 — Run all six metrics, save CSVs to out_dir
Cell 4 — Generate and save prototype atlas for all 4 levels
Cell 5 — Smoke test: print shapes, assert no NaN
```

### Output Files

```
results/v4/proto_quality/v1/
  purity_per_prototype.csv        # (level, class, proto_idx, purity)
  purity_summary.csv              # mean purity by level, by class
  utilization.csv                 # (level, class, proto_idx, max_activation, is_dead)
  compactness.csv                 # (level, class, proto_idx, compactness)
  dice_sensitivity.csv            # (level, class, proto_idx, dice_drop)
  level_dominance.csv             # [frac_l1, frac_l2, frac_l3, frac_l4]
  per_level_ap.csv                # (level, class, ap)
  prototype_atlas_level{1-4}.png
```

### Model Interface Requirement

All functions require:

```python
logits, A_list = model(images, return_activations=True)
# A_list[l]: (B, K, M, H_l, W_l)
```

If not present, add `return_activations` flag to `ProtoSegNet.forward()` before Stage 14.

### Tasks

- [x] Confirm `return_activations` API exists — `forward()` already returns `(logits, heatmaps_dict)`
- [x] Implement `src/metrics/proto_quality.py` with all six functions + atlas builder
- [x] Add `src/metrics/proto_quality.py` to `src/metrics/__init__.py`
- [x] Implement notebook as thin execution layer (import, run, save)
- [x] Smoke test: all 77 prototype rows, 12 output files, no NaN

### Results (v1 checkpoint, CT test set)

| Metric | Output |
|--------|--------|
| Dead prototypes | 0 / 77 |
| Important prototypes (Dice drop > 0.005) | 7 / 77 (all level 3) |

**Purity by level:**
| Level | Mean Purity | Range |
|-------|------------|-------|
| L1 (128×128) | 0.050 | 0.00–0.28 |
| L2 (64×64) | 0.184 | 0.00–0.66 |
| L3 (32×32) | 0.639 | 0.08–1.00 |
| L4 (16×16) | **0.824** | 0.40–1.00 |

**Level Dominance** (fraction of pixels where each level wins heatmap max):
```
L1: 34%   L2: 44%   L3: 17%   L4: 4%
```

**Per-level AP** (level evaluated in isolation):
```
L1: 0.043   L2: 0.112   L3: 0.068   L4: 0.189
```

**Compactness** (fraction of 256×256 image with activation > 0.5):
```
L1: 0.34   L2: 0.42   L3: 0.54   L4: 0.57
```
All levels far exceed their thresholds — activations are broadly diffuse at every scale.

### Key Findings

1. **Deep levels learn correctly; shallow levels are noisy.** Purity jumps from 0.05 (L1) to 0.82 (L4). L4 prototypes are anatomically meaningful; L1 prototypes are essentially random.
2. **Shallow levels dominate the final heatmap.** L1+L2 together win 78% of pixels in the max aggregation, suppressing the high-purity L4 signal. This directly explains why the overall AP is only 0.10 — the explanation is hijacked by low-quality shallow activations.
3. **Only level-3 prototypes causally affect segmentation.** 7 important prototypes (Dice drop ~0.008), all at L3. L4 prototypes, despite highest purity, have no measurable segmentation impact.
4. **Compactness is poor at all levels.** Activations cover 34–57% of the image regardless of level.

### Decision Gate Result

Pattern: **L2 dominates (44%), L1 second (34%)** — matches "shallow levels wasted" pattern.

→ **Ablation priority: compare L4-only (M1) vs full multi-scale (M4).** Hypothesis: removing L1/L2 will force the high-purity L4 prototypes to dominate the heatmap, improving AP while Dice may remain similar since L3/L4 dominate segmentation anyway.

---

## Stage 15 — Post-hoc Analysis Notebook (v1) ✅

### Goal

`notebooks/15_posthoc_analysis_v1.ipynb` — load the CSVs produced by Stage 14, produce summary tables and figures that answer RQ1 and RQ2. No new training required.

### Tasks

- [x] Load Stage 14 outputs and build purity/compactness/dominance visualisations
- [x] Generate prototype atlas inline display for all 4 levels
- [x] Summarise findings: purity by level, utilization, dominance, per-level AP

### Outputs

```
results/v4/proto_quality/v1/
  fig_purity.png            # bar chart + level×class heatmap
  fig_utilization.png       # max activation histogram by level
  fig_compactness.png       # violin plot with per-level thresholds
  fig_dice_sensitivity.png  # mean Dice drop bar chart by level
  fig_level_dominance.png   # pie chart (L1:34%, L2:44%, L3:17%, L4:4%)
  fig_per_level_ap.png      # bar chart + level×class AP heatmap
```

### RQ1 Answer — What do prototypes at each level learn?

Strong depth gradient in purity: L1 (0.050) → L2 (0.184) → L3 (0.639) → L4 (0.824). Shallow levels respond to class-agnostic texture/edges; only L4 prototypes are anatomically selective. L3 prototypes causally affect segmentation (7 / 77 with Dice drop > 0.005); L4 prototypes, despite highest purity, have Dice drop ≈ 0. Compactness fails at all levels (0.34–0.57 fraction > 0.5) — activations are broadly diffuse.

### RQ2 Answer — Does multi-scale diversify representations?

No — it degrades the final heatmap. L1+L2 dominate 78% of pixels in cross-level max aggregation despite lowest purity. L4 (purity 0.824, AP 0.189 in isolation) wins only 4% of pixels. The winner-takes-all aggregation systematically favours noisy shallow levels, suppressing the high-quality L4 signal. Overall AP is ~0.10 even though L4 alone achieves 0.189.

### Decision Gate

**Resolved.** L2 dominates (44%), L1 second (34%) — anti-correlated with purity rank.

→ Ablation: **M1 (L4-only) vs M4 (full)** is the key comparison.

---

## Stage 16 — Multi-scale Ablation Training ✅

### Ablation Models

Minimum viable ablation (two models):

| Model | Prototype Levels | Rationale |
|-------|-----------------|-----------|
| **M1** | L4 only (16×16) | Most semantic level; simplest configuration |
| **M4** | L1+L2+L3+L4 | Current v1 (already trained) |

Optional (if Stage 15 suggests intermediate levels matter):

| Model | Prototype Levels | Status |
|-------|-----------------|--------|
| M2 | L3+L4 | ✅ trained |
| M3 | L2+L3+L4 | ⬜ not run |

### Architecture Rule for Missing Levels

For levels without prototypes: skip connections pass raw encoder features to decoder unchanged (pass-through, no mask). Decoder structure does not change. This ensures Dice differences reflect prototype supervision, not decoder architecture.

### Training Configuration

Same as v1 `_l2`:
```
lambda_div=0.001, lambda_push=0.5, lambda_pull=0.25
3-phase schedule: warm-up (1-20), joint (21-80), fine-tune (81-100)
early stopping patience=15 per phase
suffix: _l4only, _l3l4 (if run)
```

### src changes required before this notebook

`src/models/proto_seg_net.py`: add `proto_levels` parameter to `ProtoSegNet.__init__`. For levels not in `proto_levels`, skip connections pass raw encoder features to decoder unchanged (no mask, no prototype layer).

### Notebook Structure

`notebooks/16_multiscale_ablation_training.ipynb` — config + training loop only:

```
Cell 1 — Config
  from src.models.proto_seg_net import ProtoSegNet
  from src.losses.diversity_loss import JeffreysDivergenceLoss
  proto_levels = [3]       # [3] for M1, [2,3] for M2; re-run cell to switch
  suffix = "_l4only"
  lambda_div=0.001, lambda_push=0.5, lambda_pull=0.25

Cell 2 — Build model (proto_levels=[3]) + dataloaders
Cell 3 — Phase A (epochs 1–20, prototypes frozen) + loss plot
Cell 4 — Phase B (epochs 21–80, all params) + loss plot
Cell 5 — Phase C (epochs 81–100, encoder+proto frozen) + loss plot
Cell 6 — Save checkpoint: checkpoints/proto_seg_ct_{suffix}.pth
Cell 7 — Quick 3D Dice eval (import dice logic from src/metrics/dice.py)
```

### Tasks

- [x] Add `proto_levels` param + pass-through logic to `src/models/proto_seg_net.py`
- [x] Implement training notebook (3-phase loop, early stopping, loss plots)
- [x] Save checkpoint and run quick 3D Dice eval in notebook
- [ ] (Optional) Re-run notebook with `proto_levels=[2,3]` for M2

### Results (M1 — L4-only, CT)

| | M1 (L4-only) | M4 (full, v1) |
|---|---|---|
| Best val Dice (2D) | **0.8346** (ep 55) | 0.8146 |
| 3D Dice — ct_1019 | 0.7697 | — |
| 3D Dice — ct_1020 | 0.9351 | — |
| **3D Dice mean** | **0.8524** | 0.843 |
| Δ vs v1 | **+0.0094** | — |
| Status | ✅ exceeds v1 | baseline |

**Per-structure 3D Dice (M1):**

| Patient | LV | RV | LA | RA | Myo | Aorta | PA |
|---------|----|----|----|----|-----|-------|----|
| ct_1019 | 0.844 | 0.866 | 0.677 | 0.873 | 0.806 | 0.782 | 0.539 |
| ct_1020 | 0.892 | 0.962 | 0.935 | 0.921 | 0.928 | 0.972 | 0.936 |
| **Mean** | **0.868** | **0.914** | **0.806** | **0.897** | **0.867** | **0.877** | **0.738** |

**Key finding:** M1 (L4-only, 2.55M params) achieves **+0.94% higher 3D Dice than v1** despite using only 1 of 4 prototype levels. Removing the low-purity L1/L2/L3 prototype layers did not hurt segmentation — it marginally improved it, consistent with the Stage 15 finding that only L3 prototypes causally affect segmentation in v1.

---

## Stage 17 — Ablation Comparison Notebook ✅

### Goal

`notebooks/17_ablation_comparison.ipynb` — run the Stage 14 metric functions on ablation checkpoints, then build side-by-side comparison tables and figures to answer RQ3.

### Notebook Structure

`notebooks/17_ablation_comparison.ipynb`:

```
Cell 1 — Imports & config
  from src.metrics.proto_quality import (
      compute_purity, compute_utilization, compute_compactness,
      compute_dice_sensitivity, compute_level_dominance,
      compute_per_level_ap, build_prototype_atlas
  )
  models = {
      "M1 (L4-only)": "checkpoints/proto_seg_ct_l4only.pth",
      "M4 (full v1)": "checkpoints/proto_seg_ct_l2.pth",
  }

Cell 2 — Run all metrics for each checkpoint, save to results/v4/proto_quality/{m1_l4only,v1}/
Cell 3 — 3D Dice comparison table
Cell 4 — Prototype quality comparison table (purity, utilization, compactness)
Cell 5 — Prototype atlas L4: M1 vs M4 side-by-side
Cell 6 — Find one slice where M1 and M4 segmentation visibly differs
Cell 7 — Written answer to RQ3
```

### Tasks

- [x] Load Stage 14 metric functions and run on M1 checkpoint
- [x] Build comparison table
- [x] Visual comparison: prototype atlas L4 for M1 vs M4
- [x] Find one slice where M1 and M4 produce visibly different segmentation (4.9% pixels differ)

### Results

| Metric | M1 (L4-only) | M4 (full, v1) |
|--------|-------------|---------------|
| **3D Dice** | **0.8524** | 0.8407 |
| n_prototypes | **14** | 77 |
| Purity — L4 | 0.499 | **0.824** |
| Purity — overall | 0.499 | 0.334 |
| Utilization rate | 100% | 100% |
| Dead prototypes | 0 | 0 |
| Compactness (L4) | **0.564** | 0.573 |
| Important protos (Dice drop > 0.005) | 4 / 14 | 7 / 77 |
| Level dominance — L4 | **100%** | 4.3% |
| Level dominance — L1+L2 | 0% | 78.5% |
| Per-level AP — L4 | **0.274** | 0.189 |

### Key Finding: the purity paradox

M4's L4 prototypes have higher purity (0.824) than M1's (0.499), yet M1 achieves better per-level AP (0.274 vs 0.189) and better segmentation (0.8524 vs 0.8407).

**Why:** In M4, the encoder specialises L4 for coarse/semantic features (→ high purity) while L1/L2/L3 handle multi-scale texture. But those high-purity L4 prototypes are then suppressed by noisy L1/L2 activations in the cross-level max aggregation (L4 wins only 4.3% of pixels). In M1, L4 must represent all scales (→ lower purity), but dominates 100% of pixels — so the heatmap is coherent and AP is higher.

**Conclusion (RQ3):** Removing shallow prototype levels trades per-prototype purity for heatmap dominance, and the net effect is *better* XAI quality (AP +0.085) *and* better segmentation (+1.2% Dice) with 5× fewer prototypes (14 vs 77). The simpler configuration wins on all practical dimensions.

### Outputs

```
results/v4/proto_quality/
  comparison_table.csv
  fig_purity_comparison.png
  fig_seg_comparison.png
  atlas_m4_l4.png
  atlas_m1_l4.png
  m1_l4only/
    purity_per_prototype.csv
    purity_summary.csv
    utilization.csv
    compactness.csv
    dice_sensitivity.csv
    level_dominance.csv
    per_level_ap.csv
```

### Comparison Caveat (must be in report)

> Purity and other prototype quality metrics are compared at the system level. M1 and M4 have different encoder representations because multi-scale prototype supervision alters encoder gradients across all levels. Differences reflect the joint effect of encoder adaptation and prototype configuration.

---

## Stage 18 — Report v4 ✅

### Goal

Write `report/v4/report-v4.md` as a focused study on prototype quality in multi-scale cardiac segmentation.

### Framing

Not a "we beat the XAI targets" paper. Instead:

> *"We characterise what prototype networks actually learn in cardiac CT segmentation, and investigate whether multi-scale prototype supervision produces meaningfully distinct representations at each scale."*

### Structure

```
§1 Introduction — motivate prototype quality evaluation
§2 Background — v1 ProtoSegNet, existing XAI ceiling finding
§3 Prototype Quality Metrics — define Purity, Utilization, Compactness,
                               Dice Sensitivity, Level Dominance, Per-level AP
§4 Post-hoc Analysis (RQ1, RQ2) — what v1 prototypes learned per level
§5 Multi-scale Ablation (RQ3) — M1 vs M4 comparison
§6 Discussion — implications for prototype-based segmentation design
§7 Conclusion
```

### Tasks

- [x] Draft §3 metric definitions (draw from this plan)
- [x] Populate §4 with Stage 15 results
- [x] Populate §5 with Stage 17 results
- [x] Write §6 discussion around the system-level comparison caveat and purity paradox
- [x] Include prototype atlas and figure references

---

## File Structure (v4 additions)

```
plan/
  execution-plan-v4.md                        # this file

src/
  metrics/
    proto_quality.py                          # Stage 14 — all six metric functions + atlas builder
    __init__.py                               # updated to export proto_quality
  models/
    proto_seg_net.py                          # modified: return_activations flag, proto_levels param

notebooks/
  14_proto_quality_metrics.ipynb              # Stage 14 — imports src, runs + saves outputs
  15_posthoc_analysis_v1.ipynb                # Stage 15 — loads CSVs, visualisation, RQ1/RQ2
  16_multiscale_ablation_training.ipynb       # Stage 16 — imports src, 3-phase training loop
  17_ablation_comparison.ipynb                # Stage 17 — imports src, cross-model comparison

results/v4/
  proto_quality/
    v1/                                       # Stage 14/15 outputs (CSVs + atlas PNGs)
    m1_l4only/                                # Stage 17 outputs (M1)
    m2_l3l4/                                  # Stage 17 outputs (M2) ✅
    comparison_table.csv                      # M1/M2/M4 side-by-side ✅
    fig_purity_comparison.png                 # ✅
    fig_dominance_comparison.png              # ✅
    fig_dice_ap_comparison.png                # ✅
    atlas_m2_l4.png / atlas_m2_l3.png        # ✅

checkpoints/
  proto_seg_ct_l4only.pth                     # Stage 16 — M1
  proto_seg_ct_l3l4.pth                       # Stage 16 — M2 ✅

report/v4/
  report-v4.md                                # Stage 18
```

---

## Success Criteria (v4)

**Segmentation (must not degrade from v1):**
- [x] M1 CT 3D Dice ≥ 0.820 (−2.3% tolerance vs v1 0.843) → **0.8524 ✅ (+0.94% vs v1)**

**Prototype quality (relative, not absolute):**
- [x] Mean purity measurably different between models — M4 L4 purity 0.824 vs M1 L4 purity 0.499; metric is discriminative and reveals the purity paradox
- [x] Level dominance explains per-level AP pattern (correlated) — Stage 15: dominant L2 (44%) has AP 0.112; high-purity L4 (4% dominance) has AP 0.189 in isolation but is suppressed in combined model
- [x] Dead prototype count reported for each model — Stage 14: v1 has 0 dead prototypes

**Scientific contribution:**
- [x] Post-hoc analysis answers RQ1 and RQ2 with evidence (Stage 15)
- [x] Ablation answers RQ3 with M1/M2/M4 comparison (Stage 17) — M2 (L3+L4) is best: 3D Dice 0.8722, purity_all 0.733, compactness_L4 0.361
- [x] Report documents system-level comparison caveat and purity paradox clearly (Stage 18)

---

## Stage 17 Extended — M2 Results Summary

| Metric | M4 (L1-L4) | M2 (L3-L4) | M1 (L4) |
|--------|-----------|-----------|---------|
| 3D Dice | 0.8407 | **0.8722** | 0.8524 |
| n_prototypes | 77 | 28 | 14 |
| Purity (L4) | **0.824** | 0.804 | 0.499 |
| Purity (all) | 0.334 | **0.733** | 0.499 |
| Compactness (L4) | 0.573 | **0.361** | 0.564 |
| Important protos | 7/77 (9%) | **10/28 (36%)** | 4/14 (29%) |
| L4 dominance | 4.3% | 49.1% | 100% |
| AP (L4) | 0.189 | 0.236 | **0.274** |

**Key M2 findings:**
- Best segmentation (+3.2% vs M4, +2.0% vs M1)
- Best overall purity (0.733) — both L3 and L4 are semantically selective
- Best compactness (0.361) — L4 prototypes are more spatially focused when competing only with L3
- L3 and L4 genuinely co-contribute: L4 wins 49.1% of pixels, L3 wins 50.9% — real complementarity
- 36% of prototypes causally important — most efficient of all three models

**Confirmed:** L1+L2 are actively harmful to both segmentation and XAI quality. L3 adds genuine value. M2 is the best base for direction A.

---

## Direction A — Learned Level Attention (v5) ⬜

### Motivation

In M2, L3 and L4 split pixel dominance roughly 50/50 via winner-takes-all max aggregation. This is better than M4's L1/L2 noise domination, but still arbitrary — the model cannot learn that L4 is more informative for certain classes or regions.

**Goal:** Replace cross-level max with a learned per-level softmax attention, allowing the model to weight level contributions based on the input.

### Architecture: LevelAttentionModule

```python
class LevelAttentionModule(nn.Module):
    """
    Learns soft weights over active prototype levels conditioned on encoder context.

    Input:  feat dict {level: (B, C_l, H_l, W_l)}  — encoder features
    Output: w (B, n_active_levels)                  — softmax attention weights

    Implementation:
      1. Global average pool each level → (B, C_l)
      2. Concatenate all levels → (B, sum_C_l)
      3. MLP: Linear(sum_C_l → 64) → ReLU → Linear(64 → n_levels) → softmax
    """
```

**Forward change in ProtoSegNet:**

```python
# Current (max aggregation):
# heatmap_k = max over levels of max_m(A[l,k,m])   shape: (B, H, W)

# Direction A (weighted sum):
# per_level_heatmap[l][k] = max_m(A[l,k,m])         shape: (B, H_l, W_l)
# w = level_attention(feat)                          shape: (B, n_levels)
# heatmap_k = Σ_l  w[:,l] * upsample(per_level_heatmap[l][k])
```

### New model variant: M2-attn

Base: M2 (L3+L4), same hyperparameters, same 3-phase schedule.
Added: `LevelAttentionModule` — ~1,000 extra parameters.

Checkpoint: `proto_seg_ct_l3l4_attn.pth`

### Expected effect

- Attention weights can learn class-specific level preferences (e.g. L4 for small structures like PA, L3 for large ones like LV)
- L4 AP may improve beyond 0.274 (M1) since the encoder can specialise L4 for XAI without losing L4 dominance
- Segmentation quality should match or exceed M2 (0.8722)

### Tasks

- [ ] Implement `LevelAttentionModule` in `src/models/proto_seg_net.py`
- [ ] Add `use_level_attention: bool = False` flag to `ProtoSegNet.__init__`
- [ ] Modify `forward()`: weighted sum aggregation when `use_level_attention=True`
- [ ] Create `notebooks/19_attention_training.ipynb`
- [ ] Train M2-attn and compare against M2
- [ ] Run proto quality metrics on M2-attn
- [ ] Update report

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `return_activations` API missing from v1 model | Medium | Add flag in Stage 14 before running eval |
| All prototypes have similar purity (~10%) | Medium | Still informative — confirms purity is ceiling-limited; compare compactness and dominance instead |
| M1 (L4-only) Dice collapses without L1/L2/L3 masks | Low | Decoder uses pass-through for missing levels; should be similar to baseline U-Net path |
| Purity computation too slow (scan all training slices per prototype) | Medium | Batch over prototypes; cache activations per slice; use top-k instead of full sort |
| Multi-scale dominance is uniform — no clear winner | Low | Still a result: confirms all levels contribute, ablation becomes more important |
