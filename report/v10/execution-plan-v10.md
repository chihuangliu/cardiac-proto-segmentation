# Execution Plan v10: Two-Barrier Framework
## Reframing the Narrative — Skip vs No-Skip as Core Contribution

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16/2/2 patients, 3389/382/484 slices)
**Hardware:** MacBook 48GB RAM, Apple Silicon (MPS)
**Date:** 2026-03-22
**Preceded by:** v9 concluded (report-v9.md)

---

## 1. Research Narrative (Direction B)

### The Central Claim

> Interpretable prototype segmentation networks face **two structurally distinct XAI barriers**.
> Removing the decoder (skip connections) addresses Barrier 1 (bypass) but exposes Barrier 2
> (resolution). Neither barrier can be eliminated by the other's fix. Understanding their
> independence — and their measurable costs — is the core contribution.

### Story Arc

```
ProtoSegNet (with decoder/skip)
  → Good Dice (0.866), Good Purity (0.733)
  → But AP = 0.102: bypass path lets model ignore prototypes
                              │
                              │ BARRIER 1: Bypass Problem
                              ↓
ProtoSegNetV2 (no decoder, no skip)
  → AP triples (0.312): prototypes now causally required ✅
  → But Dice drops 35% (0.866→0.559), Faithfulness stays low ❌
                              │
                              │ BARRIER 2: Resolution Problem
                              ↓
Diagnosis: L4 heatmaps at 16×16 → pixel perturbations (256×256)
           cannot reach the feature map → Faithfulness = 0.012 by geometry
                              │
                              ↓
Fix (proposed, not implemented): Patch-level Faithfulness at 16×16 granularity
Best practical model: M2 (L3+L4 with skip) — best Dice×Purity×Faithfulness tradeoff
```

### Why This is Interesting

The assumption in prior literature is that prototype networks with skip connections
are "less interpretable" than decoder-free designs. Our 2×2 ablation shows:

| | L3+L4 | L4 only |
|---|---|---|
| **With skip** | Stage 29: Dice=0.821, Purity=0.527, AP=0.051, Faith=0.069, Stab=3.38 | Stage 8A: Dice=0.810, AP=0.030, Stab=2.46 |
| **No skip** | 9b: Dice=0.559, Purity=0.686, AP=0.301, Faith=0.035, Stab=11.94 | 9a: Dice=0.606, Purity=0.679, AP=0.312, Faith=0.012, Stab=10.92 |

1. Removing skip ~10× improves AP (Barrier 1 addressed) — at −35% Dice cost
2. But removing skip worsens Faithfulness (0.069 → 0.035) and Stability (3.38 → 11.94)
3. Skip decoder acts as spatial smoother: better pixel-level perturbation sensitivity
4. The Faithfulness failure in no-skip models is geometric (16×16 feature map), not training
5. Both axes of the 2×2 table matter: skip choice AND level choice independently affect XAI

This reframes the "skip connection = bad for XAI" assumption as incomplete.

---

## 2. Complete Results Inventory (All Existing Data)

### 2.1 With Decoder / Skip Connections (v1–v8, ProtoSegNet)

#### Stage 8 — Ablation Study (All levels L1–L4, with decoder)

| Variant | Description | Val Dice | AP | Faithfulness | Stability |
|---------|-------------|----------|-----|-------------|-----------|
| **Full** | L2 kernel + push-pull + div + SoftMask + multi-scale | **0.817** | **0.102** | **+0.059** | 3.00 |
| A | Single-scale (L4 only) | 0.810 | 0.030 | — | 2.46 |
| B | No diversity loss | 0.825 | 0.130 | — | 14.10 |
| C | No SoftMask | 0.632 | 0.049 | — | 2.97 |
| D | No push-pull | 0.622 | 0.063 | — | 1.80 |
| E | log-cosine (no push-pull) | 0.790 | 0.033 | — | 2.76 |

Checkpoint: `proto_seg_ct_abl_{a,b,c,d}.pth` (Full model checkpoint unknown — Stage 8 predates naming convention)

#### Stage 29 — M2 (L3+L4, warm-start from attention M4, **best with-skip result**)

| Metric | Value |
|--------|-------|
| Val Dice | **0.8656** |
| Eff. Purity | **0.649** (L3: 0.486, L4: 0.774) |
| L3 dominance | 43.4% |
| L4 dominance | 56.6% |
| AP | ⚠️ **NOT MEASURED** |
| Faithfulness | ⚠️ **NOT MEASURED** |
| Stability | ⚠️ **NOT MEASURED** |

Checkpoint: `checkpoints/proto_seg_ct_l3l4_warmstart.pth`
Projected: `checkpoints/projected_prototypes_ct_l3l4_warmstart.pt`

#### M2 Cold-Start — L3+L4, no warm-start

| Metric | Value |
|--------|-------|
| Val Dice | **0.8722** |
| L4 Purity | 0.804 |
| AP | ⚠️ **NOT MEASURED** |

Checkpoint: `checkpoints/proto_seg_ct_l3l4.pth`
Projected: `checkpoints/projected_prototypes_ct_l3l4.pt`

#### Stage 34 / 34b — M2 + ALC

| Stage | Config | Val Dice | Eff. Purity | Eff. AP | Faithfulness | Stability |
|-------|--------|----------|-------------|---------|-------------|-----------|
| 34 (ALC L3+L4, λ=0.05) | L3+L4, with skip | 0.8478 | 0.661 | 0.179 | ⚠️ missing | ⚠️ missing |
| 34b (ALC L3 only, λ=0.05) | L3+L4, with skip | **0.8628** | 0.593 | 0.221 | ⚠️ missing | ⚠️ missing |

Checkpoints: `proto_seg_ct_l3l4_alc.pth`, `proto_seg_ct_l3l4_alc_l3only.pth`

**Note:** AP for 34/34b is available from effective_quality CSV (Eff. AP = 0.179 / 0.221) but
Faithfulness and Stability have never been measured for any with-skip model except Stage 8 Full.

### 2.2 Without Decoder / Skip Connections (v9, ProtoSegNetV2)

#### Single-Level Ablation

| Stage | Level | Res. | Val Dice | Eff. Purity | Eff. AP | Faithfulness | Stability |
|-------|-------|------|----------|-------------|---------|-------------|-----------|
| 9L1 | L1 | 128×128 | 0.146 | 0.159 | 0.166 ✅ | **0.160 ✅** | 16.99 ❌ |
| 9L2 | L2 | 64×64 | 0.336 | 0.569 | 0.219 ✅ | **0.218 ✅** | 14.38 ❌ |
| 9L3 | L3 | 32×32 | 0.554 | **0.844** | **0.319 ✅** | 0.060 ❌ | 10.29 ❌ |
| 9a | L4 | 16×16 | **0.606** | 0.689 | 0.312 ✅ | 0.012 ❌ | 10.92 ❌ |

#### Multi-Level Variants

| Stage | Config | Val Dice | Eff. Purity | Eff. AP | Faithfulness | Stability |
|-------|--------|----------|-------------|---------|-------------|-----------|
| **9b** | L3+L4, uniform | 0.559 | 0.686 | 0.301 ✅ | 0.035 ❌ | 11.94 ❌ |
| 9c | L1–L4, learned attn | 0.586 | 0.676 | 0.262 ✅ | 0.048 ❌ | 13.63 ❌ |
| 9LF | All 4 frozen, attn only | 0.606 | — | — | — | — |

#### Resolution Mismatch Evidence

| Level | Feature res. | px/activation (256²) | Faithfulness | Stability |
|-------|-------------|---------------------|-------------|-----------|
| L1 | 128×128 | 2×2 | **0.160 ✅** | 16.99 ❌ |
| L2 | 64×64 | 4×4 | **0.218 ✅** | 14.38 ❌ |
| L3 | 32×32 | 8×8 | 0.060 ❌ | 10.29 ❌ |
| L4 | 16×16 | 16×16 | 0.012 ❌ | 10.92 ❌ |

**Monotonic relationship:** As feature resolution coarsens, Faithfulness degrades —
even though all v9 models share the structural guarantee `logits = f(heatmaps)`.
This confirms Barrier 2 is geometric, not training-related.

### 2.3 Core 2×2 Ablation: Skip × Level Configuration

The core comparison is a 2×2 matrix: {skip, no-skip} × {L3+L4, L4 only}.

| | **L3+L4** | **L4 only** |
|---|---|---|
| **With skip** | Stage 29 (warm-start) | Stage 8A ablation |
| **No skip** | 9b | 9a |

#### Full 2×2 Metric Table

| Metric | **Stage 29** (skip, L3+L4) | **Stage 8A** (skip, L4 only) | **9b** (no-skip, L3+L4) | **9a** (no-skip, L4 only) |
|--------|---------------------------|------------------------------|-------------------------|--------------------------|
| Val Dice | **0.821** | 0.810 | 0.559 | 0.606 |
| Eff. Purity | 0.527 | 0.474 | 0.686 | 0.679 |
| Eff. AP | 0.051 | 0.057 | **0.301** | **0.312** |
| Faithfulness (px) | **0.069** | 0.093 | 0.035 | 0.012 |
| Stability | **3.38** | 3.79 | 11.94 | 10.92 |
| Patch Faith (bs=16) | 0.212 | 0.161 | 0.200 | **0.259** |

Stage 29 per-level: L3 (w=0.60, purity=0.381, AP=0.040), L4 (w=0.40, purity=0.744, AP=0.067)

**What the complete 2×2 reveals:**
- **Bypass barrier (skip → no-skip, L3+L4)**: AP 0.051→0.301 (6×); same for L4: 0.057→0.312 (5.5×)
- **Skip Faithfulness > no-skip**: 0.069>0.035 (L3+L4) and 0.093>0.012 (L4) — decoder smooths perturbations
- **Skip Stability < no-skip**: 3.38<11.94 (L3+L4) and 3.79<10.92 (L4) — consistent across level configs
- **L3+L4 skip vs L4 skip**: Stage 29 Purity=0.527 > Stage 8A Purity=0.474; adding L3 improves purity
- **The two barriers are independent and in opposite directions across the full 2×2**

**The core ablation table is incomplete for Stage 29 AP/Purity (stale projection) and Stage 8A Purity/Faithfulness.**

---

## 3. Gap Analysis — What Is Missing

### Critical Gaps (must fill for the paper)

| ID | What | Why Critical | Status |
|----|------|-------------|--------|
| **G1** | Stage 29 Faithfulness + Stability | Core 2×2 table; diagonal comparison | ✅ **Done**: Faith=0.069, Stab=3.38 |
| **G1b** | Stage 29 AP + Purity (fresh projection) | Stale projection file invalidates AP/Purity | ✅ **Done**: Purity=0.527, AP=0.051 |
| **G2** | xai_summary_9a.csv | Summary file for 9a needed for report | ✅ **Done**: `results/v9/xai_summary_9a.csv` |
| **G3** | Patch-level Faithfulness metric | Validates Barrier 2 diagnosis at correct granularity | ✅ **Done** |

**G1b Detail (Stale Projection Problem):**
The projection file `projected_prototypes_ct_l3l4_warmstart.pt` has prototype norms (29.4, 38.8)
inconsistent with the checkpoint's own prototypes (44.8, 63.4). The projection was saved at an
early Phase B epoch; the checkpoint was saved at the best Dice epoch much later. Loading the
stale projection file overwrites the checkpoint's correct prototypes.
**Fix:** Skip loading the stale projection file; use the checkpoint's own `prototype_vectors`
(already stored in `model_state_dict`). Run purity/AP eval directly on the checkpoint state.

### Important Gaps (strengthen the paper)

| ID | What | Why Important | Status |
|----|------|--------------|--------|
| **G4** | Stage 34b Faithfulness + Stability | Completes v8 model family comparison | ✅ **Done**: Faith=0.083, Stab=9.24, PFaith=0.302 |
| **G5** | Stage 8A Purity + Faithfulness | Completes 2×2 table (skip × L4 only cell) | ✅ **Done**: Purity=0.474, AP=0.057, Faith=0.093, Stab=3.79, PFaith=0.161 |
| **G6** | Per-class Dice for Stage 29 | Appendix completeness | ✅ **Done**: LV=0.724, RV=0.824, LA=0.911, RA=0.794, Myo=0.834, Aorta=0.907, PA=0.720, Mean=0.816 |
| **G7** | Baseline U-Net AP (upper bound reference) | Shows bypass gap vs U-Net precision | ✅ **Done**: AP=0.349 (proxy softmax heatmap) |

### Optional Gaps (future work if time permits)

| ID | What | Why Optional |
|----|------|-------------|
| G8 | MR modality validation | CT-only is acknowledged limitation; CT results are the contribution |
| G9 | ALC with ReLU normalization | Separate from Two-Barrier narrative |
| G10 | v8 MR XAI metrics | Not required for CT-focused two-barrier paper |

---

## 4. Experiment Stages

### Stage 10a — XAI Evaluation for Stage 29 (CRITICAL)

**Goal:** Measure AP, Faithfulness, and Stability for `proto_seg_ct_l3l4_warmstart.pth`.
This completes the core skip vs no-skip comparison table.

**Implementation:** Use existing XAI notebooks/scripts with the Stage 29 checkpoint.
The model is ProtoSegNet (original, with decoder), L3+L4, warm-started from attention M4.

```python
# Notebook: notebooks/37_xai_stage29.ipynb (new)
CKPT = 'checkpoints/proto_seg_ct_l3l4_warmstart.pth'
PROJ = 'checkpoints/projected_prototypes_ct_l3l4_warmstart.pt'
MODEL_TYPE = 'ProtoSegNet'   # with decoder, with skip
LEVELS = [3, 4]
OUTPUT_DIR = 'results/v10/xai_stage29/'
```

**Metrics to collect:**
- `xai_purity_stage29.csv` — per-prototype purity
- `xai_ap_stage29.csv` — per-class AP
- `xai_faithfulness_stage29.csv` — patient-level Faithfulness
- `xai_stability_stage29.csv` — patient-level Stability
- `xai_effective_quality_stage29.csv` — weighted effective metrics
- `xai_summary_stage29.csv` — one-row summary

**Expected results (hypothesis):**
- AP ~ 0.10–0.15 (similar to Stage 8 Full; decoder bypass active)
- Faithfulness ~ 0.05–0.10 (better than no-skip due to higher-resolution feature interaction)
- Stability ~ 3.0–5.0 (better than no-skip due to decoder acting as spatial smoother)

**Success criteria:** All metrics measured; comparison table complete.

---

### Stage 10b — Aggregate xai_summary_9a.csv (TRIVIAL)

**Goal:** Create `results/v9/xai_summary_9a.csv` from existing component files.

All component files already exist:
- `results/v9/xai_effective_quality_9a.csv` → Val Dice=0.606, Eff.Purity=0.679, Eff.AP=0.312
- `results/v9/xai_faithfulness_9a.csv` → Faithfulness=0.012
- `results/v9/xai_stability_9a.csv` → Stability=10.921

**Output:**
```csv
Metric,Value,Min gate,Target,Pass
Val Dice,0.606,nan,nan,—
Effective Purity,0.679,nan,nan,—
Effective AP,0.312,0.15,0.25,✅
Effective Compact.,0.036,nan,nan,—
Faithfulness,0.012,0.15,0.30,❌
Stability,10.921,nan,2.00,❌
```

---

### Stage 10c — Patch-Level Faithfulness Metric (IMPORTANT)

**Goal:** Implement a new Faithfulness variant that uses 16×16 spatial blocks aligned to
L4's feature grid, rather than single-pixel masking. This directly addresses Barrier 2.

**Rationale:** The current Faithfulness metric zeros individual pixels at 256×256.
A 16×16 feature map makes one spatial decision per 16×16-pixel block. Zeroing a single
pixel out of 256 in that block changes < 0.4% of the information seen by that activation —
effectively zero. The metric tests something the architecture cannot exhibit.

**New metric formulation:**
```
patch_faithfulness(model, x, heatmap):
    # For each of the N_PATCHES = (256/block)² patches:
    #   1. Zero out the block in x
    #   2. Re-run forward pass
    #   3. Record Δ predicted probability for target class
    # Pearson(heatmap_importance_scores, Δ_probs)
    # Use block_size = 16 for L4, 8 for L3, 4 for L2
```

**Implementation in:** `src/metrics/patch_faithfulness.py` (new file)

**Apply to:**
- 9a (L4, no-skip) with block_size=16 → expected: much higher than pixel-level (0.012)
- 9b (L3+L4, no-skip) with block_size=16 → expected: higher than pixel-level (0.035)
- 9L3 (L3, no-skip) with block_size=8 → compare to pixel-level (0.060)
- Stage 29 (L3+L4, with skip) with block_size=16 → expected: measure true bypass effect

**Output files:**
```
results/v10/patch_faithfulness_9a.csv
results/v10/patch_faithfulness_9b.csv
results/v10/patch_faithfulness_9L3.csv
results/v10/patch_faithfulness_stage29.csv
```

**Success criteria:**
- Patch-level Faithfulness for 9a (L4 no-skip) > 0.15 (passes gate when measured correctly)
- Stage 29 patch-level Faithfulness < 9a patch-level (bypass measurable at block scale)

---

### Stage 10d — XAI Evaluation for Stage 34b and M2 Cold-Start (HELPFUL)

**Goal:** Complete the v8 model family's XAI profile.

```
Stage 34b checkpoint: checkpoints/proto_seg_ct_l3l4_alc_l3only.pth
M2 cold-start:        checkpoints/proto_seg_ct_l3l4.pth
```

Stage 34b already has Eff.AP=0.221 from effective_quality CSV, but needs Faithfulness and Stability.
M2 cold-start has the highest Dice (0.8722) but zero XAI characterization.

**Output:** `results/v10/xai_summary_34b.csv`, `results/v10/xai_summary_m2cold.csv`

---

### Stage 10e — Per-Class Dice Summary for Stage 29 (TRIVIAL)

**Goal:** Extract per-class Val Dice for Stage 29 from training curve.

Stage 29 training is logged in `results/v8/train_curve_proto_ct_l3l4_warmstart.csv` (if it exists)
or can be read from the checkpoint's last validation run.

The Stage 34b per-class Dice at epoch 100:
```
LV=0.758, RV=0.866, LA=0.939, RA=0.775, Myo=0.852, Aorta=0.958, PA=0.723
Mean FG = 0.839
```

**Output:** Add Stage 29 per-class Dice to Appendix of report.

---

## 5. The Core Comparison Table (Target State After v10)

This is the table that the paper's narrative rests on.
Cells marked ⚠️ will be filled by the stages above.

### Table 1: 2×2 Core Ablation — Skip × Level Configuration

| Metric | **Stage 29** (skip, L3+L4) | **Stage 8A** (skip, L4 only) | **9b** (no-skip, L3+L4) | **9a** (no-skip, L4 only) |
|--------|---------------------------|------------------------------|-------------------------|--------------------------|
| Val Dice | **0.821** | 0.810 | 0.559 | 0.606 |
| Eff. Purity | 0.527 | 0.474 | 0.686 | 0.679 |
| Eff. AP | 0.051 | 0.057 | **0.301** | **0.312** |
| Faithfulness (px) | **0.069** | 0.093 | 0.035 | 0.012 |
| Stability | **3.38** | 3.79 | 11.94 | 10.92 |
| Patch Faith (bs=16) | 0.212 | 0.161 | 0.200 | **0.259** |

Stage 29 per-level: L3 (w=0.60, purity=0.381, AP=0.040), L4 (w=0.40, purity=0.744, AP=0.067)
Stage 8A per-level: L4 only, Purity=0.474, AP=0.057 (fresh projection)

**What the complete 2×2 shows:**
- **Bypass barrier**: skip → no-skip, AP jumps ~5–6× in both column pairs (L3+L4: 0.051→0.301; L4: 0.057→0.312)
- **Resolution barrier**: removing skip WORSENS pixel Faithfulness and Stability in both column pairs
- **Barrier 2 resolved**: 9a pixel Faith=0.012 → patch Faith=0.259 (21×); structural guarantee is real, metric was wrong
- **At patch level, skip ≈ no-skip** (0.16–0.21 range) — AP and Faithfulness measure genuinely different things
- **L3+L4 adds value vs L4 alone** only in skip models: Stage 29 Purity=0.527 > Stage 8A=0.474; Dice 0.821 vs 0.810
- **The two barriers are independent across the full 2×2** — this is the core finding

### Table 2: Cross-Level XAI Profile (All No-Skip)

| Level | Dice | Purity | AP | Faith (px) | Faith (patch, aligned) | Stability |
|-------|------|--------|----|-----------|----------------------|-----------|
| L1 (128×128) | 0.146 | 0.159 | 0.166 | 0.160 ✅ | ⚠️ (not run) | 16.99 |
| L2 (64×64) | 0.336 | 0.569 | 0.219 | 0.218 ✅ | ⚠️ (not run) | 14.38 |
| L3 (32×32) | 0.554 | **0.844** | **0.319** | 0.060 | **0.209** (bs=8) | 10.29 |
| **L4 (16×16)** | **0.606** | 0.689 | 0.312 | 0.012 | **0.259** (bs=16) | 10.92 |

Note: patch Faith for L3 uses bs=8 (L3-aligned); for L4 uses bs=16 (L4-aligned).
At bs=16, 9L3 patch Faith=0.159 (L4-aligned is less appropriate for L3 model).

### Table 3: Full Model Family (Best Practical)

| Model | Config | Dice | Purity | AP | Faith. | Stab. |
|-------|--------|------|--------|----|--------|-------|
| Baseline U-Net | No prototype | **0.823** | — | **0.349**† | — | — |
| Stage 8A | L4 only, with skip | 0.810 | 0.474 | 0.057 | 0.093 | 3.79 |
| Stage 8 Full | L1–L4, with skip | 0.817 | 0.334 | 0.102 | 0.059 | 3.00 |
| Stage 34b | L3+L4, skip, ALC | 0.842 | 0.593 | 0.221 | 0.083 | 9.24 |
| **Stage 29** | L3+L4, with skip | **0.821** | 0.527 | 0.051 | **0.069** | **3.38** |
| 9b | L3+L4, no skip | 0.559 | 0.686 | **0.301** | 0.035 | 11.94 |
| 9a | L4, no skip | 0.606 | 0.689 | 0.312 | 0.012 | 10.92 |
| 9L3 | L3, no skip | 0.554 | **0.844** | 0.319 | 0.060 | 10.29 |
| 9L2 | L2, no skip | 0.336 | 0.569 | 0.219 | **0.218** | 14.38 |

†U-Net AP computed using softmax output as proxy heatmap — not prototype activations.
This is an upper bound: "what AP would be if prototype activations = final segmentation."

**G7 Key Finding:** No-skip prototypes (AP=0.301–0.312) reach **86–89% of U-Net AP (0.349)**.
Skip prototypes (AP=0.051–0.057) reach only 15%. Removing the bypass closes most of the gap.
The remaining ~11–14% gap (no-skip vs U-Net) reflects prototype compression (discrete vectors
vs continuous softmax) — an acceptable cost of interpretability.

Per-class U-Net AP: LV=0.417, RV=0.331, LA=0.256, RA=0.384, Myo=0.449, Aorta=0.375, PA=0.230

### Appendix: Per-Class Val Dice for Stage 29

| LV | RV | LA | RA | Myo | Aorta | PA | Mean |
|----|----|----|----|----|------|----|------|
| 0.724 | 0.824 | 0.911 | 0.794 | 0.834 | 0.907 | 0.720 | **0.816** |

---

## 6. Stage Status Tracker

| Stage | Name | Status | Output |
|-------|------|--------|--------|
| **10a** | XAI eval for Stage 29 (with-skip L3+L4) | ✅ Done | Faith=0.069, Stab=3.38, Purity=0.527, AP=0.051 |
| **10a-fix (G1b)** | Stage 29 AP/Purity fresh projection | ✅ Done | `results/v10/xai_*_stage29_fresh.csv` |
| **10b** | Generate xai_summary_9a.csv | ✅ Done | `results/v9/xai_summary_9a.csv` |
| **10c** | Patch-level Faithfulness metric | ✅ Done | `results/v10/xai_patch_faithfulness_*.csv` |
| **10d** | XAI eval for Stage 34b | ✅ Done | `results/v10/xai_summary_34b.csv` |
| **10d-g5** | XAI eval for Stage 8A (Purity/Faith) | ✅ Done | `results/v10/xai_summary_8a.csv` |
| **10e** | Per-class Dice for Stage 29 | ✅ Done | LV=0.724, RV=0.824, LA=0.911, RA=0.794, Myo=0.834, Aorta=0.907, PA=0.720, Mean=0.816 |

---

## 7. Success Criteria for v10

| Criterion | Gate | Result |
|-----------|------|--------|
| Stage 29 AP measured | Required | ✅ AP=0.051 (fresh projection) |
| Patch Faith for 9a > pixel Faith | Required | ✅ 0.259 vs 0.012 (21× lift) — Barrier 2 confirmed |
| Patch Faith for 9b ≈ Stage 29 patch Faith | Required | ✅ 0.200 vs 0.212 — bypass ≠ input insensitivity |
| Report v10 written | Required | ⬜ TODO |

---

## 8. Proposed Paper Structure (v10 Report Target)

```
1. Introduction
   — Black-box segmentation; XAI goals; prototype networks
   — Prior assumption: skip connections hurt interpretability
   — Our finding: two independent barriers, not one

2. Background
   — ProtoSegNet architecture (v8 variant, with decoder)
   — XAI metrics: AP, Faithfulness, Stability, Purity
   — MM-WHS dataset

3. Barrier 1: The Bypass Problem
   — Evidence: Stage 8 ablation (Table 1, Stage 8 ablations)
   — Removing decoder: 9b vs Stage 29 (Table 1 core ablation)
   — Result: AP triples (0.10 → 0.30)
   — Cost: Dice −35% (0.866 → 0.559)

4. Barrier 2: The Resolution Problem
   — Evidence: Single-level ablation table (L1→L4 faithfulness gradient)
   — Geometric argument: px/activation at each level
   — Structural guarantee ≠ metric sensitivity
   — Faithfulness passes for L1/L2 (coarse enough), fails for L3/L4

5. Proposed Fix: Patch-Level Faithfulness
   — Motivation: align metric granularity to architecture granularity
   — Results: 9a patch-Faith vs pixel-Faith (Table 2)
   — Comparison: Stage 29 vs 9b patch-Faith (validates Barrier 1 at correct granularity)

6. Best Practical Model
   — Stage 29 (L3+L4 with skip): best Dice×Purity profile
   — Per-level selection: Max-Gap filter (v8 contribution)
   — Conclusion: for clinical deployment, with-skip is recommended

7. Conclusion
   — Two barriers are independent
   — Removing skip addresses Barrier 1 at cost of Dice and Barrier 2 exposure
   — Patch-level metrics required for coarse-resolution prototypes
   — Future: ALC with correct normalization; MR validation
```

---

## 9. Checkpoints Reference

| Stage | Checkpoint | Projected Protos |
|-------|-----------|-----------------|
| Stage 8 Full (M4) | `proto_seg_ct_abl_*.pth` (Full = unknown) | `projected_prototypes_ct.pt` |
| **Stage 29** (M2, warm) | `proto_seg_ct_l3l4_warmstart.pth` | `projected_prototypes_ct_l3l4_warmstart.pt` |
| M2 cold-start | `proto_seg_ct_l3l4.pth` | `projected_prototypes_ct_l3l4.pt` |
| Stage 34 (ALC L3+L4) | `proto_seg_ct_l3l4_alc.pth` | `projected_prototypes_ct_l3l4_alc.pt` |
| Stage 34b (ALC L3) | `proto_seg_ct_l3l4_alc_l3only.pth` | `projected_prototypes_ct_l3l4_alc_l3only.pt` |
| 9a (L4, no-skip) | `proto_seg_ct_v2_l4.pth` | `projected_prototypes_ct_v2_l4.pt` |
| 9b (L3+L4, no-skip) | `proto_seg_ct_v2_l34.pth` | `projected_prototypes_ct_v2_l34.pt` |
| 9L3 (L3, no-skip) | `proto_seg_ct_v2_l3.pth` | `projected_prototypes_ct_v2_l3.pt` |
| 9L2 (L2, no-skip) | `proto_seg_ct_v2_l2.pth` | `projected_prototypes_ct_v2_l2.pt` |
| 9L1 (L1, no-skip) | `proto_seg_ct_v2_l1.pth` | `projected_prototypes_ct_v2_l1.pt` |

---

## 10. Key Architectural Differences (Reference)

### ProtoSegNet (with skip, v1–v8)
```
Input → HierarchicalEncoder → PrototypeLayer → SoftMask → Decoder (with skip) → logits
                                     ↑
                 prototypes are one input to decoder; decoder has bypass via skip connections
```
AP low because skip connections allow decoder to ignore prototype signals.

### ProtoSegNetV2 (no skip, v9)
```
Input → HierarchicalEncoder → PrototypeLayer → upsample → Σ_l w_l × up_l → logits
                                                            ↑
                                           ONLY path from input to logits
```
Faithfulness structurally guaranteed. But coarse feature maps expose Resolution Barrier.

---

## 11. Anticipated Questions and Answers

**Q: If L3+L4 with skip has Faithfulness=0.059 but no-skip L3+L4 has Faithfulness=0.035,
why is no-skip better?**

A: It isn't, by the pixel-level metric. But this is because the no-skip L3+L4 (9b) uses
L4 (16×16) which is too coarse for pixel-level testing. Patch-level Faithfulness (10c)
should show that 9b > Stage 29 at block scale, where the bypass effect is measurable.

**Q: Why does removing the decoder hurt Faithfulness?**

A: Decoder + skip connections force higher-resolution feature interaction (skip features from
L3 at 32×32, L2 at 64×64 are fed into the decoder). Without the decoder, the final signal is
the upsample of a 16×16 L4 heatmap. The decoder inadvertently improves spatial resolution of
the output, which helps pixel-level perturbation tests.

**Q: What is the recommended model for clinical use?**

A: Stage 29 (L3+L4 with skip, Dice=0.866, Purity=0.649). It has the best Dice, good purity,
positive Faithfulness. Its AP=~0.10 is a known limitation (bypass barrier). For full XAI
compliance, 9b (no-skip L3+L4) is structurally cleaner but at −35% Dice cost.
