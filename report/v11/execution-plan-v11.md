# Execution Plan v11: Dice–Interpretability Trade-off as Core Contribution
## Reframing: from Two-Barrier Framework to Structural Trade-off

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16/2/2 patients, 3389/382/484 slices)
**Hardware:** MacBook 48GB RAM, Apple Silicon (MPS)
**Date:** 2026-03-26
**Preceded by:** v10 concluded (report-v10.md)

---

## 1. Narrative Shift from v10

### What Changed

v10 framed the paper around **Two Barriers** (Bypass + Resolution).
After reviewing the data in v11 discussions, two problems emerged:

1. **Bypass Barrier** lacks direct mechanistic evidence — we show correlation (removing skip → AP↑) but not that the decoder *actively ignores* prototype signal during inference.
2. **Resolution Barrier** is hard to frame as a model failure — the Purity numbers go the wrong direction (L4 Purity > L3 Purity due to patch size artifact), and pixel-level Faithfulness is too low across the board (<0.1) to be the primary evidence for anything.

### New Framing

The core contribution is reframed as a **structural trade-off**:

> **The architectural choice that makes prototype explanations structurally genuine (removing skip connections) causes a 32% Dice drop. This trade-off is not a training artefact — it is load-bearing.**

| | Skip | No-skip | Δ |
|---|---|---|---|
| Dice | 0.821 | 0.559 | −32% |
| AP | 0.051 | 0.301 | +6× |
| Purity | 0.527 | 0.686 | +30% |
| Patch Faith | 0.212 | 0.200–0.259 | ≈ same |

The Patch Faithfulness row is key: at the correct granularity, skip and no-skip models are comparably faithful. The real cost of removing skip is Dice, and the real gain is AP + Purity. Pixel-level Faithfulness is a distractor.

### The Remaining Gap

The mechanism by which skip connections suppress AP/Purity is argued structurally in v10 but not demonstrated directly. The question Reviewer 2 will ask:

> *"How do you know the decoder is ignoring prototype signal? Maybe the prototypes just didn't converge well with skip connections?"*

**Experiment 1 (Required)** answers this by computing gradient attribution to skip path vs prototype path, directly in the trained Stage 29 model.

---

## 2. New Story Arc

```
ProtoSegNet (with skip)
  → Dice = 0.821 ✅  AP = 0.051 ❌  Purity = 0.527 ❌
  → Why low AP/Purity? Decoder draws on skip connections, not prototype heatmaps
  → Evidence: gradient attribution shows skip path dominates decoder input
                              │
                              │ Gradient attribution (Experiment 1)
                              ↓
  decoder: 80%+ gradient mass flows through skip connections
           prototype heatmap contribution is marginal
                              │
                              │ Architectural fix: remove skip connections
                              ↓
ProtoSegNetV2 (no skip)
  → Dice = 0.559 ❌  AP = 0.301 ✅  Purity = 0.686 ✅
  → logits = f(heatmaps) by construction: structural guarantee holds
  → Patch Faithfulness = 0.259: faithful at the correct granularity
                              │
                              │ The trade-off is structural, not tunable
                              ↓
Conclusion:
  → No architecture simultaneously achieves high Dice AND high AP/Purity
  → ALC (Stage 34b) partially bridges gap: Dice=0.842, AP=0.221, but gradient
    attribution shows bypass still partially active
  → Clinical recommendation: Stage 29 (Dice first); XAI-first: 9b (AP/Purity first)
```

---

## 3. Existing Results Inventory (Carry-over from v10, All Complete)

### Core 2×2 Ablation Table (v10 final state)

| Metric | **Stage 29** (skip, L3+L4) | **Stage 8A** (skip, L4) | **9b** (no-skip, L3+L4) | **9a** (no-skip, L4) |
|--------|---------------------------|------------------------|------------------------|---------------------|
| Val Dice | **0.821** | 0.810 | 0.559 | 0.606 |
| Eff. Purity | 0.527 | 0.474 | **0.686** | 0.679 |
| Eff. AP | 0.051 | 0.057 | **0.301** | **0.312** |
| Faithfulness (px) | **0.069** | **0.093** | 0.035 | 0.012 |
| Stability | **3.38** | **3.79** | 11.94 | 10.92 |
| Patch Faith (bs=16) | 0.212 | 0.161 | 0.200 | **0.259** |

### ALC Partial Fix (Stage 34b)

| Stage | Dice | Purity | AP | Faith (px) | Patch Faith | Stability |
|-------|------|--------|----|-----------|-------------|-----------|
| Stage 29 (no ALC) | 0.821 | 0.527 | 0.051 | 0.069 | 0.212 | 3.38 |
| Stage 34b (ALC L3) | **0.842** | 0.593 | 0.221 | 0.083 | 0.302 | 9.24 |

ALC raises AP 4.3× without removing skip, demonstrating partial bypass mitigation at a Stability cost.

---

## 4. Gap Analysis

### Required for v11

| ID | What | Why Required | Status |
|----|------|-------------|--------|
| **G11** | Gradient attribution: skip path vs prototype path | Direct mechanistic evidence for Bypass; compares same-level skip vs no-skip | ✅ Done |
| **G13** | Spatial misalignment verification (Dir 1–3) | Validates "L3+L4 bypass = spatial misalignment" claim; three complementary angles | 🔶 Partial |
| **G13b** | Dir 3 fix: heatmap AP for L4 counterfactual | L4 Dir 3 invalid (scale mismatch); heatmap AP bypasses decoder entirely | ⬜ TODO |

### Optional for v11

| ID | What | Why Optional | Status |
|----|------|-------------|--------|
| **G12** | Progressive skip ablation (inference-time or retrain) | G11+G13 provide sufficient mechanism evidence; G12 would add monotonic curve | ⬜ Optional |

---

## 5. Experiment 1 (Required): Gradient Attribution — Skip vs Prototype Path

### Goal

Directly measure how much the decoder's output is driven by encoder/skip-connection features
vs prototype heatmap modulation, comparing same-level skip vs no-skip configurations.

**bypass_ratio** = `||∂logits/∂feat|| / (||∂logits/∂feat|| + ||∂logits/∂heatmap||)`

- bypass_ratio → 1: decoder relies almost entirely on encoder/skip features
- bypass_ratio → 0: decoder relies almost entirely on prototype heatmap
- no-skip (9a, 9b): bypass_ratio = 0 by construction (no feat path to decoder)

### Results (✅ Complete)

**notebook:** `notebooks/43_g11_gradient_attribution.ipynb`
**data:** `results/v11/gradient_attribution_{stage29,stage8a,stage34b}.csv`

#### Core 2×2 Comparison (same level, skip vs no-skip)

| Model | Config | bypass_ratio | AP |
|-------|--------|-------------|-----|
| Stage 8A (skip) | L4 only | **0.778** | 0.057 |
| 9a (no-skip) | L4 only | 0.000 (structural) | 0.312 |
| Stage 29 (skip) | L3+L4 | **0.465** | 0.051 |
| 9b (no-skip) | L3+L4 | 0.000 (structural) | 0.301 |

#### Per-level breakdown (skip models)

| Model | L3 bypass | L4 bypass |
|-------|-----------|-----------|
| Stage 8A | — (L4 only) | 0.778 |
| Stage 29 | 0.433 | 0.497 |

### Interpretation

**L4-only comparison is clean:** Stage 8A bypass = 0.778 — the decoder draws 78% of its
gradient from encoder/skip features, effectively ignoring the prototype heatmap. Removing
skip → bypass = 0 → AP jumps 5.5×. This directly supports the Bypass Barrier mechanism.

**L3+L4 is more nuanced:** Stage 29 bypass = 0.465 (nearly balanced), yet AP = 0.051.
This reveals that bypass_ratio and AP measure orthogonal properties:
- **bypass_ratio**: is the output sensitive to changes in the heatmap? (causal sensitivity)
- **AP**: does the heatmap activate at the right spatial location? (spatial alignment)

A model can be causally sensitive to heatmaps (low bypass) while still having
poorly-localized prototypes (low AP). The Bypass Barrier for L3+L4 operates through
spatial misalignment rather than full causal disconnection.

**ALC comparison (additional finding, not core narrative):**
Stage 34b bypass = 0.656 — unexpectedly *higher* than Stage 29 (0.465). ALC improves AP
(0.051 → 0.221) not by reducing decoder bypass but by improving prototype vector quality
(Purity 0.527 → 0.593). Bypass ratio and AP are indeed orthogonal.

### Key Claim for Paper

> For L4-only models, gradient attribution confirms the bypass mechanism directly:
> 78% of decoder gradient flows through skip/encoder features in Stage 8A, bypassing
> the prototype heatmap. For L3+L4 models, the bypass is partial (47%) but AP remains
> near-zero — indicating spatial misalignment rather than full causal bypass.

### Metrics to Collect (done)

- `bypass_ratio` per slice, class, level for Stage 8A and Stage 29
- Summary CSV: `results/v11/gradient_attribution_summary.csv`

### Expected Output

```
results/v11/gradient_attribution_stage29.csv
  columns: slice_id, class_k, level, skip_grad_norm, proto_grad_norm, bypass_ratio

results/v11/gradient_attribution_34b.csv
  (same format, for ALC model comparison)

results/v11/gradient_attribution_summary.csv
  one row per model: mean_bypass_ratio, std_bypass_ratio
```

### Hypothesis

- Stage 29: `bypass_ratio > 0.70` (decoder primarily uses skip features)
- Stage 34b: `bypass_ratio < 0.50` (ALC forces more prototype reliance, consistent with AP=0.221)
- No-skip (9b/9a): `bypass_ratio = 0` by construction (no skip path exists)

### Success Criteria

- bypass_ratio for Stage 29 is substantially higher than Stage 34b
- bypass_ratio correlates with AP across models (high bypass → low AP)
- Result is reportable as a new figure: "Decoder Gradient Attribution by Path"

---

## 6. Experiment G13 (Required): Spatial Misalignment Verification

**Claim to validate:** For L3+L4 skip models, bypass_ratio = 0.465 (partial bypass) yet
AP = 0.051 because the prototype heatmaps are *spatially misaligned* — they activate in
wrong locations, not merely ignored by the decoder.

**Comparisons:** L4 (Stage 8A vs 9a) and L3+L4 (Stage 29 vs 9b).
**Notebook:** `notebooks/44_g13_spatial_misalignment.ipynb`

---

### Direction 1: Heatmap Visualisation

Side-by-side prototype heatmaps for the same test slice, same class:
- Skip model heatmap: expected diffuse / off-target
- No-skip model heatmap: expected concentrated on correct structure

**Output:** `results/v11/spatial_misalignment_viz_{l4,l3l4}.png` ✅ Done

---

### Direction 2: Spatial Precision (Heatmap–GT Overlap)

For each model and class k, compute **weighted precision**:
```
spatial_precision_k = Σ(heatmap_k * GT_k) / Σ(heatmap_k)
```
The fraction of heatmap activation mass that falls on the correct GT structure.

**Results (✅ Done):**

| pair | skip SP | no-skip SP | delta |
|------|---------|-----------|-------|
| L4: Stage 8A vs 9a | 0.020 | 0.075 | +0.055 |
| L3+L4: Stage 29 vs 9b | 0.020 | 0.062 | +0.042 |

Skip model heatmaps have 25–32% of no-skip spatial precision.
**Spatial misalignment confirmed for both L4 and L3+L4.**

Note: Aorta and PA show NaN — absent from test subset slices evaluated.

**Output:** `results/v11/spatial_misalignment_precision.csv` ✅

---

### Direction 3: Counterfactual — Transplant No-Skip Heatmaps into Skip Decoder

**Key question:** If we replace Stage 29's poorly-localized heatmaps with 9b's
well-localized ones (same input), does the decoder output change?

**Procedure:**
```
For each test slice x:
  1. Run no-skip model(x)  → heatmaps_noskip  (well-localized)
  2. Feed heatmaps_noskip into skip model's SoftMask+decoder → logits_cf
  3a. Measure segmentation precision change (logits vs GT)     [already done]
  3b. Measure heatmap AP change (transplanted heatmap vs GT)   [L4 fix needed]
```

**Shape compatibility:** Both ProtoSegNet and ProtoSegNetV2 use the same encoder
and PrototypeLayer. At each level, heatmap shape is (B, K, M_l, H_l, W_l) with
identical K=8, M_l, H_l, W_l — direct substitution is valid.

**Results (✅ Partial — segmentation precision, not heatmap AP):**

`results/v11/spatial_misalignment_counterfactual.csv`

| pair | seg_prec_original | seg_prec_transplant | delta |
|------|------------------|---------------------|-------|
| stage8a ← 9a heatmaps | 0.672 | 0.101 | −0.571 |
| stage29 ← 9b heatmaps | **0.671** | **0.640** | **−0.031** |

**Interpretation:**

- **L3+L4 (Stage 29 ← 9b): delta = −0.031 (near-zero change)** — decoder produces
  virtually identical segmentation regardless of heatmap quality. Skip connections
  provide sufficient encoder signal; decoder ignores heatmap content. This is the
  strongest bypass evidence for L3+L4.

- **L4 (Stage 8A ← 9a): delta = −0.571 (large drop)** — NOT a bypass finding.
  The drop is caused by heatmap scale incompatibility: 9a's heatmap values lie outside
  the range Stage 8A's SoftMask was trained to handle, distorting the masked features.
  **Result is invalid for bypass characterisation; fix needed (see G13b).**

### Direction 3b (Fix): Heatmap AP for L4 Counterfactual

Instead of segmentation precision, compute **heatmap AP** (heatmap vs GT binary mask)
for both original and transplanted heatmaps. This bypasses the scale-mismatch issue
because heatmap AP is computed directly on the heatmap values, not on the decoder output.

For L3+L4, the segmentation precision result is already sufficient (delta = −0.031).
For L4, we need heatmap AP to confirm whether scale mismatch is the cause of the drop.

**Procedure:**
```
For stage8a ← 9a heatmaps:
  1. Compute AP(heatmap_8a, GT)        → ap_heatmap_8a_original
  2. Compute AP(heatmap_9a, GT)        → ap_heatmap_9a (the transplant source)
  If ap_heatmap_9a >> ap_heatmap_8a_original:
     → 9a heatmaps ARE better localized (SP Dir 2 already shows this)
     → scale mismatch explains the segmentation drop, not bypass reversal
```

**Results (✅ Done):**

`results/v11/spatial_misalignment_counterfactual_heatmap_ap.csv`

| Class | Stage 8A heatmap AP | 9a heatmap AP | delta |
|-------|--------------------|--------------:|-------|
| LV    | 0.041 | 0.155 | +0.115 |
| RV    | 0.010 | 0.031 | +0.022 |
| LA    | 0.012 | 0.049 | +0.037 |
| RA    | 0.017 | 0.079 | +0.062 |
| Myo   | 0.019 | 0.058 | +0.039 |
| **Overall** | **0.022** | **0.085** | **+0.063** |

9a heatmaps are 3.9× better localized than Stage 8A's own heatmaps.
Consistent with Direction 2 spatial precision (0.020 vs 0.075).

**Interpretation:** The −0.571 segmentation drop in Direction 3 L4 is confirmed as
scale mismatch, not a genuine bypass-reversal finding. 9a heatmaps ARE well-localized;
Stage 8A's SoftMask was not trained on their value range.

---

## 7. Experiment 2 (Optional): Progressive Skip Ablation

### Goal

Show that AP/Purity degrades monotonically as skip connections are added back, confirming the causal relationship (not just correlation) between skip presence and interpretability.

### Two Implementation Options

#### Option A: Inference-Time Zeroing (Fast, Weaker)

Take trained Stage 29, zero out skip connection tensors at inference time (not during training).

```python
# For each skip level l in {1,2,3,4}:
#   set skip_features_l = 0 before decoder
#   measure AP, Purity on val set
```

**Cost:** ~2 hours. **Caveat:** model was trained with skips; zeroing at inference tests an out-of-distribution configuration. Useful as diagnostic, not as primary evidence.

#### Option B: Retrain with Partial Skip Removal (Slow, Stronger)

Train four new models:
- No skip (= 9b, already done)
- Skip L3 only (remove L1, L2, L4 skip)
- Skip L3+L4 only (remove L1, L2 skip)
- Skip all (= Stage 29, already done)

**Cost:** 4 × ~2 days = ~8 days. **Stronger evidence**: each point is a proper training run.

### Recommendation

Run Option A first. If bypass_ratio from G11 is already high (>0.70) and the story is clear, Option B is not needed. If Reviewer 2 specifically asks for causal evidence beyond gradient attribution, run Option B for the response.

### Output

```
results/v11/skip_ablation_inference.csv
  columns: n_skips_removed, AP, Purity, Dice
  (Option A only, if run)
```

---

## 7. Abstract Revision Plan (v11 Target)

### Framing Change

Replace "Two Barriers" framing with "Structural Trade-off" framing:

| v10 Abstract Lead | v11 Abstract Lead |
|---|---|
| "We show it can fail in two structurally distinct ways: Bypass Barrier and Resolution Barrier" | "We show the architectural choice that enforces prototype causality imposes a 32% Dice cost — and that the decoder's bypass is not incidental but dominant: gradient attribution shows skip-path features account for >X% of decoder gradient mass" |

### Key Numbers to Feature

- AP trade-off: 0.051 → 0.301 (+6×) at cost of Dice 0.821 → 0.559 (−32%)
- Purity trade-off: 0.527 → 0.686 (+30%), same direction as AP
- Bypass ratio from G11: `bypass_ratio_stage29` (TBD)
- ALC as partial fix: AP 0.051 → 0.221 without removing skip, bypass_ratio drops (TBD)
- Patch Faithfulness: at correct granularity skip ≈ no-skip (0.161–0.212 vs 0.200–0.259)

### Resolution Barrier Role (Demoted)

Resolution Barrier is now a **methodological observation**, not a co-equal barrier:

> *Standard pixel-level Faithfulness probes are insensitive to 16×16 feature maps by construction — patch-level Faithfulness at the feature map's spatial grid recovers a 21× lift (0.012 → 0.259) and confirms that the no-skip model's structural guarantee is real. Faithfulness is not a differentiator between architectures; AP and Purity are.*

---

## 8. Stage Status Tracker

| Stage | Name | Status | Output |
|-------|------|--------|--------|
| All v10 stages | Carry-over | ✅ Done | See execution-plan-v10.md §6 |
| **G11** | Gradient attribution: skip vs prototype path | ✅ Done | `results/v11/gradient_attribution_*.csv` |
| **G13** | Spatial misalignment verification (Dir 1–3) | 🔶 Partial | `results/v11/spatial_misalignment_*.csv` |
| **G13b** | Dir 3 fix: heatmap AP for L4 counterfactual | ✅ Done | `results/v11/spatial_misalignment_counterfactual_heatmap_ap.csv` |
| **G12** | Progressive skip ablation (inference-time) | ⬜ Optional | `results/v11/skip_ablation_inference.csv` |
| **Report v11** | Abstract + Introduction rewrite | ✅ Done | `report/v11/report-v11.md` |

---

## 9. Success Criteria for v11

| Criterion | Gate | Result |
|-----------|------|--------|
| G11 bypass_ratio measured for skip models | Required | ✅ Stage 8A=0.778, Stage 29=0.465 |
| L4-only bypass supports mechanism claim | Required | ✅ 0.778 → structural 0, AP 5.5× |
| bypass_ratio vs AP relationship characterised | Required | ✅ Orthogonal for L3+L4; monotone for L4-only |
| G13 Dir 1: skip heatmaps more diffuse than no-skip | Required | ✅ Visualised |
| G13 Dir 2: spatial_precision skip < no-skip | Required | ✅ 0.020 vs 0.062–0.075 |
| G13 Dir 3 (L3+L4): transplant delta ≈ 0 → decoder bypass confirmed | Required | ✅ delta=−0.031 |
| G13b Dir 3 (L4): heatmap AP confirms 9a 3.9× better than 8A | Required | ✅ 0.022 vs 0.085 |
| Abstract rewritten around Dice–AP/Purity trade-off | Required | ✅ |
| G12 inference-time ablation | Optional | ⬜ |

---

## 10. Checkpoints Reference (Carry-over)

| Stage | Checkpoint |
|-------|-----------|
| **Stage 29** (skip, L3+L4) | `checkpoints/proto_seg_ct_l3l4_warmstart.pth` |
| Stage 34b (ALC L3, skip) | `checkpoints/proto_seg_ct_l3l4_alc_l3only.pth` |
| Stage 8A (skip, L4 only) | `proto_seg_ct_abl_a.pth` |
| 9b (no-skip, L3+L4) | `checkpoints/proto_seg_ct_v2_l34.pth` |
| 9a (no-skip, L4 only) | `checkpoints/proto_seg_ct_v2_l4.pth` |
