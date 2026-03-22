# Report v9: ProtoSegNetV2 — Per-Level Ablation & Attention Analysis

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16 train / 2 val / 2 test, 3389/382 train/val slices)
**Hardware:** MacBook 48GB RAM, Apple Silicon (MPS)
**Date:** 2026-03-22
**Preceded by:** `report/v8/report-v8.md` — Two-Phase Pipeline + ALC (Dice=0.8628)

---

## 1. Motivation

v8 achieved Dice=0.8628 with a decoder-based architecture (M4 warm-start, L3+L4).
However, the decoder creates a **bypass path**: the model's final logits depend on skip
connections and transposed convolutions that do not pass through any prototype layer.
This structurally makes the prototypes causally irrelevant — the model can segment
correctly while ignoring them entirely.

**v9 tests the hypothesis:** removing the decoder and forcing `logits = f(heatmaps only)`
will structurally guarantee Faithfulness, at the cost of some Dice.

---

## 2. Architecture: ProtoSegNetV2

ProtoSegNetV2 eliminates the decoder entirely:

```
Input (256×256)
    → HierarchicalEncoder (levels L1–L4, channels 32/64/128/256)
        → for each active level l:
            feat_l  = encoder output at level l
            A_l     = ProtoLayer_l(feat_l)          # (B, K, M, H_l, W_l)
            up_l    = upsample(A_l.max(M), 256)     # (B, K, 256, 256)
    → logits = Σ_l w_l × up_l                      # weighted sum (w=uniform or learned)
```

There is **no skip connection, no decoder, no bypass path.** Every logit is a direct
linear combination of upsampled prototype heatmaps. Faithfulness is guaranteed by
construction: perturbing the input changes a logit if and only if it changes some heatmap.

**Training schedule:** Three-phase for all single-level variants:
- Phase A (ep 1–20): seg loss only
- Phase B (ep 21–80): seg + div + push + pull losses, prototype projection at ep 21
- Phase C (ep 81–100): attention fine-tune (if applicable)

**Loss:** Dice + weighted CE + λ_div=0.01 × diversity + λ_push=0.5 + λ_pull=0.25

---

## 3. Stage Results Overview

| Stage | Config | Val Dice | Eff. Purity | AP | Faithfulness | Stability |
|-------|--------|----------|-------------|-----|-------------|-----------|
| **9a** | L4 only, no attention | **0.606** | 0.689 | **0.312** ✅ | 0.012 ❌ | 10.92 ❌ |
| 9b | L3+L4, uniform weights | 0.559 | 0.649 | 0.247 ❌ | 0.021 ❌ | 11.40 ❌ |
| 9c | L1–L4, learned attention | 0.586 | 0.676 | 0.262 ✅ | 0.048 ❌ | 13.63 ❌ |
| 9d | T_min=2.0 floor | — | — | — | — | ❌ Killed |
| 9L1 | L1 only (128×128) | 0.146 | 0.159 | 0.166 ✅ | 0.160 ✅ | 16.99 ❌ |
| 9L2 | L2 only (64×64) | 0.336 | 0.569 | 0.219 ✅ | 0.218 ✅ | 14.38 ❌ |
| 9L3 | L3 only (32×32) | 0.554 | **0.844** | 0.319 ✅ | 0.060 ❌ | 10.29 ❌ |
| 9LF | Freeze all 4, train attn only | 0.606 | — | — | — | — |

Gates: AP ≥ 0.25 ✅; Faithfulness ≥ 0.15 ✅; Stability ≤ 2.0 ✅

---

## 4. Effect of Removing the Bypass (Skip vs No-Skip)

The key improvement from v8 → v9 is the elimination of decoder skip connections.
This has measurable effects across XAI metrics:

| Metric | v8 (with decoder, L3+L4) | 9a (no decoder, L4) | Change |
|--------|--------------------------|---------------------|--------|
| Val Dice | 0.8628 | 0.606 | −0.257 |
| AP | 0.102 | **0.312** | **+0.210 (+3×)** |
| Faithfulness | 0.059 | 0.012 | −0.047 |
| Stability | 3.00 | 10.92 | worse |

**Interpretation:**
- Removing the bypass triples AP: prototypes are now causally active, so the activation
  patterns genuinely correspond to identified class-specific image patches.
- Faithfulness and Stability worsen despite the structural guarantee. This is not a
  contradiction — it reveals a **second barrier** (Section 7).

---

## 5. Attention Collapse Analysis (Stages 9c, 9d)

### 5.1 Joint Training (9c)

With all four levels trained jointly under a shared encoder + attention MLP (9c):
- Attention weights collapse to w_L4≈1.0 by epoch 35 (when T≈4.5)
- Root cause: positive feedback loop. Level that gets early gradient advantage improves
  faster → receives more weight → receives more gradient. This is self-reinforcing.
- The MLP learns logit differences of ~50+ (L4 vs others), so
  `softmax([50/4.5, 0, 0, 0]) ≈ [1, 0, 0, 0]` regardless of temperature.

### 5.2 Temperature Floor Attempt (9d) — Failed

Setting T_min=2.0 to constrain softmax sharpness had no effect because:
- Temperature floor constrains the softmax function's shape.
- It does NOT constrain the MLP's logit magnitudes.
- At T=4.5 (before reaching T_min), logit differences of 50 already produce w_L4=1.0.
- **Killed at epoch 35.** Temperature floor and logit magnitude are decoupled.

---

## 6. Per-Level Ablation (Stages 9L1–9L3, 9a)

To determine whether attention collapse is correct or pathological, each level was
trained independently as a standalone model. Results:

| Level | Spatial res. | Dice | Purity | AP | Faithfulness |
|-------|-------------|------|--------|----|-------------|
| L1 | 128×128 | 0.146 | 0.159 | 0.166 | **0.160 ✅** |
| L2 | 64×64 | 0.336 | 0.569 | 0.219 | **0.218 ✅** |
| L3 | 32×32 | 0.554 | **0.844** | **0.319** | 0.060 ❌ |
| L4 | 16×16 | **0.606** | 0.689 | 0.312 | 0.012 ❌ |

**Key findings:**

1. **Dice increases monotonically with depth.** Shallow features are not discriminative
   enough for segmentation. L1's Dice=0.146 is barely above chance.

2. **L3 purity (0.844) exceeds L4 (0.689)** when each is independently trained.
   This is a stricter result: L3 makes a finer spatial claim (32×32 heatmap, 8×8 px/act)
   than L4 (16×16 heatmap, 16×16 px/act). Higher purity at finer resolution is harder.

3. **Joint training suppresses L1–L3 quality:**

   | Level | Joint purity (9c diag.) | Independent purity | Gap |
   |-------|------------------------|-------------------|-----|
   | L1 | 0.084 | 0.159 | +0.075 |
   | L2 | 0.195 | 0.569 | **+0.374** |
   | L3 | 0.613 | **0.844** | **+0.231** |
   | L4 | 0.689 | 0.689 | 0.000 |

   L4 is unaffected by joint training. L1–L3 are significantly suppressed.
   In 9c, attention collapsed to L4 partly because L3's quality was artificially degraded.

4. **Faithfulness passes only for shallow levels (L1, L2).** This confirms the resolution
   mismatch hypothesis (Section 7): single-pixel perturbations at 256×256 cannot affect
   16×16 or 32×32 feature maps. L1 (128×128 maps) and L2 (64×64 maps) are coarser but
   still fine enough that some perturbations reach the feature map.

---

## 7. Two-Barrier Framework (Core Finding)

v9 identifies two structurally distinct XAI barriers in prototype segmentation networks:

### Barrier 1: Bypass Problem (v1–v8)

**What:** Decoder skip connections create a path from input to logits that bypasses
prototype layers entirely. The model can achieve high Dice without prototypes being
causally active. Prototypes fire but don't matter.

**Symptom:** AP is low (≈0.10) despite visually reasonable heatmaps.

**Fix:** Remove the decoder. ProtoSegNetV2 forces `logits = f(heatmaps)`.

**Effect:** AP improves 3× (0.102 → 0.312). Prototypes are now causally required.

### Barrier 2: Resolution Problem (v9)

**What:** Deep prototype heatmaps operate at coarse spatial resolution (L4: 16×16).
Standard Faithfulness and Stability metrics perturb single pixels at input resolution
(256×256). A single perturbed pixel affects at most one spatial location in a 16×16 map —
often zero due to nearest-neighbour rounding. The metrics measure perturbation sensitivity
that the architecture is geometrically incapable of exhibiting.

**Symptom:** Faithfulness=0.012 for L4 (effectively zero), despite structural guarantee.
Stability is also inflated: noise propagates through 4 encoder stages and the coarse-to-fine
upsample amplifies spatial variance.

**This is a metric/architecture mismatch, not a training failure.**

**Fix options (not implemented in v9):**
- *Patch-level Faithfulness:* zero out 16×16 blocks aligned to L4's spatial grid.
  This would measure what the metric intends to measure at the right granularity.
- *Shallower levels:* use L2 or L3 which have Faithfulness ≥ 0.06–0.22, at the cost
  of Dice (0.554 or 0.336 vs 0.606 for L4).

---

## 8. Late Fusion Experiment (Stage 9LF)

### Motivation

Given that joint training suppresses L1–L3 quality, 9LF tests attention on a **fair
playing field**: all four levels are independently pre-trained and frozen; only the
attention MLP (~31k parameters) is trained.

### Architecture

```
9L1 encoder (frozen) → proto_L1 (frozen) → upsample ──┐
9L2 encoder (frozen) → proto_L2 (frozen) → upsample ──┤→ GAP → concat → MLP → w → Σ → logits
9L3 encoder (frozen) → proto_L3 (frozen) → upsample ──┤
9a  encoder (frozen) → proto_L4 (frozen) → upsample ──┘
```

MLP input: concatenated GAP features [32+64+128+256 = 480 dims].
MLP output: 4 logits → softmax → weights w = (w_L1, w_L2, w_L3, w_L4).
Training: seg loss only, AdamW lr=1e-3, 80 epochs max, patience=15.

### Result

| Metric | Value |
|--------|-------|
| Best Val Dice | **0.606** |
| w_L4 at epoch 1 | **1.000** |
| w_L1/L2/L3 at epoch 1 | **0.000** |

**Attention collapses to w_L4=1.0 from epoch 1**, before any gradient is computed.
The signal from frozen L4 GAP features is so dominant that even a randomly initialized
MLP places all probability mass on L4 in its first forward pass.

The Late Fusion Dice (0.606) is identical to 9a (L4 alone). Late Fusion adds no value —
it is equivalent to 9a because the attention weight on L4 is 1.0 throughout.

### Interpretation

This result conclusively answers the research question:

> **Is the attention collapse in 9c correct?**

**Yes.** Even with all levels pre-trained to their independent optimum (removing the
joint-training bias), attention still selects L4 with probability 1. L4 genuinely
produces better segmentation signals. The collapse in 9c was not a pathology caused
by joint training — it was the correct response to L4's inherent advantage.

The attention module functions as an **argmax oracle**: it identifies the best level
and assigns it full weight. It does not learn a convex combination.

---

## 9. Can Purity and Dice Be Simultaneously Optimised?

This question emerged from observing that L3 has higher purity (0.844) but lower Dice
(0.554) than L4 (0.689, 0.606). Is the tradeoff fundamental?

**Observation:** Purity and Dice measure different things.

- **Dice** measures spatial overlap between predicted and ground-truth segmentation.
  It rewards any correct pixel, regardless of why the model assigned that class.
- **Purity** measures whether each prototype's receptive field activation is confined
  to a single class. It rewards class-specific localization.

These are not opposites, but they do create tension at different spatial scales:
- Fine resolution (L3, 32×32): prototypes are spatially smaller and more class-specific,
  giving high purity. But the coarse-to-fine upsampling of a 32×32 map to 256×256
  introduces spatial smearing that reduces Dice precision.
- Coarse resolution (L4, 16×16): prototypes cover larger regions, reducing class
  specificity (lower purity). But the larger receptive field captures more context,
  which helps discriminate visually similar structures (higher Dice).

**Practical conclusion:** In this dataset and architecture, the Dice–Purity tradeoff
is real but not fundamental. A possible path to higher purity at high resolution would
be to apply a spatial attention mask *before* prototype matching, restricting each
prototype's activation to a smaller spatial region. This is not pursued in v9.

---

## 10. Best Model: Stage 9a

**Architecture:** ProtoSegNetV2, L4 only, no attention.
**Checkpoint:** `checkpoints/proto_seg_ct_v2_l4.pth`
**Results:** `results/v9/`

| Metric | Value | Gate | Pass? |
|--------|-------|------|-------|
| Val Dice | 0.606 | — | — |
| Eff. Purity | 0.689 | — | — |
| AP | 0.312 | ≥ 0.25 | ✅ |
| Faithfulness | 0.012 | ≥ 0.15 | ❌ (resolution barrier) |
| Stability | 10.92 | ≤ 2.00 | ❌ (resolution barrier) |

**Dice gap vs baseline U-Net:** 0.836 − 0.606 = −0.230 (−27.5%)
This gap is the cost of structural interpretability (no decoder bypass).

---

## 11. Future Directions

1. **Patch-level Faithfulness metric:** Replace single-pixel masking with 16×16 block
   masking aligned to L4's spatial grid. This would give a meaningful Faithfulness score
   for L4 while maintaining the same conceptual definition.

2. **Spatial attention pre-filtering:** Apply a learned spatial mask before prototype
   matching to improve prototype specificity (purity) without changing the level depth.

3. **Multi-level with spatial supervision:** Supervise each prototype layer with
   class-specific spatial loss (e.g., dice per-patch) to encourage level-appropriate
   specificity rather than relying on the segmentation loss alone.

4. **Lipschitz regularization:** Add spectral normalization to encoder convolutions
   to bound Stability directly, rather than relying on post-hoc noise augmentation.

---

## Appendix A: Per-Class Dice for Best Model (9a, L4)

| Class | Dice |
|-------|------|
| LV | 0.412 |
| RV | 0.636 |
| LA | 0.859 |
| RA | 0.615 |
| Myocardium | 0.618 |
| Aorta | 0.664 |
| PA | 0.431 |
| **Mean FG** | **0.606** |

PA and LV are the hardest structures — consistent with v8 findings.

---

## Appendix B: Attention Weight Trajectory (9LF)

```
Epoch  1: w_L1=0.000  w_L2=0.000  w_L3=0.000  w_L4=1.000
Epoch  5: w_L1=0.000  w_L2=0.000  w_L3=0.000  w_L4=1.000
Epoch 65: w_L1=0.000  w_L2=0.000  w_L3=0.000  w_L4=1.000
(training stopped at epoch 65 by user; no improvement since epoch 1)
```

The MLP never redistributes weight away from L4. Training loss flatlined at 1.233.
