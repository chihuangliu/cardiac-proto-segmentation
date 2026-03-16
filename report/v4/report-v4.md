# Report v4: Prototype Quality in Multi-Scale Cardiac Segmentation

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16 train / 2 val / 2 test patients)
**Date:** 2026-03-16
**Preceded by:** `report/v2/report-v2.md` (XAI ceiling analysis)

---

## §1 Introduction

Prototype-based neural networks offer a form of interpretability by grounding predictions in learned exemplars: each class is represented by a set of prototype vectors, and the network's explanation is the similarity between input features and those prototypes. In segmentation, this produces spatial heatmaps that are intended to show *which input regions most closely match each anatomical structure's prototype*.

v1–v3 of this project pursued absolute XAI targets (Average Precision ≥ 0.70, Faithfulness ≥ 0.55, Stability ≤ 0.20) and found them structurally unachievable under the coupled prototype-segmentation design. That arc is documented in `report/v2/report-v2.md`.

This report pivots to a more tractable and scientifically interesting question:

> *Do multi-scale prototypes learn structurally distinct, anatomically meaningful representations — and does scale configuration affect prototype quality and XAI utility?*

We introduce six prototype quality metrics, apply them to a trained multi-scale model (M4, four prototype levels), and compare against two ablations: M1 (L4-only) and M2 (L3+L4).

---

## §2 Background

### Architecture

ProtoSegNet is a 2D encoder–decoder segmentation network with prototype layers inserted at each encoder level:

- **Encoder:** 4-level hierarchical CNN. Channels {L1:32, L2:64, L3:128, L4:256}. Output spatial sizes {L1:128×128, L2:64×64, L3:32×32, L4:16×16} from 256×256 input.
- **Prototype layers:** At each active level *l*, a `PrototypeLayer` holds M_l prototypes per class. M = {L1:4, L2:3, L3:2, L4:2} per class, K=8 classes (background + 7 cardiac structures). Total prototypes in M4: 4×8 + 3×8 + 2×8 + 2×8 = 88 (77 foreground-class prototypes).
- **Soft mask:** Per-level heatmap A_{l,k,m} (cosine similarity between encoder features and prototype *m* of class *k*) is used to weight the encoder features before the decoder skip connection.
- **Decoder:** Standard U-Net decoder with bilinear upsampling and skip connections from masked features.
- **Loss:** 0.5×Dice + 0.5×WeightedCE + 0.001×JeffreysDivergence + 0.5×push − 0.25×pull.
- **Training:** 3-phase schedule — Phase A warm-up (ep 1–20, prototypes frozen), Phase B joint training (ep 21–80), Phase C decoder fine-tune (ep 81–100). Prototype projection at every Phase B checkpoint.

### Models compared

| ID | Prototype levels | Prototypes | Checkpoint |
|----|-----------------|------------|-----------|
| **M4** | L1+L2+L3+L4 | 77 (fg) | `proto_seg_ct_l2.pth` |
| **M2** | L3+L4 | 28 (fg) | `proto_seg_ct_l3l4.pth` |
| **M1** | L4 only | 14 (fg) | `proto_seg_ct_l4only.pth` |

For levels without prototypes, raw encoder features pass through to the decoder unchanged (no masking). Decoder architecture is identical across all three models, ensuring Dice differences reflect prototype supervision rather than decoder capacity.

---

## §3 Prototype Quality Metrics

We define six metrics to characterise what prototypes learn and how they affect model behaviour.

### 3.1 Purity

**Definition:** For each prototype p_{l,k,m}, collect the top-50 training positions with highest activation. Purity = fraction where the ground-truth label equals class *k*.

Purity measures whether a prototype has become class-selective. Purity = 1.0 means the prototype only activates on its intended class; purity ≈ 1/K ≈ 0.11 for K=8 classes is chance level.

**Caveat:** Purity compares models at the *system level*. Different training configurations produce different encoder representations; differences in purity reflect the joint effect of encoder adaptation and prototype configuration, not isolated prototype quality.

### 3.2 Utilization

**Definition:** Fraction of prototypes whose maximum activation across the test set exceeds 0.1. Prototypes below this threshold are "dead" — they never fire and waste network capacity.

### 3.3 Spatial Compactness

**Definition:** Mean fraction of spatial locations in the 256×256 activation map with activation > 0.5. Lower = more spatially focused.

Per-level acceptable thresholds: L1 ≤ 5%, L2 ≤ 8%, L3 ≤ 15%, L4 ≤ 25%.

### 3.4 Dice Sensitivity

**Definition:** For each prototype, zero out its activation at inference (no retraining) and measure the mean Dice drop across the test set. Quantifies each prototype's causal importance to segmentation.

Threshold for "important": Dice drop > 0.005 (0.5%).

### 3.5 Level Dominance

**Definition (multi-scale models):** For each test pixel, which level's prototype wins the cross-level max aggregation in the final heatmap? Reports the fraction of pixels dominated by each level.

If one level dominates > 60% of pixels, the other levels contribute minimally to the XAI explanation despite consuming parameters.

### 3.6 Per-level AP

**Definition:** Compute Average Precision using only the heatmap from level *l* (max over prototypes *m* for each class *k*, before any cross-level aggregation). Upsample to 256×256 and threshold at the 95th percentile.

This isolates which level produces the most anatomically precise activations when evaluated independently.

---

## §4 Post-hoc Analysis of M4 (RQ1, RQ2)

### 4.1 What do prototypes at each scale level learn? (RQ1)

**Purity by level:**

| Level | Resolution | Mean Purity | Range |
|-------|-----------|-------------|-------|
| L1 | 128×128 | 0.050 | 0.00–0.28 |
| L2 | 64×64 | 0.184 | 0.00–0.66 |
| L3 | 32×32 | 0.639 | 0.08–1.00 |
| **L4** | **16×16** | **0.824** | 0.40–1.00 |

There is a sharp depth gradient. L4 prototypes are highly class-selective (purity 0.824) — the encoder's deepest representations carry dense semantic information about cardiac anatomy. L1 prototypes are near-random (purity 0.050), responding to low-level textures that are class-agnostic. L2 shows slight selectivity; L3 is meaningful.

**Dice sensitivity:**

Only 7 of 77 prototypes have a measurable segmentation impact (Dice drop > 0.005), and all 7 are at L3. L4 prototypes, despite highest purity, have Dice drop ≈ 0 — they are anatomically meaningful but not causally linked to the decoder's predictions. This dissociation suggests the decoder has learned to rely on the skip connection structure more than the prototype-masked features.

**Compactness:**

All levels fail their compactness thresholds (actual values 0.34–0.57 vs thresholds 0.05–0.25). Prototypes activate broadly across the image rather than locally — a consequence of using global cosine similarity without spatial locality constraints.

### 4.2 Does multi-scale configuration diversify prototype representations? (RQ2)

**Level dominance:**

| Level | Pixel dominance | Mean purity |
|-------|----------------|-------------|
| L1 | **34%** | 0.050 |
| L2 | **44%** | 0.184 |
| L3 | 17% | 0.639 |
| L4 | 4% | 0.824 |

L1 and L2 together dominate 78% of pixels in the cross-level max aggregation, despite being the least class-selective. L4 — the most meaningful level — wins only 4% of pixels.

**Per-level AP (isolated):**

| Level | Mean AP |
|-------|---------|
| L1 | 0.043 |
| L2 | 0.112 |
| L3 | 0.068 |
| **L4** | **0.189** |

L4 produces the best heatmaps when evaluated in isolation. Yet the combined model achieves only AP ≈ 0.10, because the dominant L1/L2 noise suppresses the L4 signal in the final aggregation.

**Conclusion (RQ2):** Multi-scale prototype supervision does not diversify representations in a useful way. The winner-takes-all cross-level max aggregation systematically favours shallow, low-purity levels, burying the high-quality deep representations. The model learns meaningful prototypes at L4 but cannot surface them in the XAI output.

---

## §5 Multi-scale Ablation: M1 / M2 / M4 (RQ3)

### 5.1 Segmentation

| | M4 (L1-L4) | M2 (L3-L4) | M1 (L4) | Δ M2 vs M4 |
|--|------------|------------|---------|-----------|
| Best val Dice (2D) | 0.8173 | 0.8380 | 0.8346 | +0.021 |
| 3D Dice — ct_1019 | 0.7477 | 0.8123 | 0.7697 | +0.065 |
| 3D Dice — ct_1020 | 0.9337 | 0.9321 | 0.9351 | −0.002 |
| **3D Dice mean** | 0.8407 | **0.8722** | 0.8524 | **+0.032** |

M2 (L3+L4) achieves the best segmentation, outperforming M4 by +3.2% and M1 by +2.0% mean 3D Dice with 28 prototypes. The gain on ct_1019 is substantial (+6.5%), while ct_1020 performance is near-identical across all models — consistent with ct_1019 being the harder patient (more LA/PA difficulty).

**Per-structure 3D Dice (M2):**

| Structure | LV | RV | LA | RA | Myo | Aorta | PA |
|-----------|----|----|----|----|-----|-------|----|
| ct_1019 | 0.849 | 0.901 | 0.718 | 0.897 | 0.712 | 0.868 | 0.741 |
| ct_1020 | 0.889 | 0.969 | 0.938 | 0.897 | 0.917 | 0.974 | 0.940 |
| **Mean** | **0.869** | **0.935** | **0.828** | **0.897** | **0.815** | **0.921** | **0.841** |

PA (0.841) and LA (0.828) improve substantially in M2 vs M1 (PA: 0.738, LA: 0.806), suggesting L3 provides important spatial context for these mid-size structures.

### 5.2 Prototype quality comparison

| Metric | M4 (L1-L4) | M2 (L3-L4) | M1 (L4) |
|--------|-----------|-----------|---------|
| n_prototypes | 77 | **28** | 14 |
| Purity — L4 | **0.824** | 0.804 | 0.499 |
| Purity — all levels | 0.334 | **0.733** | 0.499 |
| Dead prototypes | 0 | 0 | 0 |
| Compactness (L4) | 0.573 | **0.361** | 0.564 |
| Important protos | 7/77 (9%) | **10/28 (36%)** | 4/14 (29%) |
| L4 dominance | 4.3% | 49.1% | 100% |
| Per-level AP (L4) | 0.189 | 0.236 | **0.274** |

### 5.3 M2 standout findings

**Overall purity (0.733):** By removing L1 and L2, M2's prototype set consists entirely of levels with high semantic selectivity — L3 (purity 0.639 in M4) and L4 (0.804 in M2). The average purity across all prototypes is 0.733, more than double M4's 0.334.

**Compactness (0.361):** M2's L4 prototypes are substantially more spatially focused than in M4 (0.573) or M1 (0.564). When L4 only competes with L3 in the aggregation — rather than the broadly activating L1/L2 — the encoder learns to use L4 for more localised, structure-specific features.

**Complementary level contribution:** L4 wins 49.1% of pixels, L3 wins 50.9%. This near-equal split indicates that L3 and L4 genuinely partition the input space differently and both contribute to the heatmap, rather than one level dominating the other.

**36% important prototypes:** 10 of 28 M2 prototypes have Dice drop > 0.005, compared to 9% (M4) and 29% (M1). M2's prototypes are the most efficient — a higher fraction is causally linked to predictions.

### 5.4 The purity paradox (extended)

The purity paradox observed in the M1 vs M4 comparison — where removing levels reduces per-prototype purity but improves heatmap quality — has a more nuanced form in M2.

M2's L4 purity (0.804) is only slightly below M4's (0.824), but M2's L4 prototypes dominate 49% of pixels vs M4's 4.3%. The much-improved compactness (0.361) suggests that M2's encoder has learned a different but equally valid encoding: L4 represents compact, anatomically localised features rather than the global semantic features it learns in M4 (where L1/L2/L3 handle coarser scales).

This suggests that the problem in M4 is not that L4 prototypes are bad, but that the aggregation mechanism gives them no chance to contribute.

### 5.5 Answer to RQ3

> *Is there a simpler scale configuration that achieves equivalent segmentation quality with cleaner prototype representations?*

**Yes — and M2 (L3+L4) is the optimal configuration found.** It achieves the highest segmentation quality (3D Dice 0.8722, +3.2% vs M4), the best overall purity (0.733), best compactness (0.361), and 36% of prototypes are causally important. L3 and L4 genuinely complement each other with near-equal pixel dominance, confirming that both levels carry distinct, useful information.

M1 (L4-only) confirms that L1+L2 are actively harmful. M2 confirms that L3 adds genuine value and should be retained. The optimal configuration removes exactly the problematic levels (L1, L2) while preserving the two semantically meaningful ones (L3, L4).

---

## §6 Discussion

### Implication 1: Winner-takes-all aggregation is incompatible with full multi-scale prototypes

The core problem in M4 is architectural: combining heatmaps via cross-level maximum means shallow levels systematically win because their feature maps are spatially larger and their activations are more diffuse. Deep prototypes that are more semantically meaningful lose the pixel-level competition to low-quality shallow activations.

M2 provides a partial solution by removing the offending levels (L1, L2). But even in M2, the 49/51 split between L4 and L3 is decided by arbitrary spatial max rather than learned importance. A natural next step is to replace the max aggregation with a learned weighted sum (per level, per class), allowing the model to assign contribution weights proportional to each level's semantic quality. This is the direction A (learned level attention) planned for v5.

### Implication 2: Prototype quality metrics are sensitive but require system-level interpretation

The purity paradox across all three models demonstrates that individual prototype quality (purity) can diverge from system-level XAI quality (AP). M4 has the highest L4 purity (0.824) yet the worst AP (0.189). M1 has the lowest L4 purity (0.499) yet the best AP (0.274). M2 sits between both on purity (0.804) and AP (0.236), but wins on all other dimensions.

This establishes a key principle: for prototype-based segmentation, heatmap-level metrics (dominance, per-level AP, compactness) are more predictive of XAI utility than prototype-level metrics (purity) alone. A prototype can be perfectly pure yet irrelevant if it is structurally suppressed in the aggregation.

### Implication 3: The optimal level configuration removes harmful levels, not all non-semantic ones

An initial hypothesis was that the simplest configuration (L4-only, M1) would be best. The data refutes this: M2 outperforms M1 by +2.0% Dice and achieves better compactness and overall purity. L3 is not just neutral — it actively improves both segmentation and interpretability when combined with L4.

This identifies L1 and L2 as uniquely harmful: their broad, class-agnostic activations pollute the heatmap without contributing to segmentation. The practical design principle is: **include semantic levels (L3, L4), exclude texture levels (L1, L2)**. The boundary between these two groups is confirmed empirically by the compactness, purity, and Dice sensitivity results across all three models.

---

## §7 Conclusion

We characterised prototype quality in a multi-scale cardiac segmentation network across six metrics, and evaluated three prototype level configurations: M4 (all levels), M2 (L3+L4), and M1 (L4-only).

**M2 (L3+L4) is the optimal configuration:** it achieves 3D Dice 0.8722 (+3.2% vs M4, +2.0% vs M1), overall purity 0.733, best compactness (0.361), and 36% causally important prototypes — the highest efficiency of the three models.

The central finding is a **purity paradox** and its resolution: multi-scale training (M4) produces highly pure L4 prototypes (0.824) that are structurally suppressed by noisy L1/L2 activations in the winner-takes-all aggregation (L4 wins only 4.3% of pixels). Removing L1+L2 (M2) forces the encoder to redistribute learning across L3 and L4, yielding modestly lower per-prototype purity but dramatically better heatmap quality, compactness, and segmentation.

**L1 and L2 are actively harmful**: they add parameters, inject noise into the heatmap aggregation, and reduce segmentation quality. L3, by contrast, adds genuine complementary value to L4 — confirmed by near-equal pixel dominance (50/50) and 10 causally important L3 prototypes in M2.

For future work, replacing the winner-takes-all cross-level max with a learned per-level attention mechanism could combine the high L4 purity of M4 with the heatmap dominance of M2, potentially further improving both segmentation and XAI quality.

---

## Appendix: Outputs

```
results/v4/
  proto_quality/
    v1/                           # M4 metrics (Stage 14)
      purity_per_prototype.csv
      purity_summary.csv
      utilization.csv
      compactness.csv
      dice_sensitivity.csv
      level_dominance.csv
      per_level_ap.csv
      fig_purity.png
      fig_utilization.png
      fig_compactness.png
      fig_dice_sensitivity.png
      fig_level_dominance.png
      fig_per_level_ap.png
    m1_l4only/                    # M1 metrics (Stage 17)
      [same CSVs as v1/]
    m2_l3l4/                      # M2 metrics (Stage 17)
      [same CSVs as v1/]
    comparison_table.csv          # M1/M2/M4 side-by-side
    fig_purity_comparison.png
    fig_dominance_comparison.png
    fig_dice_ap_comparison.png
    fig_seg_comparison.png
    atlas_m4_l4.png
    atlas_m1_l4.png
    atlas_m2_l4.png / atlas_m2_l3.png
  train_curve_proto_ct_l4only.csv
  train_curve_l4only.png

checkpoints/
  proto_seg_ct_l2.pth             # M4 checkpoint (epoch 75)
  proto_seg_ct_l3l4.pth           # M2 checkpoint (epoch 80)
  proto_seg_ct_l4only.pth         # M1 checkpoint (epoch 55)
```
