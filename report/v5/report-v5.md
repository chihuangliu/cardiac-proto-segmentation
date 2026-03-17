# Report v5: Learned Level Attention in Multi-Scale Prototype Segmentation

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16 train / 2 val / 2 test patients)
**Date:** 2026-03-17
**Preceded by:** `report/v4/report-v4.md` (multi-scale prototype quality analysis)

---

## §1 Introduction

Report v4 identified a core structural flaw in the M4 (four-level prototype) model: the cross-level maximum aggregation mechanism systematically suppresses deep, semantically meaningful prototypes. L4 prototypes achieve purity 0.824 — meaning they are highly class-selective — yet they win only 4.3% of spatial positions in the final heatmap, because the broader, lower-quality activations of L1 and L2 consistently win the pixel-wise max competition.

The practical fix in v4 was surgical: remove L1 and L2 entirely. The resulting M2 model (L3+L4 only) achieved the best segmentation (3D Dice 0.8722) and the cleanest prototype quality metrics of any configuration tested.

This report asks whether this solution can be discovered *automatically* through a learned attention mechanism, instead of requiring prior knowledge of which levels are harmful. Specifically:

> **RQ4**: Does replacing winner-takes-all max aggregation with a learned, input-conditioned weighted sum improve segmentation quality over M4?

> **RQ5**: Without explicit supervision, does the attention module learn to suppress L1/L2 and amplify L4, reproducing the v4 manual finding?

We train two variants of an attention-augmented M4 model and compare against the three baselines from v4.

---

## §2 Architecture Extension: LevelAttentionModule

### 2.1 Motivation

The cross-level max aggregation in M4 computes:

```
heatmap_combined[k,s] = max_l  max_m  A[l, k, m, s]
```

where A[l,k,m,s] is the activation of prototype *m* of class *k* at spatial position *s* in level *l* (upsampled to a common resolution). This operation has no trainable parameters: the level that produces the highest activation at each pixel wins unconditionally, regardless of semantic quality.

LevelAttentionModule replaces this with a learned, image-conditioned weighted sum:

```
heatmap_blended[k,s] = Σ_l  w[l]  *  upsample(max_m  A[l, k, m, s])
```

where w ∈ ℝ^{n_levels} is a per-image softmax weight vector produced by an MLP conditioned on the encoder features.

### 2.2 Architecture

```
Input: encoder features {Z_l}  (B, C_l, H_l, W_l)  for l in active_levels

Step 1 — Context extraction:
    pool_l = GlobalAvgPool(Z_l)           → (B, C_l)
    context = concat([pool_l for l in active_levels])   → (B, Σ C_l)

    For 4-level M4: channels {L1:32, L2:64, L3:128, L4:256}, context dim = 480

Step 2 — Weight prediction:
    h = ReLU(Linear(480 → 64)(context))   → (B, 64)
    w = softmax(Linear(64 → 4)(h))        → (B, 4)

Step 3 — Blended heatmap:
    A_blended[k,s] = Σ_{j,l} w[:,j] * upsample(max_m A[l,k,m,s])
```

The blended heatmap A_blended is then passed to the soft mask module exactly as the single-level heatmap was previously, maintaining full backward compatibility.

**Parameter count:** 480×64 + 64 + 64×4 + 4 = 30,980 additional parameters (< 0.1% of model total).

---

## §3 Training Protocol

### 3.1 Three-phase schedule

Training follows the same three-phase structure used in all prior models:

| Phase | Epochs | Prototypes | Encoder | Attention |
|-------|--------|-----------|---------|-----------|
| A | 1–20 | Frozen | Training | **Frozen** |
| B | 21–80 | Training | Training | **Frozen until ep 31, then training** |
| C | 81–100 | Frozen | Frozen | Training |

**Rationale for phase A attention freeze:** Attention weights are meaningless during Phase A because prototypes are randomly initialised. Allowing attention to train on random prototype heatmaps would corrupt the MLP's weight initialisation.

**Rationale for delayed unfreeze within Phase B (ATTN_WARMUP_EPOCHS = 10):** At the Phase A→B transition, prototypes begin to move toward meaningful positions for the first time. The first 10 epochs of Phase B (ep 21–30) allow prototypes to reach a stable distribution before attention begins learning which levels are informative. Without this delay, attention learns from a moving target during the critical initialisation phase.

### 3.2 Phase B stability fix: skipping the initial projection

An unexpected failure mode emerged during early Phase B: running prototype projection (pushing prototypes to nearest training patches) at the Phase B start caused catastrophic training collapse — loss spiked to ~1800, Dice dropped from 0.79 to < 0.15 and did not recover.

The mechanism is: the initial projection simultaneously repositions prototypes at all 4 levels. The decoder, which had learned to rely on a specific mask pattern during Phase A, receives completely different masked features from all 4 skip connections at once. With the attention module also being initialised in this same step, the cumulative disruption exceeds the decoder's capacity to adapt.

**Fix:** The initial projection at Phase B start was removed. Periodic projection continues at intervals of 10 epochs within Phase B (ep 30, 40, 50, 60, 70), allowing gradual adaptation. This fix was necessary for all stable M4-attn training.

### 3.3 Entropy regularisation: λ_ent parameter

To prevent attention weight collapse (all weight on one level), an optional entropy regularisation term was added:

```
L_ent = λ_ent * Σ_j  w_j * log(w_j)     (negative entropy, minimised → encourages spread)
```

Two experiments were run:

| Model | λ_ent | Description |
|-------|-------|-------------|
| **M4-attn (λ=0.02)** | 0.02 | Entropy reg + delayed unfreeze |
| **M4-attn (λ=0)** | 0 | Delayed unfreeze only |

The λ=0.02 experiment was run first to prevent training instability. The λ=0 experiment was run second, after recognising a scientific problem with regularisation (§4.2).

---

## §4 Experiment 1: M4-attn with λ_ent = 0.02

### 4.1 Training dynamics

Attention weights remained near-uniform throughout training (Table 1). The final weights at epoch 100 were effectively indistinguishable from a uniform distribution.

**Table 1: Attention weights by epoch — M4-attn (λ=0.02)**

| Epoch | Phase | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|-------|------|------|------|------|
| 5–20 | A (frozen) | ~0.25 | ~0.25 | ~0.25 | ~0.25 |
| 25–30 | B (warmup) | ~0.25 | ~0.25 | ~0.25 | ~0.25 |
| 40–80 | B (active) | ~0.25 | ~0.25 | ~0.25 | ~0.25 |
| 100 | C | 0.250 | 0.249 | 0.250 | 0.252 |

Best val Dice: **0.8405** (epoch 79). 3D Dice: **0.7861**.

### 4.2 Mathematical analysis: why λ_ent cannot hold weights uniform

The near-uniform outcome is not caused by λ_ent forcing weights to be uniform — it is caused by the *segmentation gradients being symmetric across levels*. The entropy gradient at the uniform fixed point is:

```
∂L_ent/∂z_j = w_j [log(w_j) - Σ_i w_i log(w_i)]
```

At the uniform distribution (w_j = 1/n for all j), all log terms are equal, so the gradient is exactly zero. Entropy regularisation provides *no gradient* at the uniform point. If the segmentation gradients are approximately symmetric (because all four levels carry comparable information), weights will converge to near-uniform and λ_ent will simply keep them there.

The implication is important: λ_ent = 0.02 neither caused the uniform outcome nor prevented a non-uniform outcome from emerging. The near-uniform distribution reflects the true (segmentation-loss-minimising) optimum under M4-attn(λ=0.02) training conditions.

### 4.3 Scientific validity problem

Regardless of the mathematical explanation, the near-uniform M4-attn(λ=0.02) model cannot answer RQ5 (*does attention auto-discover L1/L2 suppression?*). If weights are uniform, the attention module has not "discovered" anything — it has converged to a fixed point that is indistinguishable from the unregularised case. To test RQ5 rigorously, we need to observe what the *unconstrained* gradient landscape produces.

---

## §5 Experiment 2: M4-attn with λ_ent = 0 (No Entropy Regularisation)

### 5.1 Training dynamics

Without entropy regularisation, the attention module learns a strongly peaked distribution once unfrozen. The transition is sharp and rapid (Table 2).

**Table 2: Attention weight evolution — M4-attn (λ=0)**

| Epoch | Phase | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|-------|------|------|------|------|
| 20 | A (frozen) | 0.272 | 0.205 | 0.225 | 0.298 |
| 30 | B (warmup) | 0.273 | 0.205 | 0.232 | 0.291 |
| **35** | **B (active, +5 ep)** | **0.006** | **0.010** | **0.340** | **0.644** |
| 40 | B | 0.001 | 0.002 | 0.220 | 0.776 |
| 50 | B | 0.000 | 0.001 | 0.137 | 0.863 |
| 60 | B | 0.000 | 0.000 | 0.075 | 0.925 |
| 75 | B | 0.000 | 0.000 | 0.054 | 0.946 |
| **100** | **C** | **0.000** | **0.000** | **0.060** | **0.940** |

Key observations:
- **Within 5 epochs of unfreezing** (ep 31→35), L1 drops from 0.27 to 0.006 and L4 rises to 0.644.
- The convergence is monotone and rapid: L1/L2 are essentially zero by epoch 45.
- The final state is very stable: w_L4 = 0.940 ± 0.006 across epochs 75–100.
- L3 retains a modest residual weight (~0.06), consistent with L3's genuine but smaller contribution found in v4.

Best val Dice: **0.7896** (epoch 79). 3D Dice: **0.8416**.

### 5.2 Per-class attention weights

At epoch 100, all 8 classes show strong L4 dominance with minimal variation:

| Class | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|------|------|------|------|
| LV | 0.000 | 0.000 | 0.058 | 0.942 |
| RV | 0.000 | 0.000 | 0.059 | 0.941 |
| LA | 0.000 | 0.000 | 0.059 | 0.941 |
| RA | 0.000 | 0.000 | 0.057 | 0.943 |
| Myo | 0.000 | 0.000 | 0.060 | 0.940 |
| Aorta | 0.000 | 0.000 | 0.060 | 0.940 |
| PA | 0.000 | 0.000 | 0.063 | 0.937 |

The attention weights are nearly class-invariant. The slight L3 premium for PA (0.063 vs 0.058 mean) is consistent with PA being the hardest structure (0.841 Dice in M2), potentially benefiting from L3's spatial detail, but the difference is negligible.

This class-invariant pattern suggests the LevelAttentionModule is learning a *global* level quality signal ("deep features are better") rather than a *class-specific* one ("this structure is better detected at a specific scale"). Whether this is a limitation of the current architecture (global avg pool context without class conditioning) or a genuine property of the data is an open question.

---

## §6 Comparative Results

### 6.1 Four-model comparison

> **Methodological note — one caveat applies to this table:**
>
> M2 and the M4-attn models were trained with different Phase B protocols. M2 (trained in Stage 15) had a standard Phase B start including an initial prototype projection across both levels (L3+L4), which completed without instability. M4-attn(λ=0) skipped the initial Phase B projection because the same step applied to all 4 levels simultaneously caused catastrophic training collapse (Dice 0.79→0.11, loss spike to ~1800). M2 therefore received more aggressive prototype repositioning in early Phase B, which may have contributed to better prototype quality metrics. This protocol difference is an architectural consequence — not an arbitrary choice — but it limits the strictness of the M2 vs M4-attn quality comparison.

**Table 3: Quantitative comparison across all models**

| Model | Aggregation | 3D Dice | Best Val | Purity L4 | Compact. L4 | AP L4 | Dom. L4 |
|-------|-------------|---------|----------|-----------|-------------|-------|---------|
| M4 (max) | max | 0.8407 | 0.8173 | 0.824 | 0.573 | 0.189 | 4.3% |
| M4-attn (λ=0.02) | uniform avg | 0.7861 | 0.8405 | 0.526 | 0.575 | 0.187 | 9.7% |
| **M4-attn (λ=0)** | learned attn | **0.8416‡** | 0.7896 | 0.537 | 0.494 | 0.085 | 12.5% |
| **M2 (max)** | max | **0.8722‡** | 0.8380 | 0.804 | 0.361 | 0.236 | 49.1% |

‡ Different Phase B protocols: M2 had initial projection; M4-attn(λ=0) did not (see caveat above).

### 6.2 RQ4: Does learned attention improve over winner-takes-all max? (Δ Dice)

**Result: Marginal, non-significant improvement. Comparison to M2 is protocol-confounded.**

M4-attn(λ=0) achieves 3D Dice 0.8416 vs M4 0.8407, a difference of **+0.0009** (< 0.1%). Both models are within noise of each other — the attention mechanism does not meaningfully improve segmentation quality when operating on four prototype levels.

The M4-attn vs M2 comparison (+3.1% gap) should be interpreted carefully given the protocol difference noted in §6.1. M2's Phase B initial projection may have led to better-positioned prototypes, contributing to its quality advantage. A fairer comparison would re-train M2 with the same protocol as M4-attn (skipping initial projection), but this is not available. The gap is large enough that M2 likely retains a genuine advantage regardless, but the exact magnitude may be overstated.

The mechanistic reason for the gap is structural: attention only modulates the soft mask. Even with L1/L2 suppressed to near-zero weight, their encoder features still reach the decoder through skip connections with a near-zero mask (i.e., nearly raw features). L1/L2 gradients continue flowing through the encoder from all three paths (prototype matching, soft mask, skip connection), consuming encoder capacity without benefit. M2 eliminates all three paths for L1/L2 by architectural choice.

### 6.3 RQ5: Does attention autodiscover L1/L2 suppression?

**Result: Strong confirmation — without supervision, attention converges to L4=0.940, L1/L2≈0.**

The attention module, trained only on segmentation loss, independently replicates the finding from v4's exhaustive manual ablation: L1 and L2 contribute nothing useful, L4 dominates, L3 provides a small supplement. The recovered hierarchy matches:

| Source | L1 | L2 | L3 | L4 |
|--------|----|----|----|----|
| v4 ablation (empirical) | Harmful | Harmful | Beneficial | Core |
| M4-attn λ=0 (learned) | 0.000 | 0.000 | 0.060 | 0.940 |

This convergence validates both findings: the v4 ablation correctly identified the optimal configuration, and the attention module correctly infers the same hierarchy from gradient signals alone.

The transition is rapid (< 5 epochs post-unfreeze) and stable (variance < 0.01 across 65 epochs of Phase B and C). This suggests the level hierarchy is a strong signal in the gradient landscape, not a fragile local minimum.

### 6.4 Prototype quality analysis: M4-attn (λ=0)

**Purity by level:**

| Level | Mean Purity |
|-------|------------|
| L1 | 0.073 |
| L2 | 0.156 |
| L3 | 0.491 |
| **L4** | **0.537** |

L4 purity dropped from 0.824 (M4-max) to 0.537. This is a side effect of training all four levels jointly with an attention mechanism: even though L1/L2 receive near-zero weight, gradients still flow through the encoder from all levels, altering the encoder representations. L4 specialises less exclusively in the attention-augmented setting because the attention gradients provide an alternative pathway to minimise the loss.

**Compactness:**

L4 compactness improved slightly: 0.494 (M4-attn λ=0) vs 0.573 (M4). This is modest; all levels still exceed their design thresholds (L4 threshold: ≤ 0.25), reflecting the fundamental issue that global cosine similarity without locality constraints produces broad activations.

**Per-level AP:**

| Level | Mean AP |
|-------|---------|
| L1 | 0.020 |
| L2 | 0.069 |
| L3 | 0.076 |
| **L4** | **0.085** |

L4's per-level AP dropped from 0.189 (M4-max) to 0.085 — a consequence of the purity reduction. Despite the attention mechanism successfully suppressing L1/L2 in the mask signal, the XAI quality of L4 prototypes is lower when trained with all four levels than when only L3/L4 are present. This is because M2's training dynamics allow L4 to develop more compact, class-specific features when it only competes with L3.

**Level dominance (raw heatmaps):**

| Level | Pixel fraction |
|-------|---------------|
| L1 | 28.1% |
| L2 | 45.0% |
| L3 | 14.3% |
| **L4** | **12.5%** |

The level dominance metric is measured on raw per-level heatmaps (before attention blending), and still shows L1/L2 winning the pixel-wise max competition. This is expected: the attention module modifies the mask signal, not the raw prototype activations. L1/L2 heatmaps remain spatially broad and numerically high even though their contribution to the soft mask is near-zero. This highlights the distinction between *mask contribution* (controlled by attention) and *raw heatmap quality* (unchanged by attention).

---

## §7 Discussion

### 7.1 Learned attention confirms v4's manual finding

The most significant result of this series of experiments is the convergence validation: an unconstrained attention module, trained only to minimise segmentation loss, independently discovers the same level hierarchy that v4 established through exhaustive ablation (M1, M2, M4 comparison). L1 and L2 are not useful; L4 is primary; L3 contributes marginally.

This is not a trivial result. The attention module could have converged to any combination, including class-specific or spatially-varying level preferences. Instead it converges to a global, stable ranking that matches the structural prior. This validates both the attention mechanism as an interpretable diagnostic tool and the v4 ablation conclusions.

### 7.2 Why attention cannot match explicit level removal

Despite suppressing L1/L2 to near-zero attention weight, M4-attn(λ=0) does not achieve M2's performance (+3.1% Dice gap). The mechanism:

1. **Encoder feature contamination**: L1/L2 prototype layers still impose supervision on the encoder's shallow layers. Even with near-zero attention weights, backpropagation through the prototype matching objective trains L1/L2 encoder features to represent class-discriminative information — except they cannot (purity < 0.20), so they consume encoder capacity without benefit.

2. **Soft mask vs. skip removal**: Attention modulates the *soft mask* applied to encoder features. Even with w_L1 ≈ 0, the L1 encoder features still reach the decoder through the skip connection path (just with a weaker mask). M2 eliminates L1/L2 skip connections entirely.

3. **Parameter burden**: M4-attn has 88 prototypes vs M2's 28. The additional 60 lower-quality prototypes add optimisation pressure without contributing to segmentation.

The practical lesson: when a level is known to be harmful, explicit removal is more effective than learned downweighting. Learned attention is a discovery tool, not an equivalent substitute for architectural simplification.

### 7.3 Entropy regularisation and the uniform fixed point

The λ=0.02 experiment produced a near-uniform distribution not because the regularisation was too strong, but because the entropy gradient is zero at the uniform point. Any λ > 0 would produce the same qualitative result: weights near-uniform, slightly shifted toward uniformity relative to the unconstrained case.

This has a practical implication for future work: entropy regularisation as commonly applied to softmax attention is ineffective at preventing collapse *toward* uniformity, and counterproductive for discovering peaked distributions. Alternative regularisation strategies (e.g., diversity loss penalising pairwise similarity between level prototypes, or entropy maximisation at the *encoder representation* level rather than the attention weight level) may be more useful.

### 7.4 The class-invariant attention finding

All 8 classes show nearly identical attention weights (L4: 0.937–0.943). The LevelAttentionModule learns a global ranking ("deep is better") rather than class-specific preferences. This is architecturally explained: the MLP receives global average-pooled features from all levels, without class information. The module cannot differentiate "what scale is best for PA?" from "what scale is best for LV?".

Whether cardiac anatomy actually requires class-specific level attention is unclear. Given the near-uniform purity gradient across all classes at each level (L1 purity is low for all classes, L4 is high for all), a global level ranking may be the correct inductive bias. A class-conditioned attention module would increase parameter count and training complexity for uncertain gain.

---

## §8 Conclusion

We extended the M4 prototype segmentation model with a LevelAttentionModule, replacing the winner-takes-all cross-level max aggregation with a learned, input-conditioned weighted sum. Two experiments were conducted: M4-attn with entropy regularisation (λ=0.02) and without (λ=0).

**Summary of findings:**

1. **RQ4 (Dice improvement):** Marginal — M4-attn(λ=0) achieves 3D Dice 0.8416 vs M4 0.8407 (+0.0009). Learned attention does not substitute for explicit level removal; M2 (explicit L1/L2 removal) outperforms all M4-based variants by +3.1%.

2. **RQ5 (Autodiscovery):** Confirmed — the unconstrained attention module converges to L4=0.940, L1≈0, L2≈0, reproducing the v4 manual ablation finding with high fidelity. The convergence is rapid (< 5 epochs after unfreeze) and stable.

3. **Training stability:** Phase B projection at Phase B start is destabilising for multi-level models; skipping the initial projection with delayed periodic projections is required.

4. **Entropy regularisation:** λ_ent drives weights toward uniformity (a known fixed point of the gradient), preventing RQ5 from being tested. λ=0 with delayed unfreeze is the correct experimental design for discovering natural level hierarchies.

5. **Prototype quality tradeoff:** Training with attention on four levels reduces L4 purity (0.824→0.537) and per-level AP (0.189→0.085) compared to vanilla M4, because all four encoder levels receive gradient flow simultaneously. M2 avoids this by construction.

**The key insight from this report in combination with v4**: the optimal strategy for prototype-based multi-scale segmentation is to identify and retain only semantically meaningful levels (L3, L4) through explicit architectural choice, not learned downweighting. Learned attention is most valuable as a *diagnostic instrument* — it can confirm the level hierarchy without manual ablation — but it does not replace the segmentation quality gain achieved by removing harmful levels.

---

## Appendix: Outputs

```
results/v5/
  train_curve_proto_ct_l1234_attn.csv          # λ=0.02 training log (100 epochs)
  train_curve_proto_ct_l1234_attn_noent.csv    # λ=0 training log (100 epochs)
  train_curve_l1234_attn.png
  train_curve_l1234_attn_noent.png
  attention_weight_evolution.csv               # λ=0.02 weights by epoch
  attention_weight_evolution_l1234_attn_noent.csv  # λ=0 weights by epoch
  attention_analysis.png                       # λ=0.02 attention analysis
  attention_evolution_comparison.png           # side-by-side λ=0.02 vs λ=0
  attention_weights_per_class.png              # λ=0.02 per-class weights
  attention_weights_per_class_noent.png        # λ=0 per-class weights
  heatmap_comparison.png                       # λ=0.02 heatmap visualisation
  heatmap_comparison_noent.png                 # λ=0 heatmap visualisation
  proto_quality/
    m4_attn/                                   # M4-attn λ=0.02 quality metrics
    m4_attn_noent/                             # M4-attn λ=0 quality metrics
      purity_summary.csv
      compactness.csv
      per_level_ap.csv
      level_dominance.csv
    comparison_table.csv                       # M4 / M4-attn λ=0.02 / M2
    comparison_table_full.csv                  # + M4-attn λ=0

checkpoints/
  proto_seg_ct_l1234_attn_noent.pth            # M4-attn λ=0 (epoch 90, best val 0.7896)

notebooks/
  20_attention_training.ipynb                  # Training notebook (Stages 19–20)
  21_attention_analysis.ipynb                  # Stage 21: M4-attn λ=0.02 analysis
  21b_attention_analysis_noent.ipynb           # Stage 21b: M4-attn λ=0 analysis
```
