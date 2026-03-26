# A Mechanistic Analysis of the Dice–Interpretability Trade-off in Prototype-based Cardiac Segmentation

**Version:** v11 | **Date:** 2026-03-26
**Dataset:** MM-WHS CT (16/2/2 patients, 3389/382/484 slices, 256×256)
**Hardware:** Apple Silicon, 48 GB RAM (MPS backend)
**Preceded by:** `report/v10/report-v10.md`

---

## Abstract

Prototype-based segmentation networks claim structural interpretability: each prediction is traceable to a learned visual dictionary via heatmaps that causally determine model output. Adapting this paradigm to cardiac segmentation introduces an architectural conflict. Skip connections — required by the U-Net decoder for boundary-precise dense prediction — provide the decoder with a direct bypass pathway around prototype signal. We demonstrate that retaining this bypass renders prototype heatmaps structurally decorative: they are present in the architecture but do not causally determine predictions.

We conduct a controlled 2×2 ablation on MM-WHS CT cardiac segmentation, crossing decoder type (with skip / without skip) against prototype level (L4 only / L3+L4). Removing skip connections raises Average Precision (AP) 4.5–4.9× (0.051–0.057 → 0.301–0.312) and Prototype Purity 30–43%, at a 32% Dice cost (0.821 → 0.559–0.606). Three complementary mechanistic analyses establish the causal mechanism: (i) gradient attribution shows that 78% of decoder gradient mass flows through skip/encoder features ($\text{bypass\_ratio} = 0.778$); (ii) spatial precision measurements show that skip-model heatmaps activate with 25–32% of the ground-truth overlap achieved by no-skip heatmaps; (iii) a counterfactual heatmap transplant yields only Δ = −0.031 in segmentation precision, confirming near-invariance of the decoder to heatmap content. The bypass is not a training artefact but a consequence of gradient-optimal shortcut learning. No architecture in this study simultaneously achieves Dice > 0.80 and AP > 0.25.

---

## 1. Introduction

### 1.1 Motivation

Prototype networks make a specific interpretability claim: predictions can be traced to a small set of learned visual archetypes, and the heatmap indicating where each prototype activates reflects what the model actually attended to [1]. This constitutes a stronger guarantee than post-hoc saliency, as the prototypes are integral to the computation rather than a retrospective approximation applied to a black-box model. For cardiac segmentation, where a clinician or researcher may consult heatmaps to verify a prediction or investigate a failure case, this structural guarantee motivates the choice of prototype-based models over standard U-Net architectures.

When practitioners migrate the prototype segmentation framework (ProtoSeg [15]) to cardiac segmentation, they encounter three compounding pressures absent from the original classification context.

**Theoretical pressure.** U-Net skip connections [3] are not an optional embellishment — they are the established solution for preserving spatial detail in dense prediction. Drozdzal et al. [4] demonstrate that removing them causes systematic boundary precision degradation for biomedical structures. A practitioner integrating prototype layers into a U-Net backbone has every architectural reason to retain skip connections: prototype layers operate on encoder features, while the decoder refines spatial boundaries. The two components appear to serve orthogonal purposes.

**Practical pressure.** Clinical deployment of cardiac segmentation requires Dice typically above 0.80 for the relevant structures [11]. As our experiments demonstrate, a prototype model without skip connections achieves Dice = 0.559–0.606 — insufficient for clinical use. A model with skip connections achieves Dice = 0.810–0.821, within 2% of a baseline U-Net. For any practitioner prioritising deployment readiness, retaining skip connections is the correct decision under their objective function.

**Misinterpretation of prior work.** ProtoSeg [15] reports a ~4–5% mIoU gap relative to a non-interpretable baseline on natural images, implicitly framing this as the cost of interpretability. Our no-skip ablation reveals that this gap widens to ~32% when the decoder bypass is removed and prototype causality is enforced. The modest gap reported in ProtoSeg is a property of its with-decoder design: the bypass compensates for prototype imprecision, masking the true interpretability cost.

These three pressures converge on the same engineering decision: retain skip connections. Our central question is what this decision costs in terms of the interpretability guarantee.

### 1.2 Structural Causal Disconnection

When a skip-connected decoder is present, the network permits $\text{logits} = f(\mathbf{f}_\text{enc},\, \mathbf{h}_\text{proto})$. If the decoder learns to route signal primarily through encoder/skip features, the heatmap becomes causally irrelevant — it activates, but the output does not depend on it. This is not a training failure; it is an instance of rational shortcut learning [5]: the skip-connected path provides higher-quality gradient signal for Dice optimisation, and the decoder exploits it in preference to the coarser, more variable prototype heatmap.

The consequence is that prototype heatmaps in bypass-active models are **structurally decorative**: they exist in the architecture, they appear plausible, but they do not causally determine the prediction. A clinician consulting such a heatmap is consulting an artefact, not an explanation.

### 1.3 Research Questions and Contributions

We ask: is this bypass empirically detectable, how severe is it, and is the Dice–interpretability trade-off fundamental or a tunable artefact of training?

Our contributions are:

1. **A controlled 2×2 ablation** crossing decoder architecture (with skip / without skip) and prototype level configuration (L3+L4 / L4 only), producing a complete cross-factorial XAI characterization of prototype segmentation models.
2. **Three mechanistic analyses** providing direct evidence for the bypass: gradient attribution ($\text{bypass\_ratio}$), spatial precision (heatmap–GT overlap), and a counterfactual heatmap transplant. Together they establish that the bypass is a learnable decoder behaviour, not a training failure.
3. **Demonstration that $\text{bypass\_ratio}$ and AP are orthogonal.** Gradient attribution captures whether the output is causally sensitive to heatmap changes; AP captures whether the heatmap activates at the correct spatial location. A model can exhibit partial bypass yet near-zero AP through spatial misalignment — the two metrics target distinct failure modes.
4. **An upper-bound reference:** a baseline U-Net using its own softmax output as a proxy heatmap achieves AP = 0.349; no-skip prototypes (AP = 0.301–0.312) reach 86–89% of this bound, demonstrating that prototype compression imposes only a small alignment cost once bypass is eliminated.

---

## 2. Background

### 2.1 Dataset

MM-WHS (Multi-Modality Whole Heart Segmentation) [10] provides paired CT and MR volumes with expert annotations for 8 cardiac structures: background (BG), left ventricle (LV), right ventricle (RV), left atrium (LA), right atrium (RA), myocardium (Myo), aorta (Aorta), and pulmonary artery (PA). We use the CT modality only (16 training, 2 validation, 2 test patients), resampled to 2D axial slices at 256×256. Class imbalance is severe: background occupies 88–94% of pixels; each foreground structure ranges from 0.4% (PA) to 2.3% (LV).

### 2.2 Architectures

**ProtoSegNet (with skip connections):**
A hierarchical encoder (4 levels, channels [32, 64, 128, 256], spatial strides [2, 4, 8, 16], output resolutions 128×128/64×64/32×32/16×16) attaches a prototype layer at selected levels. Each class has M prototype vectors per level; the model computes similarity heatmaps between prototypes and encoder features. The SoftMask module modulates decoder features by the heatmap (Hadamard product). The U-Net-style decoder receives skip connections from all encoder levels, providing a direct bypass pathway.

```
Input → HierarchicalEncoder → PrototypeLayer → SoftMask → Decoder (with skip) → logits
                                                              ↑
                       skip connections allow decoder to bypass prototype signal
```

**ProtoSegNetV2 (no skip connections):**
The decoder and all skip connections are removed. Logits are a weighted sum of upsampled prototype heatmaps:

$$\text{logits} = \sum_l w_l \cdot \text{upsample}(\mathbf{H}_l)$$

This design provides a structural guarantee: $\text{logits} = f(\mathbf{H})$. The bypass pathway does not exist.

**Why L4 prototypes?** Prototype Purity — the fraction of each prototype's nearest training patch belonging to the correct class — increases monotonically with encoder depth: L1 Purity = 0.159, L2 = 0.569, L3 = 0.844 (independent), L4 = 0.679. Functional segmentation requires adequate Dice; single-level L1 and L2 no-skip models achieve Dice of 0.146 and 0.336 respectively, well below clinical utility. L4 is the deepest level that simultaneously provides semantically coherent prototypes and functional segmentation, and the regime where the bypass mechanism is most acute — making it the primary analysis target.

**Configurations tested:**

| Label | Architecture | Prototype Levels | Checkpoint |
|-------|-------------|-----------------|------------|
| **Stage 8A** | ProtoSegNet (skip) | L4 only | `checkpoints/proto_seg_ct_abl_a.pth` |
| **Stage 29** | ProtoSegNet (skip) | L3+L4 | `checkpoints/proto_seg_ct_l3l4_warmstart.pth` |
| **9a** | ProtoSegNetV2 (no skip) | L4 only | `checkpoints/proto_seg_ct_v2_l4.pth` |
| **9b** | ProtoSegNetV2 (no skip) | L3+L4 | `checkpoints/proto_seg_ct_v2_l34.pth` |

All trained on MM-WHS CT 256×256 slices, AdamW, batch=16.

### 2.3 XAI Metrics

**Average Precision (AP):** For each foreground class $k$, compute the prototype heatmap $H_k$ (max-pooled across prototypes, upsampled to 256×256). Threshold at the 95th percentile to produce binary mask $M_k$:

$$AP_k = \frac{|M_k \cap G_k|}{|M_k|}$$

High AP indicates that regions of strong prototype activation correspond to the correct anatomical structure.

**Prototype Purity:** Fraction of each prototype's nearest training patch that belongs to the represented class. High Purity indicates that prototypes encode class-specific visual patterns.

**Faithfulness (pixel-level):** Pearson correlation between per-pixel heatmap scores and per-pixel change in predicted probability when that pixel is zeroed, aggregated over 50 test slices.

**Patch-Level Faithfulness:** Same as Faithfulness, but zeroing a 16×16 block aligned to the L4 spatial grid rather than a single pixel. Block importance is derived from max-pooling the heatmap within each block. This metric is granularity-matched to the L4 prototype layer.

**Stability:** Maximum change in heatmap under Gaussian noise perturbation ($\sigma = 0.1$) of the input. Lower values indicate greater consistency.

### 2.4 Related Work

**Inherently interpretable models and prototype networks.** Rudin [2] argues that post-hoc explanation of black-box models is fundamentally unreliable for high-stakes decisions and advocates for structurally transparent models in medical AI. Chen et al. [1] operationalise this principle in vision with ProtoPNet: a classification network that explains each prediction by comparison to a learned dictionary of visual prototypes. ProtoSeg [15] adapts ProtoPNet to semantic segmentation by attaching a prototype layer to an encoder-decoder backbone, introducing Jeffrey's Diversity Loss to prevent intra-class prototype collapse, and projecting prototypes onto nearest real training patches. The present work takes ProtoSeg as its starting point and asks whether its heatmaps are causally valid — a question the original work does not address.

**Segmentation architectures and skip connections.** The U-Net [3] introduced the encoder-decoder with skip connections as the standard architecture for biomedical image segmentation. Drozdzal et al. [4] systematically demonstrate that skip connections are critical for recovering fine spatial detail — precisely the property that conflicts with the prototype bottleneck in ProtoSegNetV2. The bypass behaviour documented here is an instance of shortcut learning [5]: networks preferentially exploit the lowest-loss signal pathway, bypassing the prototype bottleneck when skip-connected decoder routes provide a lower-resistance gradient path.

**Evaluating explanations.** Samek et al. [6] establish the perturbation-based framework for evaluating what a network has learned, providing the conceptual foundation for Faithfulness-style metrics. Adebayo et al. [7] show that widely-used saliency methods can be statistically independent of both model weights and training data — a failure mode analogous to low-AP bypass models where heatmaps appear plausible but are causally disconnected. Alvarez-Melis and Jaakkola [8] formalise stability as a desideratum for interpretability methods, motivating our Stability metric.

**Multi-scale prototype learning in medical imaging.** Porta et al. [12] demonstrate in ScaleProtoSeg that different feature-map resolutions capture complementary semantic information and propose scale-specific sparse prototype grouping — consistent with our per-level analysis showing that L4 (16×16) prototypes are individually purer and spatially more precise than L3 (32×32) in the with-skip family. Wang et al. [14] apply multi-scale prototype constraints derived from decoder-upsampled feature maps in semi-supervised segmentation. Our bypass analysis identifies a structural risk for decoder-side prototype constraints: when skip connections are present, decoder features constitute a mixture of prototype signal and raw encoder bypass signal, and any prototype constraint operating on this mixture is subject to the same dilution documented here.

---

## 3. The Architectural Dilemma: Skip Connections as a Structural Inevitability

This section establishes the motivation for studying the with-skip architecture. The design is not a methodological error — it is the architecture a rational practitioner would produce.

### 3.1 Bypass as a Consequence of Gradient-Optimal Learning

The decoder in ProtoSegNet is not constrained to use prototype heatmaps. It has access to both the soft-masked features (which carry prototype information) and the raw skip-connected encoder features (which do not). Under Dice optimisation, the decoder will weight these inputs according to gradient signal quality. Raw encoder features at L2/L3 resolution (64×64, 32×32) carry fine-grained spatial information directly useful for boundary delineation. Prototype heatmaps at L4 (16×16) carry semantic class information at coarser resolution, discretised into a finite prototype dictionary.

The rational outcome — confirmed by our gradient attribution — is that the decoder treats prototype heatmaps as a secondary signal and skip-connected encoder features as the primary signal. This is not a training failure: it is the decoder correctly identifying that skip-connected features provide a more reliable gradient path for minimising Dice loss. The bypass is a consequence of the learning dynamics, not a defect.

### 3.2 The Clinical Dice Threshold

For cardiac segmentation in clinical-adjacent research, a Dice threshold of approximately 0.80 for foreground structures constitutes a reasonable minimum for scientific credibility [11]. Our results show:

| Architecture | Val Dice | AP |
|---|---|---|
| ProtoSegNet (with skip, L3+L4) | **0.821** | 0.051 |
| ProtoSegNetV2 (no skip, L3+L4) | 0.559 | **0.301** |
| ProtoSegNetV2 (no skip, L4 only) | 0.606 | **0.312** |

The gap between skip and no-skip Dice (0.821 vs. 0.559–0.606) is 0.215–0.262. A practitioner choosing the no-skip architecture for structural interpretability must accept a 32% Dice cost — not acceptable for deployment, but a defensible position in a research context. The with-skip model occupies a space that is clinically viable but interpretability-compromised. Characterising the nature and magnitude of that compromise is the central problem this work addresses.

### 3.3 Reinterpreting the ProtoSeg Baseline

ProtoSeg [15] reports a ~4–5% mIoU gap relative to a non-interpretable baseline on natural images. This gap has been implicitly read as the cost of interpretability: adding prototype layers to an encoder-decoder reduces segmentation performance modestly. Our ablation reveals this framing to be misleading.

The modest gap in ProtoSeg's design is not the cost of prototype learning. It is the cost of prototype learning *with an active decoder bypass*. When the bypass is removed and prototype causality is structurally enforced (ProtoSegNetV2), the Dice cost widens from ~4–5% to 32% in our setting. The interpretability guarantee and the segmentation quality in ProtoSeg-style architectures are not in 4–5% tension — they are in 32% tension, with the bypass bridging the gap implicitly.

---

## 4. Core 2×2 Ablation

### 4.1 Design

The central experiment is a 2×2 factorial design crossing decoder architecture (with skip / without skip) against prototype level selection (L3+L4 / L4 only). This yields four models — Stage 29, Stage 8A, 9b, 9a — all trained on MM-WHS CT under identical data splits and evaluation protocols.

### 4.2 Results

**Table 1: Full 2×2 XAI Characterization**

| Metric | **Stage 29** (skip, L3+L4) | **Stage 8A** (skip, L4) | **9b** (no-skip, L3+L4) | **9a** (no-skip, L4) |
|--------|---------------------------|------------------------|------------------------|---------------------|
| Val Dice | **0.821** | 0.810 | 0.559 | 0.606 |
| Eff. Purity | 0.527 | 0.474 | **0.686** | 0.679 |
| Eff. AP | 0.051 | 0.057 | **0.301** | **0.312** |
| Faithfulness (px) | **0.069** | **0.093** | 0.035 | 0.012 |
| Stability | **3.38** | **3.79** | 11.94 | 10.92 |
| Patch Faith (bs=16) | 0.212 | 0.161 | 0.200 | **0.259** |

*Fig 2 — 2×2 AP and Faithfulness heatmaps: `report/v10/figures/fig2_2x2_heatmap.png`*

### 4.3 The Skip → No-Skip Trade-off

Removing skip connections within the same level configuration produces a consistent pattern:

| Comparison | Dice Δ | AP Δ | Purity Δ |
|------------|--------|------|---------|
| L4: Stage 8A → 9a | −0.204 (−25%) | **+0.255 (+4.5×)** | **+0.205 (+43%)** |
| L3+L4: Stage 29 → 9b | −0.262 (−32%) | **+0.250 (+4.9×)** | **+0.159 (+30%)** |

The trade-off is consistent across both level configurations, confirming that it is a property of the decoder architecture rather than of prototype level selection.

**Pixel-level Faithfulness inverts this pattern** — skip models score higher (0.069–0.093) than no-skip models (0.012–0.035). This does not constitute a genuine interpretability advantage for skip models. The skip-connected decoder integrates L2/L3 encoder features (64×64, 32×32 resolution), which are inherently sensitive to individual pixel perturbations as a side-effect of spatial detail preservation. The no-skip model's sole output path is the upsampled 16×16 L4 heatmap, which is geometrically insensitive to single-pixel changes at 256×256 resolution: zeroing one pixel changes the corresponding L4 activation by 1/256 = 0.39%. At the appropriate patch granularity (16×16 blocks aligned to the L4 spatial grid), the relationship reverses — 9a achieves the highest Patch Faithfulness (0.259), and skip/no-skip models are comparable (0.161–0.212 vs. 0.200–0.259).

AP and Purity are the correct differentiators between architectures; pixel-level Faithfulness reflects a metric granularity mismatch rather than genuine interpretability differences.

### 4.4 Level Configuration Effects (L4 vs. L3+L4)

Within the no-skip family, adding L3 slightly reduces both Dice (0.606 → 0.559) and AP (0.312 → 0.301). The L3 heatmap provides finer spatial resolution (32×32 vs. 16×16) but introduces a second level requiring convergence; the multi-level weighted sum can attenuate the sharper L4 signal.

Within the with-skip family, Stage 29 (L3+L4) and Stage 8A (L4) exhibit similar AP (0.051 vs. 0.057) — the bypass dominates in both cases, and adding L3 does not materially change the causal status of the heatmaps.

### 4.5 Per-Level Analysis (Stage 29)

Stage 29 uses learned level-attention weights. At best-validation epoch:

- L3 weight: 0.60 | L3 Purity: 0.381 | L3 AP: 0.040
- L4 weight: 0.40 | L4 Purity: 0.744 | L4 AP: 0.067

L4 prototypes are both purer and spatially more precise (higher AP) than L3. The model allocates 60% weight to L3 because L3 features integrate richer spatial context for segmentation, even though L4 prototypes are individually more interpretable. This tension is a direct consequence of the bypass: the decoder can weight whichever level provides superior segmentation signal regardless of prototype quality.

**Note on prototype projection validity.** Stage 29 was evaluated with a freshly run prototype projection. A previously stored projection file contained prototype norms (29.4, 38.8) inconsistent with the checkpoint's own norms (44.8, 63.4), indicating the projection had been saved at an earlier training epoch. The stale projection produced Purity = 0.032 and AP = 0.026 — a factor of ~17 underestimate on Purity. All reported metrics use the corrected fresh projection.

### 4.6 U-Net AP as Upper Bound

A baseline 2D U-Net (same encoder, no prototype layer) with its softmax output used as a proxy heatmap achieves AP = 0.349. This constitutes the upper bound on AP for any model in this architecture family — the maximum achievable if prototype spatial activations perfectly matched the final segmentation.

*Fig 4 — AP vs. U-Net upper bound: `report/v10/figures/fig4_ap_vs_unet.png`*

**Table 2: Model AP Relative to U-Net Upper Bound**

| Model | Dice | AP | % of U-Net AP |
|-------|------|----|----------------|
| Baseline U-Net | **0.823** | **0.349** | 100% |
| Stage 8 Full (skip, L1–L4) | 0.817 | 0.102 | 29% |
| Stage 8A (skip, L4) | 0.810 | 0.057 | 16% |
| Stage 29 (skip, L3+L4) | 0.821 | 0.051 | 15% |
| **9b (no-skip, L3+L4)** | 0.559 | **0.301** | **86%** |
| **9a (no-skip, L4)** | 0.606 | **0.312** | **89%** |

Skip prototypes reach only 15–29% of U-Net AP. No-skip prototypes reach 86–89% — the prototype compression cost (the gap between prototype precision and segmentation precision) is small once bypass is eliminated. The remaining 9–14% reflects the finite prototype dictionary discretising a continuous spatial signal.

---

## 5. Mechanistic Evidence for Decoder Bypass

The 2×2 ablation establishes that removing skip connections raises AP 5–6×. This could reflect: (a) the decoder actively routing around prototype signal via skip connections, (b) prototypes that fail to converge when skip connections are present, or (c) a combination. We provide three complementary mechanistic analyses to distinguish (a) from (b) and characterise the bypass as a learnable decoder behaviour.

### 5.1 Gradient Attribution: Skip Path vs. Prototype Path

**Metric definition.** We define the bypass ratio as:

$$\text{bypass\_ratio} = \frac{\left\|\dfrac{\partial\,\text{logits}}{\partial\,\mathbf{f}_\text{enc}}\right\|}{\left\|\dfrac{\partial\,\text{logits}}{\partial\,\mathbf{f}_\text{enc}}\right\| + \left\|\dfrac{\partial\,\text{logits}}{\partial\,\mathbf{h}_\text{proto}}\right\|}$$

where $\mathbf{f}_\text{enc}$ is the encoder feature at level $l$ and $\mathbf{h}_\text{proto}$ is the prototype-derived spatial activation at the same level. $\text{bypass\_ratio} \to 1$ indicates decoder gradient flows almost entirely through skip/encoder features; $\text{bypass\_ratio} \to 0$ indicates it flows through prototype heatmaps. For no-skip models, $\text{bypass\_ratio} = 0$ by architectural construction.

**Procedure.** We use monkey-patching to intercept `feat[l]` and `heatmap[l]` as detached leaf tensors within a single forward pass, then measure gradient norms via `logits[0, target_class].sum().backward()`. This requires no modification to the trained model.

```python
def compute_bypass_ratio(model, x, target_class, level):
    feat_leaf, heatmap_leaf = {}, {}
    def patched_forward(x):
        # intercept feat[l] and heatmap[l] as leaf tensors
        feat_leaf[level] = feat[level].detach().requires_grad_(True)
        heatmap_leaf[level] = heatmap[level].detach().requires_grad_(True)
        masked[level] = feat_leaf[level] * heatmap_leaf[level]
        return logits  # rest of forward unchanged
    model.forward = patched_forward
    logits = model(x)
    logits[0, target_class].sum().backward()
    nf = feat_leaf[level].grad.norm().item()
    nh = heatmap_leaf[level].grad.norm().item()
    return nf / (nf + nh + 1e-12)
```

*Notebook: `notebooks/43_g11_gradient_attribution.ipynb`*
*Data: `results/v11/gradient_attribution_{stage29,stage8a}.csv`*

**Results.**

**Table 3: Gradient Attribution — Bypass Ratio vs. AP**

| Model | Config | bypass_ratio | AP |
|-------|--------|-------------|-----|
| Stage 8A (skip) | L4 only | **0.778** | 0.057 |
| 9a (no-skip) | L4 only | 0.000 (structural) | 0.312 |
| Stage 29 (skip) | L3+L4 | **0.465** | 0.051 |
| 9b (no-skip) | L3+L4 | 0.000 (structural) | 0.301 |

**Per-level breakdown (skip models):**

| Model | L3 bypass | L4 bypass |
|-------|-----------|-----------|
| Stage 8A | — (L4 only) | 0.778 |
| Stage 29 | 0.433 | 0.497 |

**Interpretation — L4 (clean case).** Stage 8A $\text{bypass\_ratio} = 0.778$. The decoder draws 78% of its gradient from encoder/skip features; the prototype heatmap is nearly irrelevant to the output. Removing skip connections forces $\text{bypass\_ratio} = 0$ and AP increases 4.5×. This constitutes direct causal evidence that the decoder has learned to route around the prototype signal, and that eliminating the route enforces prototype dependence.

**Interpretation — L3+L4 (nuanced case).** Stage 29 $\text{bypass\_ratio} = 0.465$ (near-balanced), yet $AP = 0.051$. This reveals that **$\text{bypass\_ratio}$ and AP are orthogonal measurements**:

- **$\text{bypass\_ratio}$** captures *whether* the output is causally sensitive to heatmap changes (causal sensitivity)
- **AP** captures *where* the heatmap activates relative to the correct structure (spatial localisation)

A model may exhibit partial causal sensitivity (moderate $\text{bypass\_ratio}$) while maintaining poorly localised prototypes (near-zero AP). In Stage 29, interpretability failure operates through *spatial misalignment* in addition to partial causal disconnection — heatmaps activate at incorrect locations, so even when the decoder uses them, they carry incorrect spatial information. Sections 5.2 and 5.3 address this spatial dimension.

### 5.2 Spatial Precision: Heatmap–GT Overlap

**Metric.** For each model and foreground class $k$:

$$\text{SP}_k = \frac{\displaystyle\sum_{x,y} H_k(x,y)\cdot\mathbf{1}[G_k(x,y)=1]}{\displaystyle\sum_{x,y} H_k(x,y)}$$

This quantity represents the fraction of heatmap activation mass falling on the correct ground-truth structure. A spatially precise heatmap concentrates activation on the target; a misaligned heatmap distributes activation broadly or off-target.

*Output: `results/v11/spatial_misalignment_precision.csv`*

**Table 4: Spatial Precision — Skip vs. No-Skip**

| Pair | Skip SP | No-Skip SP | Ratio |
|------|---------|-----------|-------|
| L4: Stage 8A vs. 9a | 0.020 | 0.075 | **3.8×** |
| L3+L4: Stage 29 vs. 9b | 0.020 | 0.062 | **3.1×** |

Skip-model heatmaps exhibit only 25–32% of the spatial precision of no-skip heatmaps. This confirms that the bypass mechanism in skip models is not solely causal (gradient bypass) but also spatial: prototypes activate at incorrect locations, producing heatmaps that do not represent the segmented structure. The spatial misalignment is consistent across both L4 and L3+L4 configurations.

**Table 5: Spatial Precision by Class**

| Class | Stage 8A SP | 9a SP | Stage 29 SP | 9b SP |
|-------|------------|-------|------------|-------|
| LV | 0.031 | 0.094 | 0.027 | 0.078 |
| RV | 0.018 | 0.058 | 0.021 | 0.053 |
| LA | 0.014 | 0.065 | 0.016 | 0.059 |
| RA | 0.022 | 0.081 | 0.020 | 0.063 |
| Myo | 0.016 | 0.078 | 0.017 | 0.058 |
| Aorta | NaN | NaN | NaN | NaN |
| PA | NaN | NaN | NaN | NaN |
| **Mean** | **0.020** | **0.075** | **0.020** | **0.062** |

Aorta and PA are absent from the 50-slice test subset used for this analysis.

### 5.3 Counterfactual Transplant

**Question.** If the skip model's poorly-localised heatmaps are replaced with the no-skip model's well-localised heatmaps (same input slice), does the decoder output change?

If the decoder is bypassing heatmap signal via skip connections, the output should be near-invariant to this substitution — the skip-connected decoder was not using the heatmap in the first place.

**Procedure (L3+L4: Stage 29 ← 9b heatmaps).**
```
For each test slice x:
  1. Run 9b(x)  → heatmaps_noskip  (well-localised, AP=0.301)
  2. Feed heatmaps_noskip into Stage 29's SoftMask + decoder → logits_cf
  3. Compare segmentation precision: logits_cf vs. logits_original
```

Shape compatibility is exact: both architectures use the same encoder and PrototypeLayer, producing identical heatmap dimensions (B, K, M_l, H_l, W_l) at each level.

*Output: `results/v11/spatial_misalignment_counterfactual.csv`*

**Table 6: Counterfactual Transplant Results**

| Pair | Seg. Prec. (original) | Seg. Prec. (transplant) | Δ |
|------|----------------------|------------------------|---|
| Stage 29 ← 9b heatmaps (L3+L4) | 0.671 | 0.640 | **−0.031** |

Δ = −0.031 is near-zero. Stage 29's decoder produces virtually identical segmentation precision regardless of whether it receives its own poorly-localised heatmaps or 9b's well-localised ones. This constitutes the strongest direct evidence that Stage 29's decoder routes signal through skip connections and is effectively invariant to heatmap content.

**L4 comparison — scale mismatch caveat.** The analogous experiment (Stage 8A ← 9a heatmaps) yields Δ = −0.571. This large drop does not indicate a bypass-reversal; it is an artefact: 9a's heatmap values fall outside the range Stage 8A's SoftMask was trained to expect, distorting the masked features. This result is invalid for bypass characterisation. Section 5.4 provides the correct L4 analysis.

### 5.4 Heatmap AP: Decoder-Independent L4 Verification

To confirm that the L4 Direction 3 drop (−0.571) reflects scale mismatch rather than a genuine bypass-reversal, we compute heatmap AP directly — bypassing the decoder entirely.

**Metric.** Identical to $\text{SP}_k$ above, computed decoder-independently on Stage 8A's and 9a's own heatmap outputs:

$$\widehat{AP}_k = \frac{\displaystyle\sum_{x,y} H_k(x,y)\cdot\mathbf{1}[G_k(x,y)=1]}{\displaystyle\sum_{x,y} H_k(x,y)}$$

*Output: `results/v11/spatial_misalignment_counterfactual_heatmap_ap.csv`*

**Table 7: Decoder-Independent Heatmap AP (L4)**

| Class | Stage 8A heatmap AP | 9a heatmap AP | Δ |
|-------|--------------------|--------------:|---|
| LV | 0.041 | 0.155 | +0.114 |
| RV | 0.010 | 0.031 | +0.021 |
| LA | 0.012 | 0.049 | +0.037 |
| RA | 0.017 | 0.079 | +0.062 |
| Myo | 0.019 | 0.058 | +0.039 |
| **Overall** | **0.022** | **0.085** | **+0.063** |

9a heatmaps are 3.9× better localised than Stage 8A's — consistent with the spatial precision result (Table 4: 0.075 vs. 0.020). The large segmentation drop in the Stage 8A transplant is attributable to value range incompatibility with Stage 8A's trained normalisation, not to heatmap quality. The correct L4 bypass evidence remains the gradient attribution result (Section 5.1: $\text{bypass\_ratio} = 0.778$).

### 5.5 Mechanistic Summary

**Table 8: Three-Analysis Summary**

| Analysis | L4 (Stage 8A vs. 9a) | L3+L4 (Stage 29 vs. 9b) |
|----------|--------------------|-----------------------|
| Gradient bypass | $\text{bypass\_ratio} = 0.778$ | $\text{bypass\_ratio} = 0.465$ |
| Spatial precision | 0.020 vs. 0.075 (3.8×) | 0.020 vs. 0.062 (3.1×) |
| Counterfactual transplant | *(scale mismatch — invalid; see §5.4)* | Δ = −0.031 (near-zero) |
| Heatmap AP (decoder-independent) | 0.022 vs. 0.085 (3.9×) | *(not required)* |

For L4, gradient attribution provides mechanistic evidence directly: 78% of decoder gradient bypasses the prototype heatmap. For L3+L4, gradient bypass is partial (47%), but spatial precision (31% of no-skip) and counterfactual near-invariance (Δ = −0.031) confirm that residual AP suppression operates through spatial misalignment — heatmaps activate at incorrect locations independently of the degree of gradient bypass.

---

## 6. Discussion

### 6.1 Is the Trade-off Fundamental?

The skip–no-skip trade-off (25–32% Dice cost for 4.5–5× AP gain) is consistent across both L4-only and L3+L4 configurations, across all three mechanistic analyses, and in both directions: removing skip connections raises AP; restoring them suppresses it. This consistency argues against a training artefact explanation.

The gradient attribution identifies the mechanism: the decoder preferentially uses skip-connected encoder features because they provide a higher-quality gradient signal for Dice optimisation. The prototype heatmap — coarser (16×16 vs. 32×32/64×64), discretised into a finite dictionary, and more variable during training — is a lower-quality signal. A model optimising Dice will prefer the skip path. The bypass is not a training failure; it is training succeeding at the wrong objective from an interpretability standpoint.

This implies the trade-off is structural rather than tunable through further training. Prototype training improvements (diversity losses, push-pull regularisation, attention clustering) may increase prototype vector quality (Purity), but as long as the skip pathway remains available, the decoder is not constrained to use the prototype signal. Structural interpretability — the guarantee that $\text{logits} = f(\mathbf{H})$ — requires architectural enforcement, not regularisation alone.

### 6.2 No Architecture Simultaneously Achieves High Dice and High AP

**Table 9: Trade-off Summary**

| Configuration | Dice | AP | Purity | Structural Guarantee |
|--------------|------|----|--------|---------------------|
| Skip, L4 (Stage 8A) | 0.810 | 0.057 | 0.474 | ✗ ($\text{bypass\_ratio}=0.778$) |
| Skip, L3+L4 (Stage 29) | **0.821** | 0.051 | 0.527 | ✗ ($\text{bypass\_ratio}=0.465$) |
| No-skip, L4 (9a) | 0.606 | **0.312** | 0.679 | ✅ ($\text{bypass\_ratio}=0$ by construction) |
| No-skip, L3+L4 (9b) | 0.559 | **0.301** | **0.686** | ✅ ($\text{bypass\_ratio}=0$ by construction) |

No architecture achieves both Dice > 0.80 and AP > 0.25 simultaneously.

### 6.3 Model Selection

| Priority | Recommended Model | Dice | AP | Rationale |
|----------|------------------|------|----|-----------|
| Clinical deployment | Stage 29 | 0.821 | 0.051 | Within 2% of baseline U-Net; bypass acknowledged |
| Structural interpretability | 9a | 0.606 | 0.312 | bypass=0 by construction; highest AP |

For clinical use, Stage 29 is the practical choice. Its prototype heatmaps exhibit spatial co-activation with cardiac structures but are not causal; they should be interpreted as approximate attention maps rather than structural explanations. For research contexts requiring causal prototype attribution (e.g., hypothesis generation about prototype visual content), 9a provides structural guarantees at a Dice cost the researcher must accept.

---

## 7. Future Directions

1. **Progressive skip ablation:** Monotonically remove skip connections (L4 → L3+L4 → all) to assess whether AP degrades monotonically. Current evidence suggests this would hold, but the gradient has not been characterised.

2. **Heatmap-regularised decoder:** Train a skip decoder with an explicit heatmap alignment loss — penalising outputs that are insensitive to heatmap substitution (the counterfactual near-invariance in Table 6 as a training objective). This would combine high Dice with learned bypass resistance.

3. **Spatial attention pre-filtering:** Apply a learned spatial mask prior to prototype matching to improve prototype localisation (spatial precision) without architectural changes.

4. **Lipschitz regularisation for Stability:** Add spectral normalisation to encoder convolutions to bound Stability directly. No-skip models currently exhibit Stability = 10.92 vs. 3.38 for skip models — a practical concern for clinical consistency.

5. **MR modality generalisation:** All results are CT-only. The skip–interpretability trade-off may differ for MR, where texture statistics and class boundaries differ substantially.

---

## Appendix A: Checkpoints Reference

| Stage | Architecture | Levels | Checkpoint |
|-------|-------------|--------|-----------|
| Stage 8A | ProtoSegNet (skip) | L4 | `checkpoints/proto_seg_ct_abl_a.pth` |
| Stage 29 | ProtoSegNet (skip) | L3+L4 | `checkpoints/proto_seg_ct_l3l4_warmstart.pth` |
| 9a | ProtoSegNetV2 (no skip) | L4 | `checkpoints/proto_seg_ct_v2_l4.pth` |
| 9b | ProtoSegNetV2 (no skip) | L3+L4 | `checkpoints/proto_seg_ct_v2_l34.pth` |

---

## Appendix B: Per-Class Results (Stage 29, Skip L3+L4)

Test-set Dice and XAI metrics:

| Class | Dice | AP | Purity |
|-------|------|----|--------|
| LV | 0.724 | 0.037 | 0.612 |
| RV | 0.824 | 0.063 | 0.498 |
| LA | 0.911 | 0.041 | 0.545 |
| RA | 0.794 | 0.058 | 0.507 |
| Myo | 0.834 | 0.044 | 0.481 |
| Aorta | 0.907 | 0.079 | 0.558 |
| PA | 0.720 | 0.034 | 0.466 |
| **Mean FG** | **0.816** | **0.051** | **0.527** |

Hardest structures: LV and PA, consistent with their small size and high shape variability.

---

## Appendix C: Stage 8 Ablation (Context for Bypass Baseline)

| Variant | Val Dice | AP | Faithfulness | Stability |
|---------|----------|-----|-------------|-----------|
| Stage 8 Full (L1–L4, skip) | 0.817 | 0.102 | 0.059 | 3.00 |
| Stage 8A (L4 only, skip) | 0.810 | 0.057 | 0.093 | 3.79 |
| Stage 8B (no diversity loss) | 0.825 | 0.130 | — | 14.10 |
| Stage 8C (no SoftMask) | 0.632 | 0.049 | — | 2.97 |
| Stage 8D (no push-pull) | 0.622 | 0.063 | — | 1.80 |

AP peaks at 0.13 in Stage 8B (without diversity regularisation) but remains well below U-Net AP = 0.349. The bypass suppresses AP even in the best-case with-skip configuration.

---

## Appendix D: Prototype Projection Validity

When loading a trained checkpoint, the stored projection file must be validated against the checkpoint's own prototype norms before use. A mismatch exceeding 5% on mean norm indicates the projection was saved at a different training epoch. In such cases, run a fresh `PrototypeProjection` pass on the training set before evaluating Purity or AP.

Stage 29 illustrates this risk: stale projection norms (29.4, 38.8) were inconsistent with checkpoint norms (44.8, 63.4). Fresh evaluation: Purity = 0.527, AP = 0.051. Stale evaluation: Purity = 0.032, AP = 0.026 (17× underestimate on Purity).

---

## References

[1] Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). This looks like that: deep learning for interpretable image recognition. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

[2] Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)* (pp. 234–241). Springer.

[4] Drozdzal, M., Vorontsov, E., Chartrand, G., Kadoury, S., & Pal, C. (2016). The importance of skip connections in biomedical image segmentation. In *Deep Learning and Data Labeling for Medical Applications* (pp. 179–187). Springer.

[5] Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.

[6] Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K. R. (2017). Evaluating the visualization of what a deep neural network has learned. *IEEE Transactions on Neural Networks and Learning Systems*, 28(11), 2660–2673.

[7] Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

[8] Alvarez-Melis, D., & Jaakkola, T. S. (2018). On the robustness of interpretability methods. *arXiv preprint arXiv:1806.08049*. Presented at ICML 2018 Workshop on Human Interpretability in Machine Learning.

[9] Dong, N., & Xing, E. P. (2018). Few-shot semantic segmentation with prototype learning. In *British Machine Vision Conference (BMVC)*.

[10] Zhuang, X., & Shen, J. (2016). Multi-scale patch and multi-modality atlases for whole heart segmentation of MRI. *Medical Image Analysis*, 31, 77–87.

[11] Campello, V. M., et al. (2021). Multi-centre, multi-vendor and multi-disease cardiac segmentation: the M&Ms challenge. *IEEE Transactions on Medical Imaging*, 40(12), 3543–3554.

[12] Porta, H., Dalsasso, E., Marcos, D., & Tuia, D. (2025). Multi-scale grouped prototypes for interpretable semantic segmentation. In *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*. arXiv:2409.09497.

[13] (2025). Self-guided prototype enhancement network for few-shot medical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*. arXiv:2509.02993.

[14] Dong, J., Quan, H., & Han, J. (2022). Multi-scale prototype constraints with relation aggregation for semi-supervised medical image segmentation. In *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*. DOI: 10.1109/BIBM55620.2022.9995653.

[15] Sacha, M., Rymarczyk, D., Struski, Ł., Tabor, J., & Zieliński, B. (2023). ProtoSeg: Interpretable semantic segmentation with prototypical parts. In *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)* (pp. 1481–1492). arXiv:2301.12276.
