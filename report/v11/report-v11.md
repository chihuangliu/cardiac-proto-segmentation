# When Skip Connections Bypass Prototypes: A Mechanistic Analysis of the Dice–Interpretability Trade-off in Cardiac Segmentation

**Version:** v11 | **Date:** 2026-03-26
**Dataset:** MM-WHS CT (16/2/2 patients, 3389/382/484 slices, 256×256)
**Hardware:** Apple Silicon, 48 GB RAM (MPS backend)
**Preceded by:** `report/v10/report-v10.md`

---

## Abstract

Prototype-based segmentation networks claim structural interpretability: each prediction is traceable to a learned visual dictionary via heatmaps that causally determine model output. Adapting this paradigm to cardiac segmentation introduces an architectural conflict. Skip connections — required by the U-Net decoder for boundary-precise dense prediction — provide the decoder with a direct bypass pathway around prototype signal. We demonstrate that retaining this bypass renders prototype heatmaps structurally decorative: they are present in the architecture but do not causally determine predictions.

We conduct a controlled 2×2 ablation on MM-WHS CT cardiac segmentation, crossing decoder type (with skip / without skip) against prototype level (16×16 only / 32×32+16×16). Removing skip connections raises Average Precision (AP) 4.5–4.9× (0.051–0.057 → 0.301–0.312) and Prototype Purity 30–43%, at a 32% Dice cost (0.821 → 0.559–0.606). Three complementary mechanistic analyses establish the causal mechanism: (i) gradient attribution shows that 78% of decoder gradient mass flows through skip/encoder features ($\text{bypass\_ratio} = 0.778$); (ii) spatial precision measurements show that skip-model heatmaps activate with 25–32% of the ground-truth overlap achieved by no-skip heatmaps; (iii) a GT-guided counterfactual shows that skip-32+16 prototype heatmaps operate at only 20% of their achievable localization quality within their own feature space (AP = 0.078 vs. GT-optimal AP = 0.389), while skip-16 retains 89% (AP = 0.551 vs. 0.623) — establishing that spatial misalignment in skip-32+16 is not a prototype training failure but a consequence of bypass suppressing the incentive to localise. The bypass is not a training artefact but a consequence of gradient-optimal shortcut learning. No architecture in this study simultaneously achieves Dice > 0.80 and AP > 0.25.

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

1.2 establishes that skip-connected decoders should preferentially route gradient through encoder features, rendering prototype heatmaps causally irrelevant. This theoretical argument, however, does not tell us how severe the bypass is in practice, nor whether heatmap spatial misalignment and causal disconnection are the same phenomenon or distinct ones. We therefore ask: how can the bypass be empirically characterised across architecture variants, and do gradient-based causal measures and spatial precision measures capture the same failure mode or different ones?

Our contributions are:

1. **A controlled 2×2 ablation** crossing decoder type (with skip / without skip) and prototype level configuration (32×32+16×16 / 16×16 only), together with three complementary mechanistic analyses — gradient attribution ($\text{bypass\_ratio}$), spatial precision (heatmap–GT overlap), and a GT-guided counterfactual heatmap analysis — establishing that the bypass is a learnable decoder behaviour, not a training failure.
2. **A metric decomposition** showing that two interpretability metrics target distinct failure modes. We define the bypass ratio as the fraction of decoder gradient flowing through skip/encoder features rather than through prototype heatmaps — a measure of causal sensitivity. Average Precision (AP) measures spatial alignment: whether prototype activations localise the correct anatomical structure. A model can exhibit partial bypass yet near-zero AP through spatial misalignment, and the two metrics are empirically orthogonal.
3. **An empirical bound on the interpretability cost of prototype compression.** A baseline U-Net using its own softmax output as a proxy heatmap achieves AP = 0.349; no-skip prototypes reach 86–89% of this bound (AP = 0.301–0.312). Once the bypass is eliminated, prototype compression imposes only a small alignment cost — the dominant cost is the bypass, not the prototype bottleneck.

---

## 2. Background

### 2.1 Related Work

**Inherently interpretable models and prototype networks.** Rudin [2] argues that post-hoc explanation of black-box models is fundamentally unreliable for high-stakes decisions and advocates for structurally transparent models in medical AI. Chen et al. [1] operationalise this principle in vision with ProtoPNet: a classification network that explains each prediction by comparison to a learned dictionary of visual prototypes. ProtoSeg [15] adapts ProtoPNet to semantic segmentation by attaching a prototype layer to an encoder-decoder backbone, introducing Jeffrey's Diversity Loss to prevent intra-class prototype collapse, and projecting prototypes onto nearest real training patches. The present work takes ProtoSeg as its starting point and asks whether its heatmaps are causally valid — a question the original work does not address.

**Segmentation architectures and skip connections.** The U-Net [3] introduced the encoder-decoder with skip connections as the standard architecture for biomedical image segmentation. Drozdzal et al. [4] systematically demonstrate that skip connections are critical for recovering fine spatial detail — precisely the property that creates tension when a prototype bottleneck is introduced into the decoder pathway. When both pathways are available, shortcut learning [5] predicts that networks will preferentially exploit the lowest-loss signal route: the skip-connected encoder path, rather than the coarser prototype heatmap.

**Evaluating explanations.** Samek et al. [6] establish the perturbation-based framework for evaluating what a network has learned, providing the conceptual foundation for Faithfulness-style metrics. Adebayo et al. [7] show that widely-used saliency methods can be statistically independent of both model weights and training data — a failure mode analogous to low-AP bypass models where heatmaps appear plausible but are causally disconnected. Alvarez-Melis and Jaakkola [8] formalise stability as a desideratum for interpretability methods, motivating our Stability metric.

**Multi-scale prototype learning in medical imaging.** Porta et al. [12] demonstrate in ScaleProtoSeg that different feature-map resolutions capture complementary semantic information, motivating prototype attachment at multiple encoder levels. Wang et al. [14] apply prototype constraints directly to decoder-upsampled features in semi-supervised segmentation. Neither work examines what happens to these constraints when skip connections allow the decoder to bypass prototype signal: if decoder features mix prototype output with raw encoder bypass, a prototype constraint operating on that mixture may be enforcing alignment on a signal the decoder does not actually use for prediction.

**Mechanistic analysis of gradient routing and causal disconnection.** The interpretability analyses in this paper are motivated by a known failure mode in XAI: Adebayo et al. [7] demonstrate that widely-used saliency maps can be statistically independent of both model weights and training labels, meaning a visually plausible heatmap is not evidence of causal influence. The correct test is whether the decoder's gradient mass flows through the explanation pathway. Geirhos et al. [5] establish the theoretical framing: under shortcut learning, networks preferentially exploit the lowest-resistance gradient path available — in a skip-connected architecture, this predicts that the decoder will route signal through encoder/skip features rather than through prototype heatmaps, not as a failure but as a rational optimisation choice. This motivates the bypass ratio as a gradient-based causal measure rather than a spatial or visual one.

For spatial alignment, Huang et al. [16] provide the most direct precedent: they introduce quantitative consistency and stability metrics specifically for part-prototype networks, arguing that visual inspection of prototype heatmaps is insufficient to confirm localization quality, and that per-class spatial overlap with ground truth is required. Their framework directly supports the Average Precision and Spatial Precision metrics used in §3.6, extending the perturbation-based evaluation logic of Samek et al. [6] to the prototype-specific context.

The GT-guided counterfactual analysis builds on two complementary foundations. Bau et al. [17] establish in Network Dissection that a hidden unit's semantic alignment can be measured by computing the optimal overlap achievable using ground-truth region masks as a reference — separating what the feature space can represent from what training actually induces. This is the conceptual basis for our AP$_\text{GT}$ ceiling: we ask whether the encoder feature space has the capacity to localise cardiac structures (it does: AP$_\text{GT}$ = 0.389–0.623), and whether training has exploited that capacity (it has not for skip-32+16: AP$_\text{learned}$ = 0.078). Goyal et al. [18] formalise counterfactual explanation as the question of how prediction changes when a specified region's features are substituted; our GT-guided procedure is a localization-constrained variant, substituting learned prototype similarity maps with GT-oracle maps to identify the gap between current alignment and achievable alignment within the same model.

---

## 3. Methods

### 3.1 Dataset

MM-WHS (Multi-Modality Whole Heart Segmentation) [10] provides paired CT and MR volumes with expert annotations for 8 cardiac structures: background (BG), left ventricle (LV), right ventricle (RV), left atrium (LA), right atrium (RA), myocardium (Myo), aorta (Aorta), and pulmonary artery (PA). We use the CT modality only, split into 16 training, 2 validation, and 2 test patients, yielding 3,389 / 382 / 484 axial slices respectively after resampling to 256×256. Class imbalance is severe: background occupies 88–94% of pixels; each foreground structure ranges from 0.4% (PA) to 2.3% (LV).

### 3.2 Experimental Design

The central experiment is a 2×2 factorial design crossing decoder architecture (with skip / without skip) against prototype level selection (32×32+16×16 / 16×16 only). This yields four models, all trained on MM-WHS CT under identical data splits and evaluation protocols:

| Model | Architecture | Prototype Levels | Checkpoint |
|-------|-------------|-----------------|------------|
| **skip-16** | Skip-connected | 16×16 only | `checkpoints/proto_seg_ct_abl_a.pth` |
| **skip-32+16** | Skip-connected | 32×32+16×16 | `checkpoints/proto_seg_ct_l3l4_warmstart.pth` |
| **noskip-16** | No-skip (heatmap-only) | 16×16 only | `checkpoints/proto_seg_ct_v2_l4.pth` |
| **noskip-32+16** | No-skip (heatmap-only) | 32×32+16×16 | `checkpoints/proto_seg_ct_v2_l34.pth` |

All models trained on MM-WHS CT 256×256 slices, AdamW, batch=16.

### 3.3 Architectures

**Skip-connected architecture:**
A hierarchical encoder (4 levels, channels [32, 64, 128, 256], spatial strides [2, 4, 8, 16], output resolutions 128×128/64×64/32×32/16×16) attaches a prototype layer at selected levels. Each class has M prototype vectors per level; the model computes similarity heatmaps between prototypes and encoder features. The SoftMask module modulates decoder features by the heatmap (Hadamard product). The U-Net-style decoder receives skip connections from all encoder levels, providing a direct bypass pathway.

```
Input → HierarchicalEncoder → PrototypeLayer → SoftMask → Decoder (with skip) → logits
                                                              ↑
                       skip connections allow decoder to bypass prototype signal
```

**No-skip (heatmap-only) architecture:**
The decoder and all skip connections are removed. Logits are a uniform average of upsampled prototype heatmaps across the selected levels $\mathcal{L}$:

$$\text{logits} = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \mathrm{upsample}_{256}\!\left(\mathbf{H}^{(l)}\right)$$

This design provides a structural guarantee: $\text{logits} = f(\mathbf{H})$. The bypass pathway does not exist.

**Prototype Layer.** At encoder level $l$ with output feature map $\mathbf{F}^{(l)} \in \mathbb{R}^{C_l \times H_l \times W_l}$, the prototype dictionary contains $K \times M_l$ prototype vectors $\{\mathbf{p}_{k,m}^{(l)}\}_{k=1,\,m=1}^{K,\,M_l} \subset \mathbb{R}^{C_l}$, one set per class ($K=8$ including background). Similarity between spatial location $(x,y)$ and prototype $(k,m)$ is computed as:

$$s_{k,m}^{(l)}(x,y) = \exp\!\left(-\left\|\mathbf{F}^{(l)}(x,y) - \mathbf{p}_{k,m}^{(l)}\right\|^2\right)$$

The per-class heatmap aggregates over prototypes by max-pooling:

$$H_k^{(l)}(x,y) = \max_{m=1,\dots,M_l} s_{k,m}^{(l)}(x,y)$$

yielding $\mathbf{H}^{(l)} \in \mathbb{R}^{K \times H_l \times W_l}$. The SoftMask module modulates encoder features element-wise:

$$\tilde{\mathbf{F}}^{(l)}(x,y) = \mathbf{F}^{(l)}(x,y) \odot \mathbf{H}^{(l)}(x,y)$$

where $\mathbf{H}^{(l)}$ is broadcast over the channel dimension. For the no-skip architecture, the final logits aggregate heatmaps across selected levels:

$$\text{logits}(x,y) = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \mathrm{upsample}_{256}\!\left(\mathbf{H}^{(l)}(x,y)\right)$$

Prototypes are trained jointly with the encoder via $\mathcal{L} = 0.5\,\mathcal{L}_\text{Dice} + 0.5\,\mathcal{L}_\text{CE} + \lambda_\text{div}\,\mathcal{L}_\text{div}$, where $\mathcal{L}_\text{div}$ is Jeffrey's Divergence Loss penalising within-class prototype collapse. After training, each $\mathbf{p}_{k,m}^{(l)}$ is projected to the nearest real training patch feature from class $k$, grounding prototypes in observed data.

**Level selection.** The spatial resolution of encoder level $l$ controls the semantic specificity of prototypes. We characterise this via Prototype Purity — the class-agreement fraction at a prototype's nearest training patch — measured for single-level no-skip models:

| Level | Spatial res. | Prototypes per class ($M_l$) | Purity | Val Dice |
|-------|-------------|--------------------------|--------|----------------------------------|
| 1 | 128×128 | 4 | 0.159 | 0.146 |
| 2 | 64×64 | 3 | 0.569 | 0.336 |
| 3 | 32×32 | 2 | 0.844 | 0.554 |
| 4 | 16×16 | 2 | 0.679 | 0.606 |

Levels 1 and 2 provide insufficient segmentation quality (Dice < 0.35), ruling them out for functional use. Level 4 (16×16) achieves the best Dice among deployable single-level models (0.606) and, while its Purity (0.679) is slightly below level 3 (0.844), it sits at the deepest encoding where global semantic identity is most compressed — the regime where bypass is most acute and interpretability analysis is most informative.

Level 3 (32×32) has the highest standalone Purity and captures finer-grained spatial structure than 16×16. As a second level in a multi-scale configuration, it provides complementary coverage: 16×16 encodes whole-structure identity, 32×32 encodes boundary-adjacent texture. The 32×32+16×16 configuration tests whether multi-scale prototypes improve or complicate interpretability relative to the single 16×16 baseline.

### 3.4 Training

All four models were trained on MM-WHS CT 256×256 axial slices using AdamW (learning rate $3\times10^{-4}$, weight decay $10^{-5}$, batch size 16) with cosine annealing over a maximum of 100 epochs. Early stopping was applied with patience 15 on validation Dice.

**Loss curriculum.** Epochs 1–20 optimised only the segmentation loss ($0.5\,\mathcal{L}_\text{Dice} + 0.5\,\mathcal{L}_\text{CE}$), allowing the encoder and prototype vectors to initialise before diversity is enforced. From epoch 21 onward, Jeffrey's Divergence Loss ($\lambda_\text{div}=0.01$) was added and prototype projection was applied every 10 epochs: each $\mathbf{p}_{k,m}^{(l)}$ was replaced by its nearest class-$k$ training patch feature.

**Initialisation for skip-32+16.** Training skip-32+16 from random initialisation caused prototype collapse at Level 3. It was instead initialised from a fully converged skip-16 checkpoint, then fine-tuned for 100 epochs with both Level 3 and Level 4 prototype layers active (same loss curriculum as above).

No data augmentation was applied.

### 3.5 Evaluation Metrics

**Average Precision (AP):** For each foreground class $k$, compute the prototype heatmap $H_k$ (max-pooled across prototypes, upsampled to 256×256). Threshold at the 95th percentile to produce binary mask $M_k$:

$$AP_k = \frac{|M_k \cap G_k|}{|M_k|}$$

High AP indicates that regions of strong prototype activation correspond to the correct anatomical structure.

**Prototype Purity:** Fraction of each prototype's nearest training patch that belongs to the represented class. High Purity indicates that prototypes encode class-specific visual patterns. Formally, let $\hat{y}(\mathbf{p}_{k,m}^{(l)})$ denote the ground-truth label at the spatial location of prototype $\mathbf{p}_{k,m}^{(l)}$'s nearest training patch; then:

$$\text{Purity}^{(l)} = \frac{1}{K \cdot M_l} \sum_{k=1}^{K} \sum_{m=1}^{M_l} \mathbf{1}\!\left[\hat{y}\!\left(\mathbf{p}_{k,m}^{(l)}\right) = k\right]$$

For multi-level models, AP and Purity are aggregated across levels using pixel-dominance fractions $\phi_l$ — the fraction of test pixels at which level $l$'s heatmap achieves the highest activation under max-aggregation:

$$\text{Eff. AP} = \sum_{l \in \mathcal{L}} \phi_l \cdot \overline{AP}^{(l)}, \qquad \text{Eff. Purity} = \sum_{l \in \mathcal{L}} \phi_l \cdot \text{Purity}^{(l)}$$

where $\overline{AP}^{(l)}$ is the mean AP over foreground classes at level $l$. For single-level models, Eff. AP and Eff. Purity reduce to the level's own values.

### 3.6 Mechanistic Analysis Procedures

**Gradient Attribution (bypass ratio).** We define the bypass ratio as:

$$\text{bypass\_ratio} = \frac{\left\|\dfrac{\partial\,\text{logits}}{\partial\,\mathbf{f}_\text{enc}}\right\|}{\left\|\dfrac{\partial\,\text{logits}}{\partial\,\mathbf{f}_\text{enc}}\right\| + \left\|\dfrac{\partial\,\text{logits}}{\partial\,\mathbf{h}_\text{proto}}\right\|}$$

where $\mathbf{f}_\text{enc}$ is the encoder feature at level $l$ and $\mathbf{h}_\text{proto}$ is the prototype-derived spatial activation at the same level. $\text{bypass\_ratio} \to 1$ indicates decoder gradient flows almost entirely through skip/encoder features; $\text{bypass\_ratio} \to 0$ indicates it flows through prototype heatmaps. For no-skip models, $\text{bypass\_ratio} = 0$ by architectural construction.

We use monkey-patching to intercept `feat[l]` and `heatmap[l]` as detached leaf tensors within a single forward pass, then measure gradient norms via `logits[0, target_class].sum().backward()`. This requires no modification to the trained model.

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

**Spatial Precision.** For each model and foreground class $k$:

$$\text{SP}_k = \frac{\displaystyle\sum_{x,y} H_k(x,y)\cdot\mathbf{1}[G_k(x,y)=1]}{\displaystyle\sum_{x,y} H_k(x,y)}$$

This quantity represents the fraction of heatmap activation mass falling on the correct ground-truth structure. *Output: `results/v11/spatial_misalignment_precision.csv`*

**GT-Guided Counterfactual.** Cross-model heatmap transplant is confounded by separately trained encoders: feeding one model's prototype similarity scores into another model's SoftMask mixes incompatible feature distributions. We instead construct an upper bound on achievable heatmap localization *within a single model's own feature space*, using ground-truth segmentation masks as supervision.

For each test slice $x$ and model $m$ at level $l$:

1. Run encoder → $\mathbf{F}_l \in \mathbb{R}^{C_l \times H_l \times W_l}$
2. Downsample GT mask to $(H_l, W_l)$ via nearest-neighbor
3. Compute class centroid: $\mathbf{p}_k = \text{mean}_{(i,j): \text{GT}_l(i,j)=k}\,\mathbf{F}_l(:,i,j)$
4. Build Gaussian heatmap: $H_k^{\text{GT}}(i,j) = \exp\!\left(-\|\mathbf{F}_l(:,i,j)-\mathbf{p}_k\|^2/\sigma^2\right)$
5. Stack into $\{l: (1, K, 1, H_l, W_l)\}$ matching the SoftMask input format

We then measure pixel-level localization AP: for each class $k$, AP is computed by treating the upsampled heatmap $H_k$ as a continuous score and the GT mask as binary labels. This is computed for both learned prototype heatmaps (AP\textsubscript{learned}) and GT-guided heatmaps (AP\textsubscript{GT}), and compared across models. All computation stays within one model; no cross-model feature transfer occurs.

*Output: `results/v11/counterfactual_gt_guided_all.csv`*

---

## 4. Results

### 4.1 Core 2×2 Ablation

**Table 1: Full 2×2 XAI Characterization**

| Metric | **skip-32+16** (skip, 32×32+16×16) | **skip-16** (skip, 16×16) | **noskip-32+16** (no-skip, 32×32+16×16) | **noskip-16** (no-skip, 16×16) |
|--------|---------------------------|------------------------|------------------------|---------------------|
| Val Dice | **0.821** | 0.810 | 0.559 | 0.606 |
| Eff. Purity | 0.527 | 0.474 | **0.686** | 0.679 |
| Eff. AP | 0.051 | 0.057 | **0.301** | **0.312** |

*Fig 2 — 2×2 AP and Faithfulness heatmaps: `report/v10/figures/fig2_2x2_heatmap.png`*

**The Skip → No-Skip Trade-off.** Removing skip connections within the same level configuration produces a consistent pattern:

| Comparison | Dice Δ | AP Δ | Purity Δ |
|------------|--------|------|---------|
| 16×16: skip-16 → noskip-16 | −0.204 (−25%) | **+0.255 (+4.5×)** | **+0.205 (+43%)** |
| 32×32+16×16: skip-32+16 → noskip-32+16 | −0.262 (−32%) | **+0.250 (+4.9×)** | **+0.159 (+30%)** |

The trade-off is consistent across both level configurations, confirming that it is a property of the decoder architecture rather than of prototype level selection.

**Level Configuration Effects (16×16 vs. 32×32+16×16).** Within the no-skip family, adding 32×32 slightly reduces both Dice (0.606 → 0.559) and AP (0.312 → 0.301). The 32×32 heatmap provides finer spatial resolution but introduces a second level requiring convergence; the multi-level weighted sum can attenuate the sharper 16×16 signal.

Within the skip family, skip-32+16 and skip-16 exhibit similar AP (0.051 vs. 0.057) — the bypass dominates in both cases, and adding 32×32 does not materially change the causal status of the heatmaps.

### 4.2 Per-Level Analysis and U-Net Upper Bound

**Per-Level Analysis (skip-32+16).** The pixel-dominance fractions ($\phi_l$) and per-level metrics at best-validation epoch:

| Level | $\phi_l$ | Purity | AP |
|-------|---------|--------|----|
| 32×32 | 0.60 | 0.381 | 0.040 |
| 16×16 | 0.40 | 0.744 | 0.067 |

16×16 prototypes are both purer and spatially more precise (higher AP) than 32×32. Yet 32×32 dominates 60% of pixels in the max-aggregated heatmap, because 32×32 features integrate richer spatial context and produce higher raw activation values over large uniform regions — not because they are more interpretable. This tension is a direct consequence of the bypass: the decoder exploits whichever level provides the strongest activation regardless of prototype quality.

**Note on prototype projection validity.** skip-32+16 was evaluated with a freshly run prototype projection. A previously stored projection file contained prototype norms (29.4, 38.8) inconsistent with the checkpoint's own norms (44.8, 63.4), indicating the projection had been saved at an earlier training epoch. The stale projection produced Purity = 0.032 and AP = 0.026 — a factor of ~17 underestimate on Purity. All reported metrics use the corrected fresh projection.

**U-Net AP as Upper Bound.** A baseline 2D U-Net (same encoder, no prototype layer) with its softmax output used as a proxy heatmap achieves AP = 0.349. This constitutes the upper bound on AP for any model in this architecture family — the maximum achievable if prototype spatial activations perfectly matched the final segmentation.

*Fig 4 — AP vs. U-Net upper bound: `report/v10/figures/fig4_ap_vs_unet.png`*

**Table 2: Model AP Relative to U-Net Upper Bound**

| Model | Dice | AP | % of U-Net AP |
|-------|------|----|----------------|
| Baseline U-Net | **0.823** | **0.349** | 100% |
| skip-all (skip, all levels) | 0.817 | 0.102 | 29% |
| skip-16 (skip, 16×16) | 0.810 | 0.057 | 16% |
| skip-32+16 (skip, 32×32+16×16) | 0.821 | 0.051 | 15% |
| **noskip-32+16 (no-skip, 32×32+16×16)** | 0.559 | **0.301** | **86%** |
| **noskip-16 (no-skip, 16×16)** | 0.606 | **0.312** | **89%** |

Skip prototypes reach only 15–29% of U-Net AP. No-skip prototypes reach 86–89% — the prototype compression cost (the gap between prototype precision and segmentation precision) is small once bypass is eliminated. The remaining 9–14% reflects the finite prototype dictionary discretising a continuous spatial signal.

### 4.3 Gradient Attribution

**Table 3: Gradient Attribution — Bypass Ratio vs. AP**

| Model | Config | bypass_ratio | AP |
|-------|--------|-------------|-----|
| skip-16 (skip) | 16×16 only | **0.778** | 0.057 |
| noskip-16 (no-skip) | 16×16 only | 0.000 (structural) | 0.312 |
| skip-32+16 (skip) | 32×32+16×16 | **0.465** | 0.051 |
| noskip-32+16 (no-skip) | 32×32+16×16 | 0.000 (structural) | 0.301 |

**Per-level breakdown (skip models):**

| Model | 32×32 bypass | 16×16 bypass |
|-------|-----------|-----------|
| skip-16 | — (16×16 only) | 0.778 |
| skip-32+16 | 0.433 | 0.497 |

**Interpretation — 16×16 (clean case).** skip-16 $\text{bypass\_ratio} = 0.778$. The decoder draws 78% of its gradient from encoder/skip features; the prototype heatmap is nearly irrelevant to the output. Removing skip connections forces $\text{bypass\_ratio} = 0$ and AP increases 4.5×. This constitutes direct causal evidence that the decoder has learned to route around the prototype signal, and that eliminating the route enforces prototype dependence.

**Interpretation — 32×32+16×16 (nuanced case).** skip-32+16 $\text{bypass\_ratio} = 0.465$ (near-balanced), yet $AP = 0.051$. This reveals that **$\text{bypass\_ratio}$ and AP are orthogonal measurements**:

- **$\text{bypass\_ratio}$** captures *whether* the output is causally sensitive to heatmap changes (causal sensitivity)
- **AP** captures *where* the heatmap activates relative to the correct structure (spatial localisation)

A model may exhibit partial causal sensitivity (moderate $\text{bypass\_ratio}$) while maintaining poorly localised prototypes (near-zero AP). In skip-32+16, interpretability failure operates through *spatial misalignment* in addition to partial causal disconnection — heatmaps activate at incorrect locations, so even when the decoder uses them, they carry incorrect spatial information. Sections 4.4 and 4.5 address this spatial dimension.

### 4.4 Spatial Precision

**Table 4: Spatial Precision — Skip vs. No-Skip**

| Pair | Skip SP | No-Skip SP | Ratio |
|------|---------|-----------|-------|
| 16×16: skip-16 vs. noskip-16 | 0.020 | 0.075 | **3.8×** |
| 32×32+16×16: skip-32+16 vs. noskip-32+16 | 0.020 | 0.062 | **3.1×** |

Skip-model heatmaps exhibit only 25–32% of the spatial precision of no-skip heatmaps. This confirms that the bypass mechanism in skip models is not solely causal (gradient bypass) but also spatial: prototypes activate at incorrect locations, producing heatmaps that do not represent the segmented structure. The spatial misalignment is consistent across both 16×16 and 32×32+16×16 configurations.

**Table 5: Spatial Precision by Class**

| Class | skip-16 SP | noskip-16 SP | skip-32+16 SP | noskip-32+16 SP |
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

### 4.5 GT-Guided Counterfactual

For each model, we compute a GT-guided localization upper bound by replacing learned prototype heatmaps with class-centroid Gaussian heatmaps constructed from the model's own encoder features and the ground-truth mask (see §3.5). The gap between AP\textsubscript{learned} and AP\textsubscript{GT} measures how far learned prototypes fall from the best localization the model's feature space could support — without any cross-model confound.

**Table 6: GT-Guided Counterfactual — Localization AP**

| Model | AP\textsubscript{GT} (upper bound) | AP\textsubscript{learned} | Gap (AP\textsubscript{GT} − AP\textsubscript{learned}) |
|-------|-----------------------------------|--------------------------|-------------------------------------------------------|
| skip-16 (L4) | 0.623 | 0.551 | 0.072 (−11%) |
| skip-32+16 (L3+L4) | 0.389 | 0.078 | 0.311 (−80%) |

Note: Dice is not reported here. Replacing learned prototype heatmaps with GT-guided Gaussian heatmaps changes the magnitude distribution of the soft-mask inputs, and the decoder — trained on prototype-scale activations — is sensitive to this distribution shift regardless of localization content. Dice comparison under this substitution would conflate magnitude mismatch with bypass evidence and is therefore omitted.

**skip-32+16.** Learned prototype heatmaps fall 80% short of the localization quality the model's own feature space could support (AP = 0.078 vs. upper bound 0.389). The large gap confirms that skip-32+16's prototype training produces poorly-localized heatmaps even relative to what its own encoder features could achieve — not just relative to a better-trained model.

**skip-16.** Learned heatmaps reach 89% of the achievable upper bound within skip-16's own feature space (gap = −11%). This is a within-model comparison: it says skip-16's prototype optimization efficiently utilizes whatever localization capacity skip-16's L4 features provide. It does not imply good absolute localization — Section 4.4 shows SP = 0.020, only 27% of noskip-16's SP = 0.075. The absolute gap reflects that skip-16's features themselves have limited localization capacity, because training with skip connections never forces the encoder to develop strongly discriminative features for prototype matching. The −11% GT-guided gap and the 27% SP ratio are complementary, not contradictory: one measures how efficiently skip-16 uses its own feature space; the other measures how that feature space compares to a model trained without bypass. Yet bypass\_ratio = 0.778: regardless of localization quality, 78% of decoder gradient bypasses the prototype heatmaps entirely, confirming causal disconnection.

### 4.6 Mechanistic Summary

**Table 8: Three-Analysis Summary**

| Analysis | 16×16 (skip-16 vs. noskip-16) | 32×32+16×16 (skip-32+16 vs. noskip-32+16) |
|----------|--------------------|-----------------------|
| Gradient bypass | $\text{bypass\_ratio} = 0.778$ | $\text{bypass\_ratio} = 0.465$ |
| Spatial precision | 0.020 vs. 0.075 (3.8×) | 0.020 vs. 0.062 (3.1×) |
| GT-guided counterfactual (gap = AP\textsubscript{GT} − AP\textsubscript{learned}) | 0.623 → 0.551 (−11%) | 0.389 → 0.078 (−80%) |

The three analyses are not co-equal: they occupy a causal chain rather than three independent pillars.

**For skip-16**, gradient attribution establishes the cause: bypass\_ratio = 0.778 means only 22% of decoder gradient flows through prototype heatmaps. Spatial precision (SP = 0.020) is a downstream consequence — with little gradient pressure on prototype localisation, there is no incentive for prototypes to concentrate at the correct anatomical location. SP low does not independently prove bypass; it is what you expect *given* bypass. The GT-guided counterfactual (gap = −11%) adds a third reading: within skip-16's own feature space, the 22% residual gradient is enough to push prototype training relatively close to its localisation ceiling — but both the ceiling and the learned values sit far below noskip-16 in absolute terms, because skip-connected training never forced the encoder to develop strongly discriminative features.

**For skip-32+16**, the causal chain is the same but the magnitudes differ. bypass\_ratio = 0.465 means more gradient reaches the prototypes, yet SP = 0.020 — identical to skip-16. The GT-guided gap (−80%) reveals why: skip-32+16's L3+L4 feature space could support AP = 0.389, but learned prototypes reach only 0.078. The larger gap compared to skip-16 (−11%) indicates that even the 47% residual gradient is insufficient to drive prototype localisation in the multi-level case, where the decoder has additional flexibility to exploit L3 activations over large uniform regions instead of precise boundaries.

---

## 5. Discussion

### 5.1 Is the Trade-off Fundamental?

The skip–no-skip trade-off (25–32% Dice cost for 4.5–5× AP gain) is consistent across both 16×16-only and 32×32+16×16 configurations, across all three mechanistic analyses (gradient attribution, spatial precision, GT-guided counterfactual), and in both directions: removing skip connections raises AP; restoring them suppresses it. This consistency argues against a training artefact explanation.

The gradient attribution identifies the mechanism: the decoder preferentially uses skip-connected encoder features because they provide a higher-quality gradient signal for Dice optimisation. The prototype heatmap — coarser (16×16 vs. 32×32/64×64), discretised into a finite dictionary, and more variable during training — is a lower-quality signal. A model optimising Dice will prefer the skip path. The bypass is not a training failure; it is training succeeding at the wrong objective from an interpretability standpoint.

This implies the trade-off is structural rather than tunable through further training. Prototype training improvements (diversity losses, push-pull regularisation, attention clustering) may increase prototype vector quality (Purity), but as long as the skip pathway remains available, the decoder is not constrained to use the prototype signal. Structural interpretability — the guarantee that $\text{logits} = f(\mathbf{H})$ — requires architectural enforcement, not regularisation alone.

### 5.2 Single-Level vs. Multi-Level: Different Failure Modes

Although both skip-16 and skip-32+16 fail the interpretability criterion, they fail through distinct mechanisms. This distinction matters because they imply different remediation strategies.

**skip-16: pure causal bypass.** bypass\_ratio = 0.778 is the sole pathology. Only 22% of decoder gradient reaches prototype heatmaps, so prototypes receive little training pressure toward correct localisation. The 22% that does reach them is directed at a single coherent feature level (L4), and the GT-guided analysis confirms that these prototypes efficiently utilise their feature space (gap = −11%). The failure is concentrated in the causal pathway; the prototype optimisation itself is not broken within its operating domain.

**skip-32+16: four compounding mechanisms.**

**(1) L3 pixel-dominance misdirects gradient.** In max-aggregation, L3 dominates 60% of pixels. L3 features operate at 32×32 resolution (each cell spans 8×8 pixels), where the natural activation pattern is broad coverage of large anatomical regions rather than precise boundary localisation. The 53% of gradient that reaches prototypes therefore mostly rewards L3 prototypes for wide spatial coverage, not accurate localisation — more gradient, but pointed in the wrong direction. This is the most direct explanation for the large GT-guided gap (−80%): even with more gradient than skip-16, prototype training converges toward diffuse activations.

**(2) Dual bypass paths provide full spatial coverage.** skip-16's single L4 bypass is coarse (16×16); the decoder must rely on higher decoder blocks to recover boundary detail. skip-32+16 simultaneously has L3 (32×32) and L4 (16×16) bypass paths, together covering spatial scales from coarse to mid-resolution. At every scale the decoder can find a bypass route. Although bypass\_ratio is lower (0.465), the bypass is spatially more complete — harder to eliminate without degrading boundary precision.

**(3) Two prototype levels compete for gradient, both undertrained.** With a single level, all prototype gradient concentrates on L4. With two levels, gradient splits between L3 and L4 — and because the decoder dynamically selects which level contributes more (via the relative soft-mask magnitudes), each level is only intermittently load-bearing. Neither level receives stable, consistent gradient pressure sufficient to converge toward precise localisation.

**(4) The GT-guided upper bound itself is lower (0.389 vs. 0.623).** The L3+L4 joint feature space has a lower theoretical localization ceiling than L4 alone — counterintuitive given that more levels are available. The likely cause is max-aggregation: L3's higher activation magnitude over large regions dominates the combined heatmap, suppressing L4's sharper activations. GT centroid heatmaps computed in this combined space inherit the coarseness of L3, reducing AP\textsubscript{GT} even before prototype training is considered.

The practical consequence is that skip-32+16 is harder to remediate than skip-16. Removing skip connections recovers interpretability in both cases, but the recovery in the multi-level setting is slightly smaller (noskip-32+16 AP = 0.301 vs. noskip-16 AP = 0.312). Adding prototype levels does not additively improve interpretability when skip connections are present; it introduces additional misalignment mechanisms on top of the bypass.

### 5.3 No Architecture Simultaneously Achieves High Dice and High AP

**Table 9: Trade-off Summary**

| Configuration | Dice | AP | Purity | Structural Guarantee |
|--------------|------|----|--------|---------------------|
| Skip, 16×16 (skip-16) | 0.810 | 0.057 | 0.474 | ✗ ($\text{bypass\_ratio}=0.778$) |
| Skip, 32×32+16×16 (skip-32+16) | **0.821** | 0.051 | 0.527 | ✗ ($\text{bypass\_ratio}=0.465$) |
| No-skip, 16×16 (noskip-16) | 0.606 | **0.312** | 0.679 | ✅ ($\text{bypass\_ratio}=0$ by construction) |
| No-skip, 32×32+16×16 (noskip-32+16) | 0.559 | **0.301** | **0.686** | ✅ ($\text{bypass\_ratio}=0$ by construction) |

No architecture achieves both Dice > 0.80 and AP > 0.25 simultaneously.

### 5.4 Model Selection

| Priority | Recommended Model | Dice | AP | Rationale |
|----------|------------------|------|----|-----------|
| Clinical deployment | skip-32+16 | 0.821 | 0.051 | Within 2% of baseline U-Net; bypass acknowledged |
| Structural interpretability | noskip-16 | 0.606 | 0.312 | bypass=0 by construction; highest AP |

For clinical use, skip-32+16 is the practical choice. Its prototype heatmaps exhibit spatial co-activation with cardiac structures but are not causal; they should be interpreted as approximate attention maps rather than structural explanations. For research contexts requiring causal prototype attribution (e.g., hypothesis generation about prototype visual content), noskip-16 provides structural guarantees at a Dice cost the researcher must accept.

---

## 6. Future Directions

1. **Progressive skip ablation:** Monotonically remove skip connections (16×16 → 32×32+16×16 → all) to assess whether AP degrades monotonically. Current evidence suggests this would hold, but the gradient has not been characterised.

2. **Heatmap-regularised decoder:** Train a skip decoder with an explicit heatmap alignment loss — penalising outputs that are insensitive to heatmap substitution (the counterfactual near-invariance in Table 6 as a training objective). This would combine high Dice with learned bypass resistance.

3. **Spatial attention pre-filtering:** Apply a learned spatial mask prior to prototype matching to improve prototype localisation (spatial precision) without architectural changes.

4. **MR modality generalisation:** All results are CT-only. The skip–interpretability trade-off may differ for MR, where texture statistics and class boundaries differ substantially.

---

## Appendix A: Checkpoints Reference

| Model | Architecture | Levels | Checkpoint |
|-------|-------------|--------|-----------|
| skip-16 | Skip-connected | 16×16 | `checkpoints/proto_seg_ct_abl_a.pth` |
| skip-32+16 | Skip-connected | 32×32+16×16 | `checkpoints/proto_seg_ct_l3l4_warmstart.pth` |
| noskip-16 | No-skip (heatmap-only) | 16×16 | `checkpoints/proto_seg_ct_v2_l4.pth` |
| noskip-32+16 | No-skip (heatmap-only) | 32×32+16×16 | `checkpoints/proto_seg_ct_v2_l34.pth` |

---

## Appendix B: Per-Class Results (skip-32+16, Skip 32×32+16×16)

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

## Appendix C: skip-16 Ablation (Context for Bypass Baseline)

| Variant | Val Dice | AP |
|---------|----------|----|
| skip-all (all levels, skip) | 0.817 | 0.102 |
| skip-16 (16×16 only, skip) | 0.810 | 0.057 |
| skip-16-nodiv (no diversity loss) | 0.825 | 0.130 |
| skip-16-nomask (no SoftMask) | 0.632 | 0.049 |
| skip-16-nopush (no push-pull) | 0.622 | 0.063 |

AP peaks at 0.13 in skip-16-nodiv (without diversity regularisation) but remains well below U-Net AP = 0.349. The bypass suppresses AP even in the best-case with-skip configuration.

---

## Appendix D: Prototype Projection Validity

When loading a trained checkpoint, the stored projection file must be validated against the checkpoint's own prototype norms before use. A mismatch exceeding 5% on mean norm indicates the projection was saved at a different training epoch. In such cases, run a fresh `PrototypeProjection` pass on the training set before evaluating Purity or AP.

skip-32+16 illustrates this risk: stale projection norms (29.4, 38.8) were inconsistent with checkpoint norms (44.8, 63.4). Fresh evaluation: Purity = 0.527, AP = 0.051. Stale evaluation: Purity = 0.032, AP = 0.026 (17× underestimate on Purity).

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

[16] Huang, Q., et al. (2023). Evaluation and improvement of interpretability for self-explainable part-prototype networks. In *IEEE/CVF International Conference on Computer Vision (ICCV)*. arXiv:2212.05946.

[17] Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017). Network dissection: Quantifying interpretability of deep visual representations. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 6541–6549).

[18] Goyal, Y., Wu, Z., Ernst, J., Batra, D., Parikh, D., & Lee, S. (2019). Counterfactual visual explanations. In *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97, 2376–2384.
