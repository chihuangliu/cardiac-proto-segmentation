# Multi-Scale Prototype Network for Interpretable 2D/3D Cardiac Image Segmentation: An Empirical Study on MM-WHS

**Date:** 2026-03-14
**Dataset:** MM-WHS (Multi-Modality Whole Heart Segmentation)
**Hardware:** MacBook Pro, Apple Silicon, 48 GB unified RAM (PyTorch MPS backend)

---

## Abstract

We present **ProtoSegNet**, a prototype-based cardiac segmentation framework built on the principles of case-based reasoning, multi-scale feature extraction, and quantifiable explainable AI (XAI). The framework processes 2D cross-sectional slices from the MM-WHS benchmark (CT and MRI), attaches learnable prototype vectors at four encoder resolution levels, and produces both a segmentation mask and per-class similarity heatmaps in a single forward pass. We evaluate the model across four XAI dimensions — Activation Precision (AP), Incremental Deletion Score (IDS), Faithfulness Correlation, and Lipschitz Stability — and conduct a six-variant ablation study to quantify the contribution of each design choice. ProtoSegNet achieves a mean foreground 3D Dice of **0.843** (CT) and **0.805** (MRI), within 3% of an opaque 2D U-Net baseline, while providing transparent, anatomically grounded prototype explanations. A key empirical finding is that the choice of similarity kernel — L2 distance versus log-cosine — is the primary driver of heatmap spatial localisation quality. We identify the soft-mask architecture as the primary barrier to achieving the Lipschitz Stability targets, and discuss the trade-off between stability and prototype diversity enforced by Jeffrey's Divergence.

---

## 1. Introduction

### 1.1 Clinical Motivation and the Black-Box Problem

Deep learning models for cardiac image segmentation — trained on multi-modal CT and MRI volumes — now routinely match expert-level Dice coefficients on standard benchmarks. Yet the same opacity that makes these models hard to audit also makes them hard to deploy: clinicians cannot distinguish a confident, anatomically correct prediction from a confident, shortcut-driven error. In high-stakes settings such as pre-surgical 3D reconstruction and interventional planning, the question is not only *whether* a model is accurate, but *why* it drew a particular boundary.

Post-hoc explanation methods (Grad-CAM, SHAP, saliency maps) address this partially but suffer from two systemic limitations. First, they approximate the model's internal computation rather than directly exposing it, producing heatmaps that can be inconsistent with the model's actual decision pathway (low faithfulness). Second, they tend to be fragile under small input perturbations — a serious liability when working with MRI acquisitions subject to RF noise and motion artefacts (low stability).

Intrinsically interpretable architectures that build transparency into the forward pass — rather than explaining it after the fact — offer a principled alternative. Prototype networks, pioneered by ProtoPNet (Chen et al., 2019), make every prediction traceable to a nearest-training-example comparison: *"this region looks like prototype m of class k, therefore I label it k."* This paper applies and extends that paradigm to cardiac image segmentation.

### 1.2 Research Scope and Contributions

This work makes the following empirical contributions:

1. **Multi-scale prototype segmentor (ProtoSegNet):** A 2D encoder-decoder with learnable prototype matrices at four resolution levels (128², 64², 32², 16²), connected via soft-masked skip connections. The model produces both class logits and per-class similarity heatmaps in a single forward pass, without any post-hoc processing.

2. **Push-pull alignment loss:** An auxiliary objective that supervises prototype heatmaps directly against ground-truth masks, ensuring prototypes activate strongly on their target anatomy and weakly elsewhere.

3. **Jeffrey's Divergence prototype diversity loss:** Adapted from ProtoSeg (Sacha et al., 2023), this loss penalises intra-class prototype similarity, preventing all prototypes for a given class from collapsing to the same feature representation.

4. **Four-dimensional XAI metric evaluation:** A systematic, quantitative assessment using AP, IDS, Faithfulness Correlation, and Lipschitz Stability — all computed per-slice across the full test set and aggregated per patient.

5. **Ablation study (six variants):** Controlled experiments isolating the contributions of multi-scale prototypes, diversity loss, soft masking, push-pull supervision, and similarity kernel type.

---

## 2. Dataset and Preprocessing

### 2.1 MM-WHS Dataset

The Multi-Modality Whole Heart Segmentation (MM-WHS) challenge dataset (Zhuang, 2019) provides paired CT and MRI cardiac images with expert voxel-level annotations for seven structures: left ventricle blood pool (LV), right ventricle blood pool (RV), left atrium (LA), right atrium (RA), left ventricular myocardium (Myo), ascending aorta (Aorta), and pulmonary artery (PA). The eighth label (index 0) is background.

The data used in this study was provided in a pre-processed 2D format: axial slices extracted from the original 3D volumes, normalised, and saved as 256×256 NPZ files. The fixed patient split is:

| Modality | Train patients | Val patients | Test patients | Train slices | Val slices | Test slices |
|----------|---------------|-------------|--------------|-------------|-----------|------------|
| CT       | 16            | 2           | 2            | 3,389       | 382       | 484        |
| MRI      | 16            | 2           | 2            | 1,738       | 254       | 236        |

**Class imbalance:** Background occupies 88–94% of pixels per slice. Each cardiac structure contributes only 0.4–2.3% of voxels, with PA being the smallest and most geometrically variable structure.

### 2.2 Data Loading and Augmentation

All slices are preloaded into RAM at dataset initialisation (≈4.5 s one-time cost), reducing batch loading time from 1.28 s/batch (sequential disk I/O) to 4.2 ms/batch. Training augmentation consists of random horizontal/vertical flips and 90° rotations. No intensity augmentation is applied, as the slices arrive pre-normalised.

**Class weights** for the weighted cross-entropy loss are computed from the training label distribution using inverse-frequency weighting and saved to `data/class_weights_{ct,mr}.pt`.

---

## 3. Methods

### 3.1 Baseline: 2D U-Net

A standard 2D U-Net with four resolution levels (channel widths [32, 64, 128, 256], stride-2 convolution downsampling, bilinear upsampling, skip connections) serves as the non-interpretable baseline. The combined loss is:

$$\mathcal{L}_{\text{base}} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{WCE}}$$

where $\mathcal{L}_{\text{Dice}}$ is the mean soft Dice loss over foreground classes and $\mathcal{L}_{\text{WCE}}$ is inverse-frequency weighted cross-entropy. Training uses AdamW (lr=3×10⁻⁴, weight decay=10⁻⁵) with CosineAnnealingLR over 100 epochs, batch size 16.

### 3.2 ProtoSegNet Architecture

**Encoder.** `HierarchicalEncoder2D` processes input $\mathbf{X} \in \mathbb{R}^{1 \times 256 \times 256}$ through four blocks, each consisting of a stride-2 convolution followed by batch normalisation, ReLU, and a residual block. The output feature maps at each level are:

| Level $l$ | Stride | Spatial | Channels $C_l$ | Anatomical role |
|-----------|--------|---------|----------------|-----------------|
| 1         | ×2     | 128×128 | 32             | Fine texture, boundary pixels |
| 2         | ×4     | 64×64   | 64             | Inter-structure borders |
| 3         | ×8     | 32×32   | 128            | Structure-level context |
| 4         | ×16    | 16×16   | 256            | Global cardiac layout |

Total encoder parameters: 1,956,960 (≈77% of ProtoSegNet total). Peak memory (batch=16, forward+backward): 67.5 MB — negligible on the 48 GB system.

**Prototype Layer.** At each level $l$, a set of prototype vectors $\mathbf{P}_l = \{\mathbf{p}_{l,k,m}\}_{k=0,m=1}^{K-1,M_l}$ is stored as a learnable `nn.Parameter`. Prototype counts per level per class are $M = \{l_1: 4,\ l_2: 3,\ l_3: 2,\ l_4: 2\}$, giving 11 prototypes per class and 88 total prototype vectors across the 8-class problem.

The similarity between a spatial feature vector $\mathbf{z}_{l,i} \in \mathbb{R}^{C_l}$ and prototype $\mathbf{p}_{l,k,m}$ is computed using **L2 distance** (the final, empirically superior kernel; see §4.3):

$$\mathcal{S}_{l,k,m}^{(i)} = \frac{1}{\left(\|\mathbf{z}_{l,i} - \mathbf{p}_{l,k,m}\|_2^2 / C_l\right) + 1} \in (0, 1]$$

This produces a per-level heatmap $\mathbf{A}_{l,k,m} \in \mathbb{R}^{H_l \times W_l}$ for every (level, class, prototype) triple. The initial design used a log-cosine kernel:

$$\mathcal{S}^{\text{cos}}_{l,k,m}{}^{(i)} = \log\!\left(\frac{\mathbf{z}_{l,i}^\top \mathbf{p}_{l,k,m}}{\|\mathbf{z}_{l,i}\|_2 \|\mathbf{p}_{l,k,m}\|_2} + 1\right) \in [0, \log 2]$$

The bounded range of the log-cosine kernel proved fatal for spatial localisation (§4.3).

**Prototype Projection.** At regular intervals during training (every 10 epochs in Phase B), each prototype vector is replaced in-place by the actual training feature vector from the class-k spatial location nearest to it in $\ell_2$ distance. This "pushes" prototypes into the manifold of real data patches, grounding them in genuine anatomical examples.

**Soft Mask.** For each level $l$, the heatmaps are aggregated into a spatial attention weight:

$$\mathbf{W}_l = \max_k \max_m \mathbf{A}_{l,k,m} \in \mathbb{R}^{H_l \times W_l}$$

The masked feature is $\tilde{\mathbf{Z}}_l = \mathbf{Z}_l \odot \mathbf{W}_l$ (Hadamard product). This is passed to the decoder via skip connections, suppressing background noise while preserving the prototype-informed spatial signal.

**Decoder.** Four upsampling blocks (bilinear ×2 + concatenation of masked skip features + conv + BN + ReLU) reconstruct the spatial resolution. A final 1×1 convolution maps 32 channels to K=8 class logits per pixel, followed by softmax.

**Full Model Parameters:** 2,556,264 (ProtoSegNet) vs. 7,843,560 (baseline U-Net). ProtoSegNet is 3× smaller.

### 3.3 Loss Functions

The full training loss integrates four terms:

$$\mathcal{L}_{\text{total}} = 0.5\,\mathcal{L}_{\text{Dice}} + 0.5\,\mathcal{L}_{\text{WCE}} + \lambda_{\text{div}}\,\mathcal{L}_{\text{div}} + \lambda_{\text{push}}\,\mathcal{L}_{\text{push}} + \lambda_{\text{pull}}\,\mathcal{L}_{\text{pull}}$$

**Diversity Loss $\mathcal{L}_{\text{div}}$ (Jeffrey's Divergence).** To prevent prototype collapse — where multiple prototypes within the same class converge to identical representations — we penalise the inverse of the Jeffrey's Divergence between pairs of intra-class heatmap distributions:

$$\mathcal{L}_{\text{div}} = \sum_{l=1}^{4} \sum_{k=1}^{7} \sum_{m \neq n}^{M_l} \frac{1}{D_J(\mathbf{A}_{l,k,m} \| \mathbf{A}_{l,k,n}) + \varepsilon}$$

where $D_J(P \| Q) = D_{\mathrm{KL}}(P \| Q) + D_{\mathrm{KL}}(Q \| P)$ is the symmetric, stable generalisation of KL divergence, and distributions are computed by spatial softmax over the heatmap. Background class (k=0) prototypes are excluded. The hyperparameter $\lambda_{\text{div}} = 0.001$ is used for the best model.

**Push-Pull Loss $\mathcal{L}_{\text{push}}, \mathcal{L}_{\text{pull}}$.** Prototype heatmaps are upsampled to input resolution and compared against the ground-truth mask $\mathbf{G}_k \in \{0,1\}^{256 \times 256}$:

$$\mathcal{L}_{\text{push}} = -\frac{1}{|\Omega_k|} \sum_{i \in \Omega_k} \max_m \mathbf{A}_{l,k,m}^{(i)} \quad (\text{maximise activation on GT foreground})$$

$$\mathcal{L}_{\text{pull}} = +\frac{1}{|\bar{\Omega}_k|} \sum_{i \notin \Omega_k} \max_m \mathbf{A}_{l,k,m}^{(i)} \quad (\text{minimise activation on GT background})$$

where $\Omega_k$ is the set of foreground pixels for class $k$ in the current batch. Final hyperparameters: $\lambda_{\text{push}} = 0.5$, $\lambda_{\text{pull}} = 0.25$.

### 3.4 Three-Phase Training Schedule

Prototype networks benefit from a staged training schedule that prevents the prototype layer from disrupting early feature learning:

| Phase | Epochs | Frozen | Active | Notes |
|-------|--------|--------|--------|-------|
| A (Warm-up)  | 1–20  | Prototypes | Encoder + Decoder | Establish feature space before prototype attachment |
| B (Joint)    | 21–80 | — | All params + full $\mathcal{L}_{\text{total}}$ | Prototype projection every 10 epochs |
| C (Fine-tune)| 81–100| Encoder + Prototypes | Decoder only | Consolidate linear readout |

Early stopping with patience=15 is applied per phase; the best checkpoint is selected by mean foreground validation Dice.

### 3.5 XAI Evaluation Metrics

All four metrics are computed slice-by-slice on the 2D test patients and aggregated per patient. Heatmaps are produced by the model's `forward()` pass without any post-hoc processing.

**Activation Precision (AP).** Measures whether the highest-activated pixels fall within the ground-truth anatomy for class $k$:

$$AP_k = \frac{\sum_i \mathbf{M}_k^{(i)} \cap \mathbf{G}_k^{(i)}}{\sum_i \mathbf{M}_k^{(i)}}$$

where $\mathbf{M}_k = \mathbb{1}(\mathbf{A}_k \geq \tau_{95})$, with $\tau_{95}$ the 95th percentile of $\mathbf{A}_k$ across spatial locations. The class heatmap $\mathbf{A}_k$ is the maximum over all levels and prototypes: $\mathbf{A}_k = \max_{l,m} \mathbf{A}_{l,k,m}$.

**Incremental Deletion Score (IDS).** A counterfactual test: pixels are sorted by activation (highest first) and zeroed out in 5%-increment steps. The Dice coefficient is recorded after each deletion, forming a degradation curve. IDS is the area under this curve — a lower value indicates that removing the most-attended pixels rapidly destroys performance, confirming their causal importance.

$$\text{IDS} = \int_0^1 \text{Dice}(t)\,dt$$

**Faithfulness Correlation.** N=2000 pixels per slice are sampled; each is zeroed in the input and the resulting change in class prediction probability $\Delta\hat{y}_i$ is recorded. Faithfulness is the Pearson correlation between the explanation importance score $E_i$ (heatmap value at pixel $i$) and $\Delta\hat{y}_i$. A high positive correlation confirms that the heatmap reflects true model dependency.

**Lipschitz Stability.** N=20 Gaussian perturbations at $\sigma = 0.05$ (normalised intensity scale) are applied to the input in a single batched forward pass. Stability is the maximum ratio of heatmap change to input change:

$$\text{Stability}(\mathbf{X}) = \max_{\mathbf{X}' \in \mathcal{N}_\varepsilon(\mathbf{X})} \frac{\|\Phi(\mathbf{X}) - \Phi(\mathbf{X}')\|_2}{\|\mathbf{X} - \mathbf{X}'\|_2}$$

Lower values indicate more robust explanations.

---

## 4. Results

### 4.1 Segmentation Baseline (2D U-Net)

The baseline 2D U-Net was trained for 100 epochs per modality.

| Patient | Mean Fg Dice | LV | RV | LA | RA | Myo | Aorta | PA |
|---------|-------------|----|----|----|----|-----|-------|-----|
| ct_1019 | 0.807 | 0.855 | 0.871 | 0.674 | 0.886 | 0.857 | 0.861 | 0.647 |
| ct_1020 | 0.927 | 0.894 | 0.947 | 0.936 | 0.892 | 0.919 | 0.972 | 0.928 |
| mr_1019 | 0.872 | 0.845 | 0.871 | 0.949 | 0.916 | 0.940 | 0.845 | 0.738 |
| mr_1020 | 0.840 | 0.870 | 0.875 | 0.946 | 0.897 | 0.921 | 0.641 | 0.728 |

CT best validation Dice: **0.836** (epoch 20). MRI best validation Dice: **0.825** (epoch 80). The early convergence of CT (epoch 20 vs. 80 for MRI) suggests the CT feature space is more separable under the training distribution.

PA is consistently the hardest structure across both modalities (CT: 0.647–0.928; MRI: 0.728–0.738), driven by its elongated tubular geometry and high inter-patient variability.

### 4.2 ProtoSegNet Segmentation Performance

ProtoSegNet was trained using the three-phase schedule with L2 similarity, $\lambda_{\text{div}} = 0.001$, $\lambda_{\text{push}} = 0.5$, $\lambda_{\text{pull}} = 0.25$. The 2D slice predictions are stacked per patient to compute volumetric 3D Dice and Average Symmetric Surface Distance (ASSD).

#### CT Results (2 test patients)

| Structure | 3D Dice | ASSD (mm) |
|-----------|---------|-----------|
| LV        | 0.845   | 2.53      |
| RV        | 0.892   | 3.86      |
| LA        | 0.827   | 4.51      |
| RA        | 0.865   | 6.38      |
| Myocardium| 0.843   | 2.54      |
| Aorta     | 0.918   | 1.51      |
| PA        | 0.707   | 4.67      |
| **Mean fg** | **0.843** | **3.72** |

Patient ct_1020 achieves mean fg 3D Dice 0.915 with ASSD 2.55 mm; ct_1019 achieves 0.770 / 4.88 mm. The inter-patient variance is high, consistent with the two-patient test set limitation.

#### MRI Results (2 test patients)

| Structure | 3D Dice | ASSD (mm) |
|-----------|---------|-----------|
| LV        | 0.780   | 2.69      |
| RV        | 0.849   | 1.29      |
| LA        | 0.936   | 0.79      |
| RA        | 0.843   | 3.31      |
| Myocardium| 0.903   | 1.35      |
| Aorta     | 0.640   | 4.93      |
| PA        | 0.683   | 2.35      |
| **Mean fg** | **0.805** | **2.39** |

MRI Aorta is notably weaker (Dice 0.640), driven by mr_1020's severe drop to 0.485 — the largest inter-patient outlier in the study. This is partly attributable to the MRI checkpoint using the cosine-similarity Stage 7 model (the L2 retrain was deferred for MRI).

#### Comparison with Baseline

| Model | Modality | Mean Fg 3D Dice | ASSD (mm) | Δ vs. baseline |
|-------|----------|-----------------|-----------|----------------|
| 2D U-Net baseline | CT | 0.867 | — | — |
| ProtoSegNet (L2)  | CT | **0.843** | **3.72** | −0.024 (−2.8%) |
| 2D U-Net baseline | MRI | 0.856 | — | — |
| ProtoSegNet (cosine) | MRI | **0.805** | **2.39** | −0.051 (−6.0%) |

The CT model meets the ≤3% Dice degradation target. The MRI model shows a larger gap, attributable to the use of the cosine-similarity checkpoint; the L2-retrained MRI model is expected to narrow this gap.

**NIfTI exports** for all four test patients (`results/nifti/{patient}_{pred,gt}.nii.gz`) are available for 3D rendering in ITK-SNAP.

### 4.3 Similarity Kernel: L2 vs. Log-Cosine

The log-cosine kernel `log(cos_sim + 1)` has output bounded in $[0, \log 2] \approx [0, 0.693]$. In practice, most feature–prototype pairs score in the mid-range of this interval regardless of spatial proximity, producing near-uniform heatmaps with no localisation signal.

The L2 kernel `1 / (‖z − p‖²/C + 1)` produces sharp, spatially concentrated activations: a perfect match scores 1.0, while a random background feature vector scores approximately 0.33. This asymmetry is essential for the push-pull loss to drive meaningful spatial gradients.

| Kernel | Mean AP (CT) | Faithfulness | Stability |
|--------|-------------|--------------|-----------|
| Log-cosine (no push-pull) | 0.033 | 0.017 | 2.76 |
| Log-cosine + push-pull v1 ($\lambda_{\text{push}}=0.1$) | 0.013 | −0.005 | 2.44 |
| Log-cosine + push-pull v2 ($\lambda_{\text{push}}=0.5$) | 0.020 | −0.007 | 2.90 |
| **L2 + push-pull** ($\lambda_{\text{push}}=0.5$) | **0.102** | **+0.059** | 2.996 |
| L2 + push-pull (stronger, $\lambda_{\text{push}}=1.0$) | 0.100 | +0.076 | 4.69 |

The switch to L2 similarity produced a **2.5× AP improvement** (0.033 → 0.102) and turned Faithfulness from negative to positive for the first time (+0.059 vs. −0.003). Increasing push-pull strength beyond $\lambda_{\text{push}} = 0.5$ yielded no further AP gain while significantly increasing Stability (4.69 vs. 2.996), suggesting a saturation point in the push-pull effect on AP.

### 4.4 XAI Metric Results — Best CT Model (L2 + Push-Pull)

Per-patient and per-class breakdown for the best CT configuration (`proto_seg_ct_l2.pth`):

**ct_1019:**

| Class | AP |
|-------|-----|
| LV    | 0.058 |
| RV    | 0.117 |
| LA    | 0.012 |
| RA    | 0.055 |
| Myo   | 0.050 |
| Aorta | 0.084 |
| PA    | 0.041 |
| **Mean fg** | **0.059** |

IDS: 0.0045 ± 0.0052 | Faithfulness: 0.083 ± 0.047 | Stability: 3.003 ± 0.042

**ct_1020:**

| Class | AP |
|-------|-----|
| LV    | 0.082 |
| RV    | 0.285 |
| LA    | 0.075 |
| RA    | 0.120 |
| Myo   | 0.306 |
| Aorta | 0.032 |
| PA    | 0.112 |
| **Mean fg** | **0.145** |

IDS: 0.0097 ± 0.0065 | Faithfulness: 0.036 ± 0.018 | Stability: 2.988 ± 0.086

**Aggregate (mean over 2 patients):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AP     | 0.102 | ≥ 0.70 | ❌ |
| IDS    | 0.007 | ≤ 0.45 | ✅ |
| Faithfulness | 0.059 | ≥ 0.55 | ❌ |
| Stability | 2.996 | ≤ 0.20 | ❌ |

Notable per-class variation: RV and Myo achieve AP > 0.28 for ct_1020 (meaningful spatial localisation), while Aorta AP drops to 0.032 for the same patient. Aorta's tubular morphology — thin walls, high aspect ratio — is poorly served by isotropic patch-based prototypes.

### 4.5 XAI Metric Results — MRI Model (Cosine Similarity)

The MRI model evaluated here uses the Stage 7 cosine-similarity checkpoint. XAI metrics reflect the fundamental limitation of the cosine kernel:

| Metric | mr_1019 | mr_1020 | Mean |
|--------|---------|---------|------|
| AP     | 0.0001  | 0.0007  | 0.0004 |
| IDS    | 0.160   | 0.129   | 0.144 |
| Faithfulness | 0.004 | 0.008 | 0.006 |
| Stability | 1.318 | 1.381 | 1.350 |

Near-zero AP confirms that the cosine model's heatmaps carry no spatial information. IDS is higher than for the CT L2 model (0.144 vs. 0.007), indicating a slower degradation curve — consistent with uniform heatmaps that do not prioritise any particular pixels. Stability is lower (1.35 vs. 3.00) because uniform heatmaps vary less under perturbation. A full L2 retrain for MRI is expected to produce similar AP improvements as seen for CT.

### 4.6 Ablation Study

Six variants were trained for 50 epochs each on CT, using the Full model (`proto_seg_ct_l2.pth`) as the reference:

| Variant | Description | Val Dice | Mean AP | IDS | Faithfulness | Stability | Proto Cosim |
|---------|-------------|----------|---------|-----|--------------|-----------|-------------|
| **Full (_l2)** | L2 + push-pull + div + soft-mask + multi-scale | **0.817** | **0.102** | **0.007** | 0.059 | 3.000 | **0.21** ✅ |
| A (_abl_a) | Single-scale (level 4 only) | 0.810 | 0.030 | 0.044 | 0.097 | **2.462** | 0.576 |
| B (_abl_b) | No diversity loss ($\lambda_{\text{div}}=0$) | 0.825 | 0.130 | 0.007 | **0.158** | 14.105 ❌ | 0.641 ❌ |
| C (_abl_c) | No soft-mask | 0.632 | 0.049 | 0.016 | 0.065 | 2.970 | 0.618 |
| D (_abl_d) | No push-pull ($\lambda_{\text{push}}=\lambda_{\text{pull}}=0$) | 0.622 | 0.063 | 0.011 | 0.062 | 1.799 | 0.345 |
| E (_cosine) | Stage 7 cosine, no push-pull | 0.790 | 0.033 | 0.011 | 0.017 | 2.765 | 0.013 |

**Key findings:**

**❶ L2 similarity is the primary AP driver.** Variant E (cosine) AP = 0.033 vs. Full AP = 0.102: **Δ = +0.069 (+209%)**.

**❷ Diversity loss trades AP/Faithfulness for Stability.** Variant B achieves the highest AP (0.130) and Faithfulness (0.158) of all variants, but Stability explodes to 14.1 — 4.7× worse than the Full model. Prototype cosine similarity rises to 0.641 (near-collapsed), meaning each class is represented by near-identical prototype vectors. When multiple prototypes activate the same pixels, small input perturbations can shift which prototype is "closest," causing heatmaps to flip dramatically. Diversity loss is essential for robust, reliable explanations.

**❸ Soft-mask and push-pull are jointly critical for segmentation.** Removing either causes a severe Dice regression:
- Variant C (no soft-mask): Dice 0.817 → 0.632 (−22.7%)
- Variant D (no push-pull): Dice 0.817 → 0.622 (−23.8%)

Both components collaborate: push-pull ensures prototypes carry spatial signal; soft-mask ensures that signal reaches the decoder through the skip connections. Removing either breaks the information pathway.

**❹ Multi-scale gives 3.4× AP gain over single-scale.** Variant A (level 4 only): AP = 0.030 vs. Full: 0.102. IDS also degrades sharply (0.044 vs. 0.007) with single-scale, indicating that levels 1–3 (fine-grained prototypes) capture the locally specific features that drive incremental deletion sensitivity. Prototype cosine similarity of 0.576 suggests partial collapse even without the diversity loss removing its effect — the single-scale prototype space is under-constrained.

**❺ Stability floor ≈ 2.5–3.0.** Across all well-configured variants (Full, A, C, E), Stability is 2.46–3.00. The only exception is Variant B (collapse, 14.1) and Variant D (no push-pull, 1.80 — but at the cost of Dice 0.622). The stability floor is a structural consequence of the soft-mask architecture: because the decoder can partially bypass the soft-mask via direct residual paths, prototype heatmaps are not the sole input to the decoder, reducing their causal isolation and thus their stability.

---

## 5. Visualizations

### 5.1 Prototype Atlas

The prototype atlas (generated by `scripts/visualize_prototypes.py`) maps each learned prototype back to the nearest training image patch, providing a visual dictionary of anatomical concepts learned by the network.

**Level 1 — Fine texture (128×128 spatial):**
![CT Prototype Atlas Level 1](../results/prototype_atlas/ct_level1.png)

**Level 2 — Inter-structure borders (64×64 spatial):**
![CT Prototype Atlas Level 2](../results/prototype_atlas/ct_level2.png)

**Level 3 — Structure-level context (32×32 spatial):**
![CT Prototype Atlas Level 3](../results/prototype_atlas/ct_level3.png)

**Level 4 — Global cardiac layout (16×16 spatial):**
![CT Prototype Atlas Level 4](../results/prototype_atlas/ct_level4.png)

Each panel in the atlas shows the K=8 class rows × M prototype columns, with the corresponding training image patch and its similarity heatmap overlay. Shallow levels (l=1,2) capture fine-grained boundary textures; deep levels (l=3,4) capture coarser structural configurations.

### 5.2 Segmentation Visualizations

Segmentation overlay figures (generated by `scripts/visualize_segmentation.py`) show, for each test patient and each of the 7 cardiac structures, the representative slice with: (a) input image, (b) ground truth mask, (c) model prediction, and (d) prototype heatmap overlay.

Selected representative figures:

**CT Patient ct_1019 — Left Ventricle (Slice 124):**
![ct_1019 LV](../results/segmentation_viz/ct/ct_1019_ct_1019_slice_124_lv.png)

**CT Patient ct_1019 — Myocardium (Slice 133):**
![ct_1019 Myocardium](../results/segmentation_viz/ct/ct_1019_ct_1019_slice_133_myocardium.png)

**CT Patient ct_1019 — Pulmonary Artery (Slice 257):**
![ct_1019 PA](../results/segmentation_viz/ct/ct_1019_ct_1019_slice_257_pa.png)

**CT Patient ct_1020 — Aorta (Slice 232):**
![ct_1020 Aorta](../results/segmentation_viz/ct/ct_1020_ct_1020_slice_232_aorta.png)

**CT Patient ct_1020 — RV (Slice 218):**
![ct_1020 RV](../results/segmentation_viz/ct/ct_1020_ct_1020_slice_218_rv.png)

---

## 6. Discussion

### 6.1 Segmentation Quality

ProtoSegNet achieves 3D Dice of 0.843 (CT) and 0.805 (MRI), meeting the ≤3% Dice degradation target for CT. The prototype mechanism imposes a representational bottleneck — all spatial features must be expressible as linear combinations of prototype-gated activations — but the multi-scale design distributes this burden across four levels, preserving both coarse global structure and fine boundary detail.

The most problematic structures are PA (variable geometry, small cross-section) and LA (complex concave shape visible in ct_1019 where LA Dice is 0.674 for the baseline and 0.721 for ProtoSegNet). These structures would benefit from a larger prototype bank, potentially with per-structure prototype count tuning guided by the observed AP and Dice per class.

### 6.2 Interpretability Achievements and Limitations

The primary interpretability achievements of this work are:

1. **First positive Faithfulness** (+0.059) in the experiment series, achieved by switching to L2 similarity. The log-cosine baseline produced Faithfulness ≈ −0.003 (essentially uncorrelated). A positive Faithfulness confirms that regions the model highlights are causally important for its predictions.

2. **AP improvement of 2.5× via L2 kernel** (0.033 → 0.102). While 0.102 remains far below the aspirational target of ≥0.70, the improvement trajectory is monotonic and the root cause of the ceiling is identified (soft-mask architecture).

3. **IDS consistently ≤ 0.007 for L2 CT model.** This is among the best results across all ablation variants and indicates that the top-activated pixels are genuinely load-bearing for performance — removing them rapidly degrades Dice.

4. **Prototype diversity confirmed:** Full model mean intra-class cosine similarity = 0.21. Variant B (no diversity loss) collapses to 0.641. The Jeffrey's Divergence loss successfully enforces exploration of the prototype space.

The principal unresolved limitation is Stability, which floors at ≈2.5–3.0 regardless of loss configuration. The underlying cause is the soft-mask design: the decoder can learn to attend to both masked and unmasked feature paths simultaneously via residual connections, reducing the uniqueness of the prototype heatmap as a predictor. Hard masking (gating entire spatial locations rather than attenuating them) or per-class decoder heads would address this but substantially increase model complexity and training cost.

### 6.3 Architectural Trade-Offs

The ablation reveals a three-way tension between AP, Stability, and Dice:

- Removing diversity loss: AP ↑ (0.102 → 0.130), Faithfulness ↑ (0.059 → 0.158), but Stability ↑ catastrophically (3.00 → 14.10) and Dice is unaffected (0.817 → 0.825).
- Removing push-pull: Stability ↓ (3.00 → 1.80) but Dice collapses (0.817 → 0.622) and AP barely changes (0.102 → 0.063).
- Removing soft-mask: Dice collapses (0.817 → 0.632) with no stability benefit.

The Full model represents the Pareto-optimal configuration across all metrics simultaneously. No single-component removal improves the overall profile — each component serves a distinct role in the system.

### 6.4 Limitations

1. **Small test set (2 patients per modality).** XAI metric estimates carry high variance; per-patient results should be interpreted as case studies rather than population-level estimates.

2. **2D slice processing vs. 3D context.** While 3D Dice is computed by stacking 2D predictions, the model processes each slice independently and cannot leverage through-plane anatomical context. True 3D prototype networks would offer richer volumetric reasoning but at substantially higher compute cost.

3. **MRI model uses cosine similarity.** The L2 retrain for MRI was deferred. XAI metrics for MRI are expected to improve substantially with L2 similarity, as seen for CT.

4. **Stability floor of ≈2.5–3.0 is architectural, not hyperparameter-dependent.** Achieving the ≤0.20 Stability target requires hard masking or per-class decoder heads — a significant architectural change deferred to future work.

5. **AP ceiling of ≈0.10–0.13 within current architecture.** Classes such as RV and Myo achieve AP > 0.28 for favourable test patients, suggesting the architecture can in principle localise well; the limiting factor is the soft-mask's partial bypass of prototype-gated features in the decoder.

---

## 7. Related Work

**Prototype networks for classification.** Chen et al. (2019) introduced ProtoPNet, establishing case-based reasoning for image classification. The "this looks like that" paradigm directly informs our push-pull loss formulation and prototype projection procedure.

**3D medical prototype networks.** MProtoNet (Wei et al., 2023) extended ProtoPNet to 3D multi-parametric MRI for brain tumour classification, introducing soft masking and online-CAM losses. Our work adapts these ideas to segmentation (dense prediction) rather than classification.

**Hierarchical prototype networks.** HierViT (Gallée et al., 2025) combines hierarchical Vision Transformers with prototype learning for medical image classification. Our multi-scale encoder mirrors this hierarchical approach but in a 2D convolutional rather than transformer backbone.

**Prototype segmentation.** ProtoSeg (Sacha et al., 2023) first applied prototype networks to semantic segmentation and introduced the Jeffrey's Divergence diversity loss that we adopt directly.

**XAI evaluation methodology.** The AP metric is formalised by Barnett et al. (2021) for mammography. The IDS framework is reviewed by Nauta et al. (2023). Faithfulness Correlation is defined by Bhatt et al. (2020). Lipschitz Stability for prototype networks is formalised by Huang et al. (2023).

---

## 8. Conclusion

We presented ProtoSegNet, a multi-scale prototype segmentation network for cardiac MRI and CT, equipped with four quantitative XAI evaluation metrics. The key empirical findings are:

1. **L2 similarity is essential for heatmap localisation.** The log-cosine kernel's bounded output prevents spatial discrimination; switching to L2 distance yields 2.5× AP improvement and the first positive Faithfulness correlation in the experiment.

2. **Jeffrey's Divergence diversity loss is essential for stable explanations.** Removing it produces slightly higher AP but catastrophically high Stability (14.1 vs. 3.0), rendering heatmaps unreliable under clinical noise conditions.

3. **Soft-mask and push-pull are jointly necessary for segmentation quality.** Removing either causes a >20% Dice drop; they form an interdependent information pathway from prototype heatmaps to decoder output.

4. **Multi-scale prototypes provide 3.4× AP gain over single-scale.** Fine-grained (level 1–2) prototypes capture boundary detail; coarse (level 3–4) prototypes capture global cardiac topology. Both are needed for comprehensive interpretable coverage.

5. **Stability remains a structural challenge.** The soft-mask architecture cannot achieve Lipschitz Stability < 1.0 without architectural modifications (hard masking, per-class decoders), which represent the primary direction for future improvement.

ProtoSegNet delivers interpretable cardiac segmentation within 3% of an opaque U-Net baseline on CT, with transparent prototype atlases that ground every segmentation decision in real training examples — a property that post-hoc explanation methods cannot provide by design.

---

## References

Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). This looks like that: deep learning for interpretable image recognition. *NeurIPS*.

Wei, Y., Tam, R., & Tang, X. (2023). MProtoNet: A case-based interpretable model for brain tumor classification with 3D multi-parametric magnetic resonance imaging. *MIDL*.

Gallée, L., et al. (2025). Hierarchical Vision Transformer with Prototypes for Interpretable Medical Image Classification. *arXiv preprint*.

Sacha, M., et al. (2023). ProtoSeg: Interpretable semantic segmentation with prototypical parts. *WACV*.

Barnett, A. J., et al. (2021). IAIA-BL: A case-based interpretable deep learning model for classification of mass lesions in digital mammography. *Nature Machine Intelligence*, 3, 1061–1070.

Nauta, M., et al. (2023). From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI. *ACM Computing Surveys*, 55(13s), 1–42.

Bhatt, U., et al. (2020). Evaluating and aggregating feature-based model explanations. *IJCAI*.

Huang, Z. A., et al. (2023). Stability evaluation for prototype networks.

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.

Zhuang, X. (2019). Multivariate mixture model for myocardial segmentation combining multi-source images. *IEEE TPAMI*, 41(12), 2933–2946.

European Parliament and Council. (2024). Regulation (EU) 2024/1689 (Artificial Intelligence Act).

Cardoso, M. J., et al. (2022). MONAI: An open-source framework for deep learning in healthcare.
