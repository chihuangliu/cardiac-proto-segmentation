# Execution Plan: Multi-Scale Prototype 3D Cardiac Segmentation Framework

**Project:** Interpretable 3D Cardiac Image Segmentation with Quantifiable XAI
**Dataset:** MM-WHS (60 CT + 60 MRI, 7 cardiac structures)
**Hardware:** MacBook 48GB RAM (Apple Silicon MPS backend, no discrete GPU)
**Last Updated:** 2026-03-12
**Stage 0 Completed:** 2026-03-11
**Stage 1 Completed:** 2026-03-12

---

## Hardware Constraints & Mitigation Strategy

| Constraint | Original Plan | Revised (Post Stage 0) |
|---|---|---|
| No dedicated CUDA GPU | Use PyTorch MPS backend | Same — confirmed MPS available |
| 48GB unified RAM | 96³ patch-based training | Not needed — data is 2D 256×256 slices |
| Slow training vs. GPU cluster | 3D ResNet-18, mixed precision | 2D backbone much faster per epoch |
| Large 3D volumes | MONAI lazy loading | No NIfTI volumes — NPZ slices load in 18ms/batch |
| Prototype projection memory | CPU eval mode, batch size 1–2 | 2D projection is trivial, can run inline |

**Actual training time per epoch (MPS, 2D):** ~85s CT / ~42s MRI (batch 16, preload=True)
**Max batch size recommendation:** 16 (confirmed, ~4GB peak RAM)

---

## Stage Overview

| Stage | Name | Deliverable | Status |
|---|---|---|---|
| 0 | Environment & Data | Working data pipeline | ✅ Complete |
| 1 | Baseline 2D U-Net | Reproducible segmentation baseline | ✅ Complete |
| 2 | Multi-Scale Backbone | Multi-resolution feature maps Z_l | ⬜ Next |
| 3 | Prototype Layer | Prototype similarity heatmaps | ⬜ |
| 4 | Diversity Loss | Jeffrey's Divergence L_div | ⬜ |
| 5 | Decoder & Full Pipeline | End-to-end trainable prototype segmentor | ⬜ |
| 6 | XAI Metrics | AP, IDS, Faithfulness, Stability modules | ⬜ |
| 7 | Training & Evaluation | Full CT+MRI benchmark results | ⬜ |
| 8 | Ablation & Visualization | Ablation table + prototype atlas | ⬜ |

---

## Stage 0 — Environment & Data Pipeline ✅ COMPLETE

### Goal
Reproducible project environment, preprocessed MM-WHS data ready for DataLoader.

### Key Finding: Data is Pre-Processed 2D Slices
The provided data pack (`data/pack/processed_data/`) contains **2D slices (256×256)** in NPZ format,
already normalized and split. The architecture is adapted to **2D multi-scale prototype segmentation**
instead of 3D — this is more feasible on Mac hardware and the framework principles are identical.

Data summary (from `data/splits.json`):
- CT: 3389 train / 382 val / 484 test slices, 16/2/2 patients
- MR: 1738 train / 254 val / 236 test slices, 16/2/2 patients
- Severe class imbalance: Background ~88–94%, each structure ~0.4–2.3%

### Tasks
- [x] Set up `pyproject.toml` with dependencies: `torch`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tqdm`, `nibabel`
- [x] Create `.venv` with `uv` (Python 3.12), install all dependencies
- [x] Inspect data pack: confirmed 2D NPZ slices, already normalized, pre-split
- [x] Verify label mapping: {0:BG, 1:LV, 2:RV, 3:LA, 4:RA, 5:Myo, 6:Aorta, 7:PA}
- [x] Implement `MMWHSSliceDataset` (slice-level) with random flip/rotate augmentation
- [x] Implement `MMWHSPatientDataset` (patient-level, groups slices for volumetric eval)
- [x] Implement `make_dataloaders()` utility
- [x] Save split info to `data/splits.json`
- [x] Run `scripts/explore_data.py`: verified shapes, timing, label distribution
- [x] Save sample visualizations to `results/sample_ct.png`, `results/sample_mr.png`

### Actual Outcome
```
data/
  pack/processed_data/   # pre-processed 2D NPZ slices
    ct_256/{train,val,test}/npz/
    mr_256/{train,val,test}/npz/
  splits.json            # patient/slice counts per split
src/data/mmwhs_dataset.py  # MMWHSSliceDataset + MMWHSPatientDataset
scripts/explore_data.py
results/sample_ct.png
results/sample_mr.png
```
DataLoader yields `(B, 1, 256, 256)` image tensors, `(B, 256, 256)` label tensors in ~0.018s/batch.

---

### Stage 0 Retrospective — Deviations from Original Plan

#### ❶ Data is 2D, not 3D [MAJOR — cascades through all stages]
- **Expected:** Raw 3D NIfTI volumes requiring preprocessing (clipping, resampling, patch extraction)
- **Actual:** Data is already sliced into 2D 256×256 NPZ files, normalized, and split
- **Impact:** Entire architecture shifts from 3D to 2D. All `3D` references in Stages 1–8 must be replaced with `2D`. No MONAI or nibabel needed. Training is significantly faster.

#### ❷ Preprocessing step is eliminated [MINOR — positive]
- **Expected:** `scripts/preprocess.py` to normalize and resample raw volumes (~1–2h compute)
- **Actual:** Not needed. Preprocessing was done upstream by the data provider.
- **Impact:** Removes Stage 0 preprocess task. Simplifies pipeline.

#### ❸ Splits are fixed, not 5-fold CV [MINOR]
- **Expected:** Implement 5-fold cross-validation over 60 patients per modality
- **Actual:** Fixed splits: 16 train / 2 val / 2 test patients per modality (already defined in directory structure)
- **Impact:** The test set of only **2 patients** is very small. XAI metric estimates in Stage 6 will have high variance. Mitigation: report per-patient results individually rather than aggregated mean, and note limitation.

#### ❹ Severe class imbalance discovered [MODERATE — affects Stages 1, 7]
- **Expected:** Mild imbalance typical of cardiac segmentation
- **Actual:** Background occupies 88–94% of voxels; each cardiac structure only 0.4–2.3%
- **Impact:** Plain CE loss will collapse to predicting background. Must use **class-weighted CE** or **focal loss** in Stage 1 onwards. Dice loss alone (which is class-balanced by design) becomes even more critical.

#### ❺ MONAI not needed; nibabel reinstated for NIfTI export [MINOR]
- **Expected:** MONAI as core dependency for 3D transforms and sliding-window inference
- **Actual:** Pure PyTorch + NumPy suffices for training. MONAI removed (~800MB saved).
- **nibabel** was initially removed but reinstated in Stage 1 — required for NIfTI export in Stages 7/8 (3D volume reconstruction for ITK-SNAP rendering).
- **Impact:** `pyproject.toml` keeps `nibabel`. IDS metric uses per-slice inference (no sliding window).

---

## Stage 1 — Baseline 2D U-Net ✅ COMPLETE

### Goal
Establish a reproducible 2D segmentation baseline (mean foreground Dice ≥ 0.75 on CT, ≥ 0.70 on MRI) to validate the data pipeline before adding prototype complexity.

> **Revised from original:** 3D U-Net → 2D U-Net. MONAI dropped. Class-weighted loss added to address severe imbalance.

### Tasks
- [x] Implement `src/models/unet.py` — standard 2D U-Net:
  - Encoder: 4 resolution levels, channels [32, 64, 128, 256], stride-2 conv downsampling
  - Decoder: bilinear upsample + conv (lighter than transposed conv)
  - Output: 8-class softmax, input (B,1,256,256)
- [x] Implement `src/losses/segmentation.py`:
  - `DiceLoss`: per-class soft Dice, mean over foreground classes (exclude background)
  - `WeightedCELoss`: inverse-frequency class weights computed from training set label distribution
  - Combined: `0.5 * DiceLoss + 0.5 * WeightedCELoss`
- [x] Compute and save class weights to `data/class_weights_{ct,mr}.pt`
- [x] Implement `scripts/train_baseline.py`:
  - Optimizer: AdamW, lr=3e-4, weight_decay=1e-5
  - LR scheduler: CosineAnnealingLR, T_max=100 epochs
  - Device: MPS with CPU fallback
  - Validation: per-class Dice every 5 epochs; save best checkpoint by mean foreground Dice
  - CSV logging: `results/train_log_baseline_{ct,mr}.csv`
- [x] Save best checkpoints: `checkpoints/baseline_unet_{ct,mr}.pth`
- [x] Confirm per-patient volumetric Dice using `MMWHSPatientDataset`

### Expected Outcome
| Metric | CT | MRI |
|---|---|---|
| Mean foreground Dice | ≥ 0.75 | ≥ 0.70 |
| Training time (100 ep) | ~2h | ~1h |
| Inference (1 patient) | < 5s | < 3s |

### Actual Outcome

| Metric | CT | MRI | vs. Target |
|---|---|---|---|
| Best val mean fg Dice | **0.836** (ep 20) | **0.825** (ep 80) | ✅ Exceeded |
| Training time (100 ep) | ~2.4h | ~1.2h | ✅ On target |
| Epoch time | ~85s | ~42s | ✅ |

**CT test patients:**
| Patient | Mean Fg Dice | LV | RV | LA | RA | Myo | Aorta | PA |
|---|---|---|---|---|---|---|---|---|
| ct_1019 | 0.807 | 0.855 | 0.871 | 0.674 | 0.886 | 0.857 | 0.861 | 0.647 |
| ct_1020 | 0.927 | 0.894 | 0.947 | 0.936 | 0.892 | 0.919 | 0.972 | 0.928 |

**MRI test patients:**
| Patient | Mean Fg Dice | LV | RV | LA | RA | Myo | Aorta | PA |
|---|---|---|---|---|---|---|---|---|
| mr_1019 | 0.872 | 0.845 | 0.871 | 0.949 | 0.916 | 0.940 | 0.845 | 0.738 |
| mr_1020 | 0.840 | 0.870 | 0.875 | 0.946 | 0.897 | 0.921 | 0.641 | 0.728 |

**Checkpoints:** `checkpoints/baseline_unet_{ct,mr}.pth`
**Logs:** `results/train_log_baseline_{ct,mr}.csv`

---

### Stage 1 Retrospective — Deviations from Plan

#### ❶ Data loading was the bottleneck, not GPU compute [MAJOR — fixed]
- **Expected:** ~85s/epoch dominated by MPS computation
- **Actual (before fix):** 272s/epoch — sequential NPZ file I/O was the bottleneck (~1.28s/batch)
- **Fix:** Added `preload=True` to `MMWHSSliceDataset` — loads entire split into RAM at init (4.5s one-time cost). Post-fix: 4.2ms/batch data, ~85s/epoch.
- **Impact on later stages:** All future datasets and training scripts should use `preload=True` by default.

#### ❷ Best CT checkpoint at epoch 20, not epoch 100 [MODERATE]
- **Expected:** Steady improvement to epoch 100
- **Actual:** CT val Dice peaked at **0.8364** at epoch 20, then plateaued 0.81–0.83 for remaining 80 epochs. MRI continued improving, peaking at **0.8246** at epoch 80.
- **Implication:** CT training could be reduced to ~30 epochs. Consider `ReduceLROnPlateau` over `CosineAnnealingLR` for faster adaptation.
- **Impact on Stage 7:** ProtoSegNet training should use early stopping with patience=15, not fixed 100 epochs.

#### ❸ PA (Pulmonary Artery) is the hardest structure [NOTABLE]
- CT PA Dice: 0.647 / 0.928 (very high inter-patient variance)
- MRI PA Dice: 0.738 / 0.728 (consistently lower)
- LA Dice for ct_1019 was only 0.674 despite other structures performing well
- **Implication:** PA and LA will require more prototypes or a larger prototype bank. Diversity loss may help push prototypes to cover more PA shape variants.

#### ❹ MRI achieves comparable Dice to CT despite fewer training slices [POSITIVE]
- MRI: 1738 train slices → test Dice 0.872 / 0.840
- CT: 3389 train slices (2× more) → test Dice 0.807 / 0.927
- **Implication:** The dataset quality is high. Cross-modal warm-up (CT→MRI) is likely not needed.

---

## Stage 2 — Multi-Scale Hierarchical Backbone ⬜

### Goal
Replace U-Net encoder with a 2D backbone that exposes 4 feature maps `Z_l` at different spatial resolutions for prototype attachment.

> **Revised from original:** 3D → 2D spatial dimensions. MONAI Swin-UNETR dropped. Spatial maps are now (H/s × W/s) instead of (D/s × H/s × W/s).

### Architecture Specification
Input: `(B, 1, 256, 256)`

| Level | Stride | Output Spatial | Channels | Anatomical Role |
|---|---|---|---|---|
| l=1 | ×2 | 128×128 | 32 | Fine texture, pixel-level boundary |
| l=2 | ×4 | 64×64 | 64 | Local edge, inter-structure border |
| l=3 | ×8 | 32×32 | 128 | Structure-level context |
| l=4 | ×16 | 16×16 | 256 | Global cardiac layout |

### Tasks
- [ ] Implement `src/models/encoder.py` — `HierarchicalEncoder2D`:
  - 4 encoder blocks, each: `[Conv2d(stride=2) → BN → ReLU → ResBlock]`
  - `forward(x)` → returns `{1: Z_1, 2: Z_2, 3: Z_3, 4: Z_4}` dict
- [ ] Unit test: assert output shapes for input `(2, 1, 256, 256)` match spec above
- [ ] RAM profile: measure peak RAM during forward+backward with batch=16 using `tracemalloc`
- [ ] Verify peak RAM ≤ 4GB for batch=16 (budget is generous for 2D)
- [ ] Smoke-test: plug backbone into Stage 1 training loop, confirm Dice not regressed

### Expected Outcome
Module `src/models/encoder.py` exports `HierarchicalEncoder2D`.
Forward pass (batch=16, 256×256) < 1GB peak RAM.
All 4 `Z_l` feature maps accessible for prototype attachment.

---

## Stage 3 — Prototype Layer ⬜

### Goal
Implement learnable prototype matrices `P_l = {p_{l,k,m}}` and 2D similarity heatmap computation.

> **Revised from original:** Heatmap shape is now `(B, K, M, H_l, W_l)` (2D) instead of `(B, K, M, D_l, H_l, W_l)`. Prototype projection is simpler — no 3D patch search needed. Background class (k=0) prototypes excluded from diversity loss.

### Architecture Specification
- K = 8 classes; prototype counts per level: `M = {l1: 4, l2: 3, l3: 2, l4: 2}`
- Total: 8 × 11 = 88 prototype vectors. Each `p_{l,k,m} ∈ ℝ^{C_l}`
- Similarity: `S = log(cosine_sim(z, p) + 1)` → range [0, log(2)]

### Tasks
- [ ] Implement `src/models/prototype_layer.py` — `PrototypeLayer(n_classes, n_protos, feature_dim)`:
  - `self.prototypes = nn.Parameter(torch.randn(n_classes, n_protos, feature_dim))`
  - `forward(Z_l)` → heatmap `A` shape `(B, K, M, H_l, W_l)`
  - Use `einsum` for efficient batched cosine similarity across all spatial positions
- [ ] Implement `SoftMaskModule`: aggregate heatmap per class → mask → multiply with `Z_l`
- [ ] Unit tests:
  - Output heatmap shape matches `(B, K, M, H_l, W_l)`
  - Similarity scores in [0, log(2)]
  - Gradients flow to `self.prototypes`
  - Soft mask output shape matches input feature map shape
- [ ] Implement `PrototypeProjection` (run every N epochs, CPU-safe):
  - Extract encoder features for all training slices → build feature bank
  - For each prototype, find nearest neighbour in feature bank
  - Replace with real feature vector; record (slice filename, spatial position) for visualization
  - Save to `checkpoints/projected_prototypes.pt`
  - **Runtime target:** < 2 min on CPU (2D feature bank is much smaller than 3D)

### Expected Outcome
`src/models/prototype_layer.py` with `PrototypeLayer` and `SoftMaskModule`.
Prototype projection < 2 min.
Heatmaps visually activate near correct anatomy in 2D overlay.

---

## Stage 4 — Diversity Loss (Jeffrey's Divergence) ⬜

### Goal
Prevent prototype collapse by penalizing intra-class prototype similarity via Jeffrey's Divergence.

> **No structural changes from original** — this loss operates on heatmap distributions and is dimension-agnostic. The background class (k=0) should be excluded from diversity enforcement since background prototypes don't need anatomical separation.

### Tasks
- [ ] Implement `src/losses/diversity_loss.py` — `prototype_diversity_loss(A_dict, exclude_bg=True)`:
  ```
  L_div = Σ_l Σ_{k≠0} Σ_{m≠n} [ 1 / (D_J(A_{l,k,m} || A_{l,k,n}) + eps) ]
  where D_J(P||Q) = KL(P||Q) + KL(Q||P)
  ```
  - Flatten spatial dims before softmax normalization to probability distribution
  - Manual KL: `(p * (p.log() - q.log())).sum()` for numerical stability
  - `eps = 1e-8`
- [ ] Unit tests:
  - Identical heatmaps → high loss (penalized)
  - Orthogonal heatmaps → low loss (rewarded)
  - Background class excluded: loss unchanged when bg heatmaps are identical
  - Gradient flows to prototype parameters
- [ ] Implement combined: `L_total = 0.5*L_dice + 0.5*L_wce + lambda_div * L_div`
- [ ] Hyperparameter: `lambda_div = 0.01` (start), sweep `[0.001, 0.01, 0.1]` in Stage 7

### Expected Outcome
`src/losses/diversity_loss.py`.
Prototype pairwise cosine similarity (foreground classes) < 0.5 after convergence.
No prototype collapse observed in heatmap visualizations.

---

## Stage 5 — Full Prototype Decoder Pipeline ⬜

### Goal
Connect encoder → prototypes → decoder into a single end-to-end `ProtoSegNet` model.

> **Revised from original:** 3D → 2D decoder. `1×1×1` Conv → `1×1` Conv. Training phases shortened (faster per epoch). OOM risk essentially eliminated for 2D.

### Architecture
```
Input X (B, 1, 256, 256)
  → HierarchicalEncoder2D → {Z_1(B,32,128,128), Z_2(B,64,64,64),
                               Z_3(B,128,32,32), Z_4(B,256,16,16)}
  → PrototypeLayer (per level) → {A_{l,k,m}} heatmaps
  → SoftMask: Z_l * agg(A_l) → masked features
  → 2D Decoder (bilinear upsample + skip connections from masked features)
  → 1×1 Conv: linear aggregation of upsampled prototype activations
  → Softmax → y_hat (B, K, 256, 256)
```

### Tasks
- [ ] Implement `src/models/proto_seg_net.py` — `ProtoSegNet`:
  - `forward(x)` → `(logits, heatmaps_dict)` where `heatmaps_dict[l]` has shape `(B,K,M,H_l,W_l)`
  - Decoder: 4 upsample blocks, each `[Upsample(×2) → concat skip → Conv → BN → ReLU]`
  - Final `1×1 Conv` maps concatenated prototype activations to `K` logits per pixel
- [ ] Implement 3-phase training in `scripts/train.py`:
  - **Phase A** (epochs 1–20): backbone + decoder only; prototypes frozen
  - **Phase B** (epochs 21–80): all params unfrozen; full `L_total`; projection every 10 epochs
  - **Phase C** (epochs 81–100): backbone + prototypes frozen; fine-tune decoder only
- [ ] RAM profile: batch=16, full forward+backward ≤ 4GB
- [ ] Save training curve: `results/train_curve_proto_{ct,mr}.csv`

### Expected Outcome
`ProtoSegNet` trains end-to-end without OOM.
Segmentation Dice within **3%** of 2D U-Net baseline.
`heatmaps_dict` returned at inference for downstream XAI metric computation.

---

## Stage 6 — XAI Metrics Modules ⬜

### Goal
Implement the 4 quantitative XAI evaluation modules: AP, IDS, Faithfulness, Stability.

> **Revised from original:** All metrics now operate per-slice (2D). "Volume" = patient's slice stack. IDS re-inference is per-slice (no sliding window needed). Faithfulness sampling raised to N=2000 pixels per slice (feasible in 2D). Runtime target reduced from 2h to 30 min.
>
> **Small test set caveat:** Only 2 patients (CT) and 2 patients (MRI) in test split. Report results per-patient individually; do not over-interpret aggregated means.

### 6.1 Activation Precision (AP) — unchanged
```
AP_k = |M_k ∩ G_k| / |M_k|    (per 2D slice, averaged over patient's slices)
where M_k = I(A_k > 95th percentile of A_k)
```
- [ ] Implement `src/metrics/activation_precision.py`
- [ ] Aggregate prototype heatmaps per class: `A_k = max_m(A_{l,k,m})` over levels and protos
- [ ] Compute per-slice AP, then average per-patient
- [ ] Unit test: perfect heatmap → AP=1.0; uniform heatmap → AP ≈ foreground fraction

### 6.2 Incremental Deletion Score (IDS) — simplified
```
IDS = AUC of mean_Dice(t) as top-t% activated pixels zeroed, t ∈ {5,10,...,100}%
```
- [ ] Implement `src/metrics/incremental_deletion.py`
- [ ] Per-slice: sort pixels by activation, iteratively zero and re-infer (no sliding window needed for 2D)
- [ ] Plot deletion curve; save to `results/ids_curve_{modality}.png`
- [ ] Run on all test slices (≤ 484 CT, ≤ 236 MRI); manageable in 2D

### 6.3 Faithfulness Correlation — more feasible in 2D
```
Faithfulness = Pearson(E_i, Δy_hat_i)  over N=2000 sampled pixels per slice
```
- [ ] Implement `src/metrics/faithfulness.py`
- [ ] Sample N=2000 pixels per slice; zero each pixel in input, re-infer, record Δy_hat
- [ ] Note: perturbation is in input space (pixel zeroing), not feature space, for simplicity

### 6.4 Lipschitz Stability — unchanged logic
```
Stability = max_{X' ∈ N_eps(X)} [ ||Φ(X) - Φ(X')||_2 / ||X - X'||_2 ]
```
- [ ] Implement `src/metrics/stability.py`
- [ ] N=20 Gaussian perturbations at σ=0.05 (normalized intensity scale)
- [ ] Run per-slice, report mean ± std per patient

### Expected Outcome
All 4 metric modules importable from `src/metrics/`.
`scripts/evaluate_xai.py` prints a per-patient summary table.
Full XAI evaluation (all 4 metrics, both modalities) completes in **< 30 min**.

---

## Stage 7 — Training & Full Benchmark Evaluation ⬜

### Goal
Train ProtoSegNet on CT and MRI separately, evaluate segmentation + XAI metrics.
Also add a lightweight **3D volume reconstruction evaluation** to align with the research title "3D cardiac segmentation" without incurring significant compute cost.

> **Revised from original:** Batch size raised (16–32 feasible in 2D). Epoch count reduced. No 5-fold CV. Dice targets adjusted to 2D baseline.
>
> **Added (post-review):** 2D slice predictions are stacked along Z-axis per patient to compute **3D Dice and ASSD**, closing the gap between the "2D training, 3D claim" discrepancy. XAI metrics remain 2D (computationally justified).

### Training Configuration
```python
device = "mps"              # Apple Silicon
batch_size = 16             # revised up from 2 (2D is cheap)
input_size = (256, 256)
max_epochs = 100            # Phase A:20 + Phase B:60 + Phase C:20
optimizer = AdamW(lr=3e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(T_max=100)
lambda_div = 0.01           # sweep [0.001, 0.01, 0.1] in ablation
projection_interval = 10    # epochs between prototype projections
```

### Tasks
- [ ] Run `scripts/train.py --modality ct` → `checkpoints/proto_seg_ct.pth`
- [ ] Run `scripts/train.py --modality mr` → `checkpoints/proto_seg_mr.pth`
- [ ] Evaluate **2D segmentation**: per-class Dice + mean foreground Dice on test patients
- [ ] **3D volume reconstruction evaluation** (`scripts/eval_3d.py`):
  - Stack per-patient 2D predictions along Z-axis: `pred_vol[s, h, w] = argmax(logits_s)`
  - Compute **3D Dice** per class (volumetric TP/FP/FN across stacked slices)
  - Compute **ASSD** (Average Symmetric Surface Distance) using `scipy.ndimage` distance transform
  - Save per-patient 3D predictions as NIfTI (`.nii.gz`) using `nibabel` for ITK-SNAP review
- [ ] Run `scripts/evaluate_xai.py --modality {ct|mr}` (2D per-slice XAI metrics)
- [ ] Generate comparison table vs. Baseline 2D U-Net

### Expected Results Table

| Model | Modality | 2D Fg Dice | 3D Dice | ASSD (mm) | AP | IDS | Faithfulness | Stability |
|---|---|---|---|---|---|---|---|---|
| Baseline 2D U-Net | CT | ≥0.75 | ≥0.75 | ≤5.0 | N/A | N/A | N/A | N/A |
| ProtoSegNet | CT | ≥0.72 | ≥0.72 | ≤6.0 | ≥0.70 | ≤0.45 | ≥0.55 | ≤0.20 |
| Baseline 2D U-Net | MRI | ≥0.70 | ≥0.70 | ≤6.0 | N/A | N/A | N/A | N/A |
| ProtoSegNet | MRI | ≥0.67 | ≥0.67 | ≤7.0 | ≥0.65 | ≤0.50 | ≥0.50 | ≤0.25 |

> Note: Results are over 2 test patients per modality — treat as case studies, not statistical estimates.
> 3D Dice ≈ 2D Fg Dice for slice-independent inference (no slice-context model); ASSD captures boundary quality.

---

## Stage 8 — Ablation Studies & Visualization ⬜

### Goal
Quantify the contribution of each design choice; produce visualizations for the prototype atlas and 3D cardiac renders.

> **Revised from original:** Prototype atlas is 2D patch grid. Ablation training reduced to 50 epochs.
>
> **Added (post-review):** Export per-patient 3D NIfTI predictions for 3D Slicer / ITK-SNAP rendering, providing the "3D cardiac segmentation" visual narrative expected in the research paper.

### Ablation Variants
| Variant | Change | Purpose |
|---|---|---|
| A: No multi-scale | Single prototype level (l=4 only) | Validate multi-scale benefit on AP |
| B: No diversity loss | λ_div = 0 | Show prototype collapse without L_div |
| C: No soft mask | Skip SoftMask module | Validate spatial focusing benefit |
| Full model | All components | Best configuration |

### Tasks
- [ ] Train variants A, B, C (50 epochs, CT only, same seed)
- [ ] Compare AP and IDS across all 4 variants → `results/ablation_table.csv`
- [ ] Implement `scripts/visualize_prototypes.py`:
  - Load `projected_prototypes.pt`; retrieve source slice + spatial position per prototype
  - Crop 64×64 patch centred on projection; overlay similarity heatmap
  - Arrange as grid: rows = classes, cols = prototypes per level
  - Save to `results/prototype_atlas/{modality}_level{l}.png`
- [ ] Implement `scripts/visualize_segmentation.py`:
  - For each test patient: 4-panel per slice → input / GT / prediction / heatmap overlay
  - Save representative slices (1 per class maximally activated)
- [ ] **3D visualization export** (`scripts/export_nifti.py`):
  - Stack 2D predictions → 3D volume (S, H, W)
  - Save as `results/nifti/{patient}_pred.nii.gz` and `_gt.nii.gz` using `nibabel`
  - These can be opened in 3D Slicer or ITK-SNAP for publication-quality renders
  - No extra model inference needed — reuse predictions from Stage 7
- [ ] Prototype collapse check: mean pairwise cosine similarity within each class
  - Variant B expected > 0.8 (collapsed), Full model < 0.5 (diverse)

### Expected Outcome
Ablation table showing multi-scale AP gain ≥ +3%.
2D prototype atlas showing anatomically interpretable patches per cardiac structure.
3D NIfTI exports ready for ITK-SNAP rendering (fulfils "3D segmentation" research claim).
Visualization scripts complete in < 2 min per patient.

---

## File Structure (Target — updated post Stage 0)

```
cardiac-proto-segmentation/
├── src/
│   ├── models/
│   │   ├── unet.py             # ✅ Baseline 2D U-Net (Stage 1)
│   │   ├── encoder.py          # HierarchicalEncoder2D (Stage 2)
│   │   ├── prototype_layer.py  # PrototypeLayer, SoftMaskModule (Stage 3)
│   │   └── proto_seg_net.py    # Full ProtoSegNet model (Stage 5)
│   ├── losses/
│   │   ├── segmentation.py     # ✅ Dice + WeightedCE (Stage 1)
│   │   └── diversity_loss.py   # Jeffrey's Divergence (Stage 4)
│   ├── metrics/
│   │   ├── dice.py             # ✅ per-class Dice + mean_fg_dice (Stage 1)
│   │   ├── activation_precision.py  (Stage 6)
│   │   ├── incremental_deletion.py  (Stage 6)
│   │   ├── faithfulness.py          (Stage 6)
│   │   └── stability.py             (Stage 6)
│   └── data/
│       └── mmwhs_dataset.py    # ✅ MMWHSSliceDataset (preload) + MMWHSPatientDataset
├── notebooks/                  # Jupyter notebooks for interactive experiments
│   ├── 00_data_exploration.ipynb   # Stage 0 — EDA, label distribution, sample viz
│   ├── 01_baseline_analysis.ipynb  # Stage 1 — training curves, per-class Dice analysis
│   ├── 02_backbone_debug.ipynb     # Stage 2 — feature map visualization per level
│   ├── 03_prototype_viz.ipynb      # Stage 3 — heatmap overlay, projection analysis
│   ├── 06_xai_metrics.ipynb        # Stage 6 — AP/IDS/Faithfulness/Stability exploration
│   └── 08_ablation_results.ipynb   # Stage 8 — ablation comparison tables & figures
├── scripts/
│   ├── explore_data.py         # ✅ Done
│   ├── train_baseline.py       # ✅ Stage 1 (training)
│   ├── train.py                # Stage 7 (ProtoSegNet)
│   ├── eval_3d.py              # Stage 7 — 3D Dice + ASSD + NIfTI export (new)
│   ├── evaluate_xai.py         # Stage 6
│   ├── visualize_prototypes.py # Stage 8
│   ├── visualize_segmentation.py  # Stage 8
│   └── export_nifti.py         # Stage 8 — 3D Slicer renders (new)
├── checkpoints/
│   ├── baseline_unet_ct.pth    # Stage 1 output
│   └── baseline_unet_mr.pth    # Stage 1 output
├── results/
│   ├── prototype_atlas/
│   ├── nifti/                  # Stage 7/8 — 3D exports for ITK-SNAP
│   ├── sample_ct.png           # ✅ Done
│   ├── sample_mr.png           # ✅ Done
│   └── train_log_baseline_*.csv  # Stage 1 output
├── data/
│   ├── pack/processed_data/    # ✅ source data (2D NPZ slices)
│   ├── splits.json             # ✅ Done
│   └── class_weights_*.pt      # ✅ Stage 1 output
├── report/
│   └── progress/
│       └── execution-plan.md   # this file
└── pyproject.toml              # ✅ (torch, numpy, scipy, matplotlib, sklearn, tqdm, nibabel)
```

### Notebooks Convention
Each notebook is self-contained and importable from the project root (adds `src/` to path). They serve as interactive experiment logs — results produced here should eventually be hardened into `scripts/` for reproducibility. Notebooks are numbered to match their Stage.

---

## Risk Register (updated post Stage 1)

| Risk | Likelihood | Status / Mitigation |
|---|---|---|
| ~~OOM during IDS (3D re-inference ×20)~~ | ~~High~~ | ✅ **Resolved** — 2D per-slice inference is cheap |
| ~~Class imbalance causes BG-only collapse~~ | ~~High~~ | ✅ **Resolved** — Weighted CE + Dice loss confirmed working; no collapse observed |
| ~~Low MRI Dice (small training set)~~ | ~~Medium~~ | ✅ **Resolved** — MRI achieved 0.825 val Dice comparable to CT; no warm-up needed |
| ~~File I/O bottleneck (272s/epoch)~~ | ~~High~~ | ✅ **Resolved** — `preload=True` brings to 85s/epoch |
| Prototype collapse despite L_div | Medium | Increase λ_div; monitor pairwise cosine similarity during Stage 7 training |
| MPS backend crashes on complex ops | Medium | `PYTORCH_MPS_FALLBACK=1`; test `einsum` similarity ops on MPS before Stage 5 |
| High variance in XAI metrics (only 2 test patients) | **High** | Report per-patient results; use prototype atlas as qualitative evidence |
| PA / LA segmentation remains hard for ProtoSegNet | Medium | Allocate M=4 prototypes for PA/LA; monitor these classes separately in Stage 7 |
| Prototype projection finds irrelevant BG patches | Medium | Filter projections: only accept patches where class k is present in GT label |
| CT best epoch at ep 20 — ProtoSegNet may also converge early | Medium | Use early stopping (patience=15) in Stage 7, not fixed 100 epochs |

---

## Success Criteria (updated post Stage 1)

**Segmentation**
- [x] Baseline 2D U-Net achieves mean foreground Dice ≥ **0.75** (CT) / **0.70** (MRI) — CT: **0.836**, MRI: **0.825** ✅
- [ ] ProtoSegNet achieves within **3% Dice** of baseline (≥0.81 CT / ≥0.80 MRI, updated to match actual baseline)

**XAI Metrics** *(over 2 test patients per modality — interpret as case studies)*
- [ ] Mean AP ≥ **0.70** across all 7 cardiac structures (CT); ≥ **0.65** (MRI)
- [ ] IDS ≤ **0.45** (CT); ≤ **0.50** (MRI)
- [ ] Faithfulness correlation ≥ **0.55** (Pearson)
- [ ] Stability score ≤ **0.20**

**Training & Code Quality**
- [x] No OOM on MacBook 48GB RAM at any stage (confirmed through Stage 1) ✅
- [x] Class imbalance handled — no collapse to background-only predictions ✅
- [ ] Full pipeline reproducible with `scripts/train.py --modality {ct|mr}`

**Interpretability**
- [ ] Prototype atlas shows anatomically coherent 2D patches per cardiac structure per level
- [ ] Ablation confirms multi-scale ≥ +3% AP over single-scale
- [ ] Prototype cosine similarity < 0.5 within same class (diversity confirmed)
