# When Skip Connections Bypass Prototypes

## What This Research Does

This project investigates whether prototype-based segmentation networks provide genuine structural interpretability when applied to cardiac segmentation on the MM-WHS CT dataset. The central finding is a fundamental **Dice–interpretability trade-off**: U-Net skip connections — required for boundary-precise dense prediction — create a bypass pathway that renders prototype heatmaps causally decorative.

A controlled 2×2 ablation crosses decoder type (with skip / without skip) against prototype level (16×16 only / 32×32+16×16). Three mechanistic analyses characterise the bypass:

1. **Gradient attribution** (`bypass_ratio`): 78% of decoder gradient flows through skip/encoder features rather than prototype heatmaps in the skip-16 model.
2. **Spatial precision**: Skip-model heatmaps activate at only 25–32% of the ground-truth overlap achieved by no-skip heatmaps.
3. **GT-guided counterfactual**: skip-32+16 learned prototypes reach only 20% of the localization quality their own feature space could support (AP = 0.078 vs. ceiling 0.389).

The core result: **no architecture simultaneously achieves Dice > 0.80 and AP > 0.25**.

| Model | Dice | AP | Structural Guarantee |
|-------|------|----|----------------------|
| skip-32+16 (skip, 32×32+16×16) | **0.821** | 0.051 | No (bypass_ratio = 0.465) |
| skip-16 (skip, 16×16) | 0.810 | 0.057 | No (bypass_ratio = 0.778) |
| noskip-16 (no-skip, 16×16) | 0.606 | **0.312** | Yes (bypass = 0 by construction) |
| noskip-32+16 (no-skip, 32×32+16×16) | 0.559 | 0.301 | Yes (bypass = 0 by construction) |

Full report: `report/v11/report-v11.md`

---

## Experiment Notebooks

### Model Training

| Notebook | Description |
|----------|-------------|
| `notebooks/00_data_exploration.ipynb` | Dataset exploration (MM-WHS NPZ slices, class statistics) |
| `notebooks/01_baseline_analysis.ipynb` | Baseline 2D U-Net training and evaluation |
| `notebooks/05_proto_seg_training.ipynb` | Skip-connected ProtoSegNet training (skip-16, skip-32+16) |
| `notebooks/27_two_stage_warmstart.ipynb` | Warmstart training strategy for skip-32+16 |
| `notebooks/29_warmstart_l3l4.ipynb` | L3+L4 warmstart fine-tuning |
| `notebooks/30_fixed_stage1.ipynb` | Fixed-stage training experiments |
| `notebooks/31_ablation_level_selection.ipynb` | Single-level ablation to select prototype levels (L1–L4) |
| `notebooks/35_proto_v2_training.ipynb` | No-skip (heatmap-only) ProtoSegNet training (noskip-16, noskip-32+16) |

### XAI Evaluation (v10 — 2×2 Ablation Metrics)

| Notebook | Description |
|----------|-------------|
| `notebooks/37_xai_stage29.ipynb` | skip-32+16: faithfulness + stability |
| `notebooks/38_g1b_stage29_projection.ipynb` | skip-32+16: AP + purity (fresh prototype projection) |
| `notebooks/39_g3_patch_faithfulness.ipynb` | Patch-level faithfulness for all 4 models |
| `notebooks/40_g4_stage34b_xai.ipynb` | noskip-32+16: full XAI evaluation |
| `notebooks/41_g5_stage8a_xai.ipynb` | noskip-16: purity + faithfulness |
| `notebooks/42_g7_baseline_unet_ap.ipynb` | Baseline U-Net AP upper bound |
| `notebooks/43_v10_figures.ipynb` | v10 summary figures |
| `notebooks/44_v10_visual_figures.ipynb` | v10 visual/qualitative figures |

### Mechanistic Analysis (v11 — Bypass Characterisation)

| Notebook | Description |
|----------|-------------|
| `notebooks/43_g11_gradient_attribution.ipynb` | Gradient attribution: bypass_ratio for skip-16 and skip-32+16 |
| `notebooks/44_g13_spatial_misalignment.ipynb` | Spatial precision: heatmap–GT overlap across all 4 models |
| `notebooks/46_v11_counterfactual_gt_guided.ipynb` | GT-guided counterfactual: AP_learned vs. AP_GT ceiling |
