# Execution Plan v2: Hard-Mask ProtoSegNet

**Project:** Interpretable 2D/3D Cardiac Segmentation with Quantifiable XAI
**Dataset:** MM-WHS (60 CT + 60 MRI, 7 cardiac structures)
**Hardware:** MacBook 48GB RAM (Apple Silicon MPS backend)
**Last Updated:** 2026-03-14
**Stage 9 Completed:** 2026-03-14
**Stage 10 Completed:** 2026-03-14
**Status:** ✅ Complete — see `report/v3/execution-plan-v3.md` for next steps

---

## Context: Why v2?

v1 delivered good segmentation (CT 3D Dice 0.843, MRI 0.805) but failed XAI targets:

| Metric | v1 Best (CT) | Target | Gap |
|--------|-------------|--------|-----|
| AP | 0.102 | ≥ 0.70 | −0.60 |
| Faithfulness | 0.059 | ≥ 0.55 | −0.49 |
| Stability | 3.00 | ≤ 0.20 | ×15 |
| IDS | 0.007 | ≤ 0.45 | ✅ |

**Root cause (confirmed in Stage 8 ablation):** The soft-mask architecture allows the decoder to bypass prototype heatmaps via skip connections. Prototypes become redundant for the final prediction, so their heatmaps carry no causal signal. This is structural — it cannot be fixed by adjusting loss weights (λ_push saturation confirmed at 0.5, λ_div stability floor at ~3.0).

**v2 hypothesis:** Replacing soft-mask with a hard spatial gate that blocks the bypass path will force the decoder to rely exclusively on prototype-activated features, directly improving AP and Faithfulness. Stability may also improve as heatmaps become causally load-bearing.

---

## v2 Stage Overview

| Stage | Name | Deliverable | Status |
|---|---|---|---|
| 9 | Hard Mask Module | `HardMaskModule` replacing `SoftMaskModule` | ✅ |
| 10 | CT Retrain (Hard Mask) | `proto_seg_ct_hm2.pth` + XAI eval | ✅ |

All v2 stages complete. Continuation: `report/v3/execution-plan-v3.md`

---

## Stage 9 — Hard Mask Module ⬜

### Goal    

Replace `SoftMaskModule` with `HardMaskModule` that applies a binarised spatial gate, blocking all feature information from decoder skip connections that does not pass through the prototype heatmap threshold.

### Root Cause Recap

```
Current (v1 soft-mask):
  Z_masked = Z_l × W_l          # W_l ∈ (0,1], continuous
  → decoder receives Z_masked + residual Z_l via skip → can ignore W_l

Target (v2 hard-mask):
  mask = (W_l > τ)              # binary spatial gate
  Z_masked = Z_l × mask         # non-activated positions = 0.0
  → decoder has no alternative path to the zeroed features
```

### Design: Straight-Through Estimator (STE) Hard Mask

Binarisation is non-differentiable. We use the **Straight-Through Estimator** (Bengio et al., 2013): the forward pass uses the hard binary mask; the backward pass treats it as the identity (gradient passes through as if no thresholding occurred).

```python
class HardMaskModule(nn.Module):
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.quantile = quantile   # spatial percentile threshold

    def forward(self, Z_l, A_l):
        # A_l: (B, K, M, H, W) — prototype heatmaps
        W = A_l.max(dim=2).values.max(dim=1).values  # (B, H, W)
        W = W.unsqueeze(1)                            # (B, 1, H, W)

        # Compute threshold per sample
        tau = W.flatten(1).quantile(self.quantile, dim=1)
        tau = tau.view(-1, 1, 1, 1)

        # Hard mask (forward)
        mask_hard = (W >= tau).float()

        # STE: gradient flows through soft W, value uses hard mask
        mask_ste = mask_hard + (W - W.detach())

        return Z_l * mask_ste
```

The `quantile` hyperparameter controls sparsity: 0.5 zeros out the bottom 50% of spatial locations. Start at 0.5 and sweep [0.3, 0.5, 0.7] in Stage 12 ablation.

### Tasks

- [x] Implement `HardMaskModule` in `src/models/prototype_layer.py` ✅
- [x] Add `--hard-mask` and `--mask-quantile` CLI flags to `train_proto.py` ✅
- [x] Update `ProtoSegNet.__init__` to accept `hard_mask: bool = False, mask_quantile: float = 0.5` ✅
- [x] Update `evaluate_xai.py` to auto-read `hard_mask` / `mask_quantile` from checkpoint config ✅
- [x] Unit tests (`scripts/test_hard_mask.py`) — 8/8 passed ✅:
  - Hard-masked output has exact zeros at bottom-quantile spatial locations ✅
  - Mask is binary (0 or 1) in forward, continuous in backward ✅
  - Gradient flows to prototype parameters through STE ✅
  - End-to-end shapes unchanged: logits (B,8,256,256) ✅
  - Checkpoint round-trip (hard_mask key persists) ✅
  - quantile=0.0 → no zeroing ✅
  - quantile=0.99 → >95% zeroed ✅
  - Hard mask ⊆ soft mask activations ✅
- [x] Smoke-test: 5-step training on CT, loss decreases (1.569→1.460), no NaN ✅

### Actual Outcome

`HardMaskModule` added to `src/models/prototype_layer.py`. `ProtoSegNet` selects between
`SoftMaskModule` (default) and `HardMaskModule` via `hard_mask=True`. All existing
v1 checkpoints load unchanged (backward compatible via `.get()` defaults). No parameter
count change — `HardMaskModule` has no learnable parameters.

---

## Stage 10 — CT Retrain with Hard Mask ✅ COMPLETE

### Goal

Retrain ProtoSegNet on CT using the hard-mask module with the best v1 hyperparameter configuration (L2 similarity, λ_div=0.001, λ_push=0.5, λ_pull=0.25), and evaluate all XAI metrics.

### Training Configuration

```python
# Unchanged from v1 best (proto_seg_ct_l2)
similarity       = "l2"
lambda_div       = 0.001
lambda_push      = 0.5
lambda_pull      = 0.25
mask_quantile    = 0.5
max_epochs       = 100          # Phase A:20 + Phase B:60 + Phase C:20
batch_size       = 16
lr               = 3e-4
projection_interval = 10
# Phase A: hard_mask_active=False (soft-mask fallback)
# Phase B onward: hard_mask_active=True (binary gate activated)
```

### Tasks

- [x] Run `_hm` (first attempt — Phase A bug): Dice 0.629 ❌ → diagnosed and fixed ✅
- [x] Fix: add `hard_mask_active` flag; Phase A uses soft-mask fallback; hard gate activates at Phase B start ✅
- [x] Run `_hm2` (fixed): `proto_seg_ct_hm2.pth` — best val Dice 0.8146 at ep 55 ✅
- [x] `scripts/eval_3d.py` → `results/v2/eval_3d_ct_hm2.txt` ✅
- [x] `scripts/evaluate_xai.py` → `results/v2/xai_proto_seg_ct_hm2.txt` ✅

### Actual Training Results

| Run | Suffix | Phase A Bug | Best Val Dice | Best Epoch |
|-----|--------|------------|--------------|-----------|
| First attempt | `_hm` | ❌ Hard gate active in Phase A | 0.7861 | 70 |
| Fixed retrain  | `_hm2` | ✅ Soft fallback in Phase A | **0.8146** | 55 |

### Actual Segmentation Results — `_hm2`

| Structure | 3D Dice | ASSD (mm) |
|-----------|---------|-----------|
| LV        | 0.839   | 2.16      |
| RV        | 0.913   | 2.92      |
| LA        | 0.846   | 4.17      |
| RA        | 0.878   | 4.14      |
| Myocardium| 0.856   | 3.41      |
| Aorta     | 0.806   | 2.60      |
| PA        | 0.703   | 3.87      |
| **Mean fg** | **0.834** | **3.33** |

3D Dice ≥ 0.80 ✅ | ASSD ≤ 6.0 mm ✅ | vs v1 _l2: −0.009 Dice (−1.1%), −0.39 mm ASSD

### Actual XAI Results — `_hm2`

| Metric | v1 _l2 (soft-mask) | v2 _hm2 (hard-mask) | Change |
|--------|-------------------|---------------------|--------|
| **AP** | 0.102 | **0.064** | ↓ −37% ❌ |
| **IDS** | 0.007 | **0.027** | ↓ worse ❌ |
| **Faithfulness** | 0.059 | **0.015** | ↓ −75% ❌ |
| **Stability** | 3.00 | **2.14** | ↓ +29% ✅ |

Per-patient detail:

| Patient | AP | IDS | Faithfulness | Stability |
|---------|-----|-----|-------------|-----------|
| ct_1019 | 0.009 | 0.036 | 0.012 | 2.18 |
| ct_1020 | 0.120 | 0.017 | 0.018 | 2.10 |
| **Mean** | **0.064** | **0.027** | **0.015** | **2.14** |

AP gate (≥ 0.25) not met. Hard mask hypothesis **refuted**: all XAI metrics regressed except Stability.

---

### Stage 10 Retrospective

#### ❶ Phase A hard-mask causes training collapse [BUG — fixed in `_hm2`]

- **Symptom:** `_hm` best val Dice 0.786, 3D Dice **0.629**, ASSD **36.6 mm** — LA Dice 0.060, RV Dice 0.139.
- **Root cause:** During Phase A, prototype parameters are frozen at random initialisation. L2 similarity scores are near-uniform at ~0.33 everywhere. With quantile=0.5, the hard mask becomes a **random Bernoulli(0.5) gate** that stochastically zeroes half the feature map at every step. The decoder learns on randomly masked inputs, then Phase B activates the true prototype-guided gate, catastrophically disrupting the learned decoder representations.
- **Fix:** `hard_mask_active` flag (default `False`) set to `True` only at Phase A→B transition in `train_proto.py`. Phase A always uses the soft-mask fallback (`_soft_mask_fallback`), regardless of `hard_mask=True`. Code change: `proto_seg_net.py` + `train_proto.py` + `evaluate_xai.py` + `eval_3d.py`.

#### ❷ Hard mask hypothesis refuted — AP and Faithfulness regressed [KEY FINDING]

- **Expected:** Hard gate blocks decoder bypass → prototypes become causally load-bearing → AP ↑, Faithfulness ↑.
- **Actual:** AP 0.102 → 0.064 (−37%), Faithfulness 0.059 → 0.015 (−75%). **Hypothesis wrong.**
- **Root cause analysis:** Prototype heatmaps serve two conflicting roles:
  1. **XAI role:** Precise spatial localisation (AP, Faithfulness require high activation *only* on GT foreground)
  2. **Decoder role:** Feature conditioning (hard mask requires activation *everywhere* the decoder needs features)

  With soft-mask, push-pull loss can enforce role (1) because the decoder still receives attenuated (not zero) features from non-activated locations. With hard-mask, zeroed locations are truly inaccessible — the decoder will fail unless prototypes activate broadly. The model therefore learns **broader, more diffuse activations** to ensure decoder coverage, directly counteracting push-pull's localisation objective. AP falls because diffuse activation ↓ precision; Faithfulness falls because the broader response is less causally specific.

- **Why Stability improved (3.00 → 2.14):** Binary gating creates discrete stable states — small input perturbations that do not flip any mask location leave the heatmap unchanged. This is the one mechanism that works as hypothesised.

#### ❸ Structural conclusion: prototype dual-role conflict is the true ceiling [ACCEPTED LIMITATION]

The XAI ceiling in this architecture is not caused by a single fixable component (soft-mask bypass) but by a **fundamental dual-role conflict**: the same heatmap must simultaneously be (a) a spatial localiser for XAI evaluation and (b) a feature selector for the decoder. These objectives are in tension — improved localisation (narrow activation) reduces decoder feature coverage, which the model compensates for by broadening activation.

This finding is architecturally informative: a solution requires **decoupling** the two roles — a separate decoder pathway that does not depend on prototype heatmaps, with prototype heatmaps used only as XAI readout. See `report/v3/execution-plan-v3.md` for the v3 implementation plan.

---

## File Structure (v2)

```
src/models/prototype_layer.py   # + HardMaskModule (Stage 9)
src/models/proto_seg_net.py     # + hard_mask, mask_quantile, hard_mask_active (Stage 9)
scripts/train_proto.py          # + --hard-mask, --mask-quantile flags (Stage 9)
scripts/evaluate_xai.py         # + auto-reads hard_mask config (Stage 9)
scripts/eval_3d.py              # + reads hard_mask from checkpoint (Stage 9)
scripts/test_hard_mask.py       # Stage 9 unit tests (8/8 passed)

checkpoints/
  proto_seg_ct_hm2.pth          # Stage 10 (fixed hard-mask CT)

results/v2/
  eval_3d_ct_hm2.txt            # Stage 10
  xai_proto_seg_ct_hm2.txt      # Stage 10
  train_log_proto_ct_hm2.txt    # Stage 10

report/v2/
  execution-plan-v2.md          # this file
```

---

## Success Criteria (v2)

**Architecture milestones:**
- [x] HardMaskModule implemented with STE (Stage 9)
- [x] CT hard-mask trained, Phase A bug diagnosed and fixed (Stage 10)

**Segmentation:**
- [x] CT 3D Dice ≥ 0.80 with hard mask: 0.834 ✅

**XAI:**
- [x] Hard-mask Stability improvement confirmed: 3.00 → 2.14 ✅
- [x] Hard-mask hypothesis refuted and root cause diagnosed ✅

---

## Risk Register (v2)

| Risk | Outcome |
|------|---------|
| Hard mask causes Dice collapse in Phase B | ❌ Occurred in `_hm` run; **fixed** via `hard_mask_active` flag |
| STE gradient instability | ✅ Not observed |
| AP improves but Stability stays at 3.0 | ✅ Stability improved (2.14); AP regressed — root cause diagnosed |
