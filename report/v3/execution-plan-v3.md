# Execution Plan v3: Decoupled ProtoSegNet-D

**Project:** Interpretable 2D/3D Cardiac Segmentation with Quantifiable XAI
**Dataset:** MM-WHS (60 CT + 60 MRI, 7 cardiac structures)
**Hardware:** MacBook 48GB RAM (Apple Silicon MPS backend)
**Last Updated:** 2026-03-14
**Preceded by:** `report/v2/execution-plan-v2.md`

---

## Context: Why v3?

v2 tested the hypothesis that a hard spatial gate (HardMaskModule) would force prototypes to become causally load-bearing, improving AP and Faithfulness. The hypothesis was **refuted** (Stage 10):

| Metric | v1 soft-mask | v2 hard-mask | Change |
|--------|-------------|-------------|--------|
| AP | 0.102 | 0.064 | ↓ −37% ❌ |
| Faithfulness | 0.059 | 0.015 | ↓ −75% ❌ |
| Stability | 3.00 | 2.14 | ↑ +29% ✅ |
| 3D Dice | 0.843 | 0.834 | −1.1% ✅ |

**Diagnosed root cause — dual-role conflict:**

```
Current ProtoSegNet (v1 soft-mask, v2 hard-mask):

  Encoder ──► Z_l ──► ProtoLayer ──► A (B,K,M,H,W)
                                      │
                                 mask_module
                                      │
                           Z_masked = Z_l × mask(A)
                                      │
                                  Decoder ──► logits

  Conflict: mask(A) must be wide enough for decoder coverage
            AND narrow enough for XAI precision.
            Push-pull optimises narrowness; decoder penalises it.
```

The prototype heatmap simultaneously serves two conflicting roles:
1. **XAI role:** Precise spatial localisation (AP, Faithfulness require activation *only* on GT foreground)
2. **Decoder role:** Feature conditioning (mask requires activation *everywhere* the decoder needs features)

Under hard masking, zeroed locations are truly inaccessible — the decoder fails unless prototypes activate broadly. The model learns diffuse activations to ensure decoder coverage, counteracting push-pull's localisation objective.

**v3 hypothesis:** Decoupling these two roles — giving the decoder a bypass that does not depend on prototype heatmaps, while keeping a small prototype contribution to the output — allows push-pull to enforce narrow, precise activations without penalisation from decoder coverage needs.

---

## v3 Stage Overview

| Stage | Name | Deliverable | Status |
|---|---|---|---|
| 11 | Decoupled Architecture | `ProtoSegNet-D` + `ProtoHead` + unit tests | ⬜ |
| 12 | CT Train ProtoSegNet-D | `proto_seg_ct_dec.pth` + XAI eval | ⬜ |
| 13 | Report v3 | `report/v2/report-v3.md` (3-arch comparison) | ⬜ |

**Estimated epoch times:** CT ~40s/epoch (ProtoSegNet, MPS, batch=16)

---

## ProtoSegNet-D: Decoupled Architecture

```
Proposed ProtoSegNet-D:

  Encoder ──► Z_l ──────────────────────────────────► Decoder ──► logits_dec
                    └──► ProtoLayer ──► A ──► ProtoHead ──► logits_proto

  Final: logits = logits_dec + λ_xai × logits_proto
```

**Key properties:**
- Decoder receives full encoder features at all levels — no masking, no bypass blocked. Segmentation quality is not at risk.
- Prototype heatmaps A are computed as before and drive the XAI evaluation (AP, Faithfulness, Stability, IDS).
- `ProtoHead` is a lightweight 1×1 conv that aggregates heatmaps into per-class logits — forces prototypes to be class-discriminative without constraining their spatial extent.
- `λ_xai = 0.1`: prototype contribution is small; decoder dominates. Push-pull can enforce precise, narrow activations because decoder coverage is guaranteed independently.
- Faithfulness is measurable: setting `λ_xai=0` at inference produces the "ablated" prediction. The correlation between heatmap mass and output change gives a clean faithfulness signal.

### Hypothesis

| Metric | v1 soft-mask | v2 hard-mask | v3 decoupled (hypothesis) |
|--------|-------------|-------------|--------------------------|
| AP | 0.102 | 0.064 | ↑ (push-pull not fighting decoder) |
| Faithfulness | 0.059 | 0.015 | ↑ (λ_xai creates causal link) |
| Stability | 3.00 | 2.14 | similar or ↓ |
| 3D Dice | 0.843 | 0.834 | ≥ 0.840 (decoder unconstrained) |

---

## Stage 11 — Decoupled Architecture (ProtoSegNet-D) ⬜

### Goal

Implement `ProtoSegNet-D`: a variant of ProtoSegNet where prototype heatmaps are decoupled from the decoder feature path. The decoder operates on raw encoder features; a separate `ProtoHead` converts aggregated heatmaps to logits that additively contribute to the final prediction with weight `λ_xai`.

### Design

#### ProtoHead

```python
class ProtoHead(nn.Module):
    """Aggregates multi-scale prototype heatmaps into per-class logits (B, K, H, W)."""
    def __init__(self, n_levels: int, n_classes: int):
        super().__init__()
        # 1x1 conv: n_levels channels (max-aggregated per level) → K classes
        self.conv = nn.Conv2d(n_levels, n_classes, kernel_size=1)

    def forward(self, A_list: list[torch.Tensor], out_size: tuple) -> torch.Tensor:
        # A_list[l]: (B, K, M, H_l, W_l)
        # max over M then max over K → (B, 1, H_l, W_l) per level
        maps = []
        for A in A_list:
            m = A.max(dim=2).values.max(dim=1, keepdim=True).values  # (B, 1, H_l, W_l)
            maps.append(F.interpolate(m, size=out_size, mode="bilinear", align_corners=False))
        x = torch.cat(maps, dim=1)   # (B, n_levels, H, W)
        return self.conv(x)          # (B, K, H, W)
```

#### ProtoSegNet-D forward

```python
# Forward (decoupled=True):
feat = self.encoder(x)                     # Z_l at 4 levels
A    = [pl(feat[l]) for l, pl in ...]      # prototype heatmaps (unchanged)
logits_dec   = self.decoder(feat)          # full skip connections, no masking
logits_proto = self.proto_head(A, (H, W))
logits = logits_dec + self.lambda_xai * logits_proto
return logits, A                           # same API as v1/v2
```

#### Faithfulness evaluation (decoupled mode)

The standard faithfulness metric perturbs input patches and measures output change. In ProtoSegNet-D, faithfulness can also be measured directly via `λ_xai` ablation: compare `logits_dec + λ_xai * logits_proto` vs `logits_dec` at each spatial location, correlating the per-location difference with prototype heatmap mass. This is a cleaner signal than the patch-perturbation method.

### Implementation Changes

| File | Change |
|------|--------|
| `src/models/prototype_layer.py` | Add `ProtoHead` class |
| `src/models/proto_seg_net.py` | Add `decoupled: bool = False`, `lambda_xai: float = 0.1`; add `self.proto_head`; change forward to skip masking when `decoupled=True` |
| `scripts/train_proto.py` | Add `--decoupled` and `--lambda-xai` CLI flags; save to checkpoint |
| `scripts/evaluate_xai.py` | Auto-read `decoupled`, `lambda_xai` from checkpoint; add `--direct-faithfulness` flag for λ_xai-ablation method |
| `scripts/eval_3d.py` | Auto-read `decoupled`, `lambda_xai` from checkpoint (already reads via `.get()` defaults) |

### Tasks

- [ ] Implement `ProtoHead` in `src/models/prototype_layer.py`
- [ ] Update `ProtoSegNet.__init__` and `forward` for `decoupled=True` path
- [ ] Update `train_proto.py` with `--decoupled` / `--lambda-xai` flags
- [ ] Update `evaluate_xai.py` with λ_xai-ablation faithfulness option
- [ ] Unit tests (`scripts/test_decoupled.py`):
  - Decoupled model output shape (B,8,H,W) unchanged ✓
  - Decoder logits unaffected when `lambda_xai=0` ✓
  - Prototype heatmaps A identical to non-decoupled path ✓
  - Gradient flows to prototype parameters via logits_proto ✓
  - Checkpoint round-trip (`decoupled`, `lambda_xai` keys persist) ✓
  - λ_xai sweep: output changes proportionally ✓
- [ ] Smoke-test: 5-step training, loss decreases, no NaN

### Risk

- **λ_xai too large** → decoder learns to rely on proto_head → dual-role conflict returns. Start at 0.1; sweep [0.05, 0.1, 0.2] in evaluation.
- **ProtoHead collapses** → prototypes learn trivial per-class activations. Monitor proto cosim; div loss remains active.
- **Faithfulness near 0** → if decoder completely dominates (λ_xai=0.1 too small), causal signal is weak. Use λ_xai-ablation faithfulness — it measures the contribution of the decoupled head directly.

---

## Stage 12 — CT Training ProtoSegNet-D + XAI Eval ⬜

### Goal

Train ProtoSegNet-D on CT with the same hyperparameters as v1 `_l2` (best confirmed config), then run full XAI evaluation. Compare all three architectures: v1 soft-mask, v2 hard-mask, v3 decoupled.

### Training Configuration

```python
# Same as v1 _l2, plus:
decoupled  = True
lambda_xai = 0.1
suffix     = "_dec"
```

### Tasks

- [ ] Run training:
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train_proto.py \
    --modality ct \
    --lambda-div 0.001 --lambda-push 0.5 --lambda-pull 0.25 \
    --decoupled --lambda-xai 0.1 \
    --suffix _dec \
    2>&1 | tee results/v3/train_log_proto_ct_dec.txt
  ```
- [ ] Run 3D eval → `results/v3/eval_3d_ct_dec.txt`
- [ ] Run XAI eval → `results/v3/xai_proto_seg_ct_dec.txt`
- [ ] Build 3-architecture comparison table (v1 soft-mask / v2 hard-mask / v3 decoupled)
- [ ] (Optional) λ_xai ablation: rerun XAI eval with λ_xai ∈ {0.05, 0.1, 0.2}

### Expected Outcome

| Metric | v1 _l2 | v2 _hm2 | v3 _dec (target) |
|--------|--------|---------|-----------------|
| 3D Dice | 0.843 | 0.834 | ≥ 0.840 |
| AP | 0.102 | 0.064 | ≥ 0.15 |
| Faithfulness | 0.059 | 0.015 | ≥ 0.10 |
| Stability | 3.00 | 2.14 | ≤ 2.50 |
| IDS | 0.007 | 0.027 | ≤ 0.015 |

**Estimated training time:** ~40s/epoch × 100 epochs ≈ **67 min**

---

## Stage 13 — Report v3 ⬜

### Goal

Write `report/v3/report-v3.md` as a complete scientific report incorporating all three architecture generations:
- v1: soft-mask ProtoSegNet (baseline)
- v2: hard-mask ProtoSegNet (negative result — dual-role conflict diagnosed)
- v3: decoupled ProtoSegNet-D (dual-role fix — results from Stage 12)

The report's central contribution is the **dual-role conflict analysis**: documenting the failure mode, the negative result from hard-masking, and the architectural solution.

### Structure

```
§1 Introduction
§2 Related Work
§3 Methods
  §3.1 ProtoSegNet (v1 soft-mask)
  §3.2 Hard-Mask Extension (v2) — STE, Phase A bug, fix
  §3.3 Decoupled Architecture (v3) — ProtoHead, λ_xai formulation
§4 Experiments
  §4.1 Segmentation Results (3D Dice, ASSD) — all three versions
  §4.2 XAI Evaluation (AP, Faithfulness, Stability, IDS) — all three versions
  §4.3 Architecture Ablation Table
§5 Analysis
  §5.1 Dual-Role Conflict Diagnosis
  §5.2 Dice–Interpretability Trade-off
§6 Discussion & Limitations
§7 Conclusion
```

### Tasks

- [ ] Draft §3.2 (hard-mask) and §3.3 (decoupled) based on execution plan retrospectives
- [ ] Populate §4 tables from result files
- [ ] Write §5.1 dual-role conflict analysis (the key scientific contribution)
- [ ] Update success criteria table

---

## File Structure (v3 additions)

```
src/models/prototype_layer.py   # + ProtoHead class (Stage 11)
src/models/proto_seg_net.py     # + decoupled, lambda_xai params (Stage 11)
scripts/train_proto.py          # + --decoupled, --lambda-xai flags (Stage 11)
scripts/evaluate_xai.py         # + --direct-faithfulness flag (Stage 11)
scripts/test_decoupled.py       # Stage 11 unit tests

checkpoints/
  proto_seg_ct_dec.pth          # Stage 12

results/v3/
  eval_3d_ct_dec.txt            # Stage 12
  xai_proto_seg_ct_dec.txt      # Stage 12
  train_log_proto_ct_dec.txt    # Stage 12

report/v2/
  report-v2.md                  # Stage 13 output (covers v1+v2+v3)

report/v3/
  execution-plan-v3.md          # this file
```

---

## Success Criteria (v3)

**Architecture milestones:**
- [ ] `ProtoHead` implemented, unit tests pass (Stage 11)
- [ ] `ProtoSegNet-D` decoupled forward path implemented (Stage 11)
- [ ] CT decoupled model trained, all metrics computed (Stage 12)

**Segmentation (must not degrade):**
- [ ] CT 3D Dice ≥ 0.840 with decoupled (decoder unconstrained)

**XAI — primary targets (Stage 12):**
- [ ] CT AP ≥ 0.15 (v1: 0.102, v2: 0.064) — confirms push-pull freed from decoder pressure
- [ ] CT Faithfulness ≥ 0.10 (v1: 0.059, v2: 0.015)
- [ ] CT Stability ≤ 2.50 (v1: 3.00, v2: 2.14)
- [ ] IDS ≤ 0.015 (v1: 0.007, v2: 0.027)

**Scientific contribution:**
- [ ] Decoupled architecture confirms or refutes the dual-role fix hypothesis (Stage 12)
- [ ] Report documents the full failure→fix arc across v1/v2/v3 (Stage 13)

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Dual-role conflict persists (λ_xai too large) | Low-Medium | Start λ_xai=0.1; if AP still low, reduce to 0.05 |
| ProtoHead produces trivial class activations | Low | Monitor proto cosim; div loss still active; AP will reflect collapse |
| Faithfulness near 0 (λ_xai too small, decoder dominates) | Low-Medium | Use λ_xai-ablation faithfulness (direct causal signal); also try λ_xai=0.2 |
| Prototype collapse under decoupled training | Low | Same as v1 — div loss remains active |
