# Execution Plan v9: ProtoSeg-Style Architecture with Learned Level Attention

**Project:** Interpretable 2D Cardiac Segmentation — Prototype-Only Prediction Path
**Dataset:** MM-WHS CT (16/2/2 patients), MRI (16/2/2 patients)
**Preceded by:** `report/v8/report-v8.md`
**Created:** 2026-03-21

---

## Motivation

All v1–v8 experiments shared a fundamental architectural flaw: the U-Net decoder had access
to encoder features via skip connections, allowing it to bypass the prototype layer entirely.
This made prototypes causally irrelevant to predictions, producing near-zero Faithfulness
(−0.003 → +0.059 at best) and a Stability floor of ~3.0 that no loss configuration could fix.

The root cause is structural, not hyperparameter-dependent. The fix is equally structural:
**remove the decoder and all skip connections, so that prototype similarity heatmaps are the
sole prediction pathway.** This is the design used in the original ProtoSeg (Sacha et al., 2023),
which never had the bypass problem — but which was never evaluated with Faithfulness or Stability.

Additionally, v5–v7 showed that the LevelAttentionModule failed in the old architecture because
it optimised the wrong objective (which level helps the decoder, not which level is most
interpretable). In the new no-skip architecture, the segmentation objective and the
interpretability objective are the same thing: a precise heatmap = a correct prediction.
The attention mechanism is now correctly aligned by construction.

---

## Experimental Design

Three experiments are run in sequence, each adding exactly one component. This isolates
the contribution of each design choice and builds a cumulative narrative.

```
Stage 9a — Single-scale (L4), no skip
    Answers: Does removing the bypass alone fix XAI?
    Direct comparison against v1–v8 which all had bypass.
         ↓
Stage 9b — Multi-scale (L3+L4), no skip, no attention
    Answers: Does multi-scale help in the clean no-bypass setting?
    Uniform weights (equal contribution per level).
         ↓
Stage 9c — Multi-scale (all 4 levels), no skip, learned attention
    Answers: Does learned attention further improve over fixed multi-scale?
    Attention auto-selects level weights; no manual level choice needed.
         ↓
Stage 10 — Final comparison table
    All three new models + v1–v8 baselines.
```

---

## Design Decisions

### Shared across all three stages

- **No skip connections, no decoder** — the structural fix. `logits = f(heatmaps)` only.
  Verified by unit test: zero all prototype parameters → logits must be zero.
- **L2 similarity kernel** — Stage 8 confirmed 2.5× AP gain over log-cosine.
- **Loss:** `0.5*L_dice + 0.5*L_wce + 0.001*L_div + 0.5*L_push + 0.25*L_pull`
  (identical to Stage 8 best; no changes to loss functions needed)
- **Prototype projection** every 10 epochs in Phase B. Initial projection at Phase B
  start is SAFE — no decoder to destabilise.

### Stage-specific choices

| Stage | Levels | Attention | Justification |
|-------|--------|-----------|---------------|
| 9a | L4 only | None | Mirrors original ProtoSeg; highest purity (0.824 from v4) |
| 9b | L3+L4 | None (uniform) | v4 confirmed optimal fixed configuration; equal weights |
| 9c | L1+L2+L3+L4 | Learned (shared) | Attention auto-selects; no manual pre-filtering needed |

**Why L4 for Stage 9a (not L3):**
v4 established L4 purity = 0.824, L3 purity = 0.639. L4 is the cleanest single-level baseline.

**Why L3+L4 for Stage 9b (not all 4):**
v4's M2 result: L3+L4 outperforms M4 (+3.2% Dice) and M1 (+2.0% Dice) with uniform max-aggregation.
Using L3+L4 with uniform weights gives the cleanest test of multi-scale benefit without
confounding from L1/L2 noise.

**Why all 4 levels for Stage 9c:**
In the no-skip architecture, L1/L2 receiving low attention weight is correct negative feedback
(their imprecise heatmaps directly hurt the loss). No feedback loop, no bypass to exploit.
Attention acts as an online, input-conditioned level selector.

**Why shared (not per-class) attention in Stage 9c:**
v5 showed class-invariant convergence (w_L4: 0.937–0.943 across all 8 classes) even with
per-class capacity. Class-specificity is already encoded in the prototype layer.
Per-class attention is recorded as Alternative B below.

---

### Alternative Architectures (Not Selected — Recorded for Reference)

**Alternative A: Per-class level attention**
- `w[k] = softmax(MLP_k(context))` — separate MLP per class
- Theoretically motivated (Myo → L3, Aorta → L4)
- Risk: v5 showed class-invariant convergence; small test set (2 patients) limits reliability
- *Add as variant in Stage 10 ablation if Stage 9c shows meaningful per-class weight variation*

**Alternative B: Cross-attention similarity (Transformer-style)**
- Replace L2 distance with `dot_product(Q=prototype, K=spatial_features) / √D`
- The attention map IS the prototype heatmap — most principled formulation
- Higher implementation complexity; training instability risk on small dataset
- *Deferred to future work; mention in Discussion as natural extension*

---

## Stage Overview

| Stage | Name | Config | Deliverable | Status |
|-------|------|--------|-------------|--------|
| 9a | Single-scale baseline | L4 only, no skip | CT + MRI Dice; XAI metrics | ⬜ |
| 9b | Multi-scale fixed | L3+L4, no skip, uniform | CT + MRI Dice; XAI metrics | ⬜ |
| 9c | Multi-scale + attention | All levels, no skip, learned | CT + MRI Dice; XAI metrics; attention weights | ⬜ |
| 10 | Final comparison | All models + v1–v8 baselines | Comparison table; report v9 | ⬜ |

---

## Architecture: `ProtoSegNetV2`

**File:** `src/models/proto_seg_net_v2.py`

One class handles all three stages via `proto_levels` and `use_attention` arguments.

```python
class ProtoSegNetV2(nn.Module):
    def __init__(self, n_classes=8, proto_levels=(4,), use_attention=False):
        self.encoder    = HierarchicalEncoder2D()
        self.proto_layers = nn.ModuleDict({
            str(l): PrototypeLayer(n_classes, PROTOS_PER_LEVEL[l], CHANNELS[l])
            for l in proto_levels
        })
        self.level_attention = LevelAttentionModule(proto_levels) if use_attention else None

    def forward(self, x, T=1.0):
        features  = self.encoder(x)
        heatmaps  = {}
        upsampled = {}
        for l in self.proto_levels:
            A = self.proto_layers[str(l)](features[l])      # (B, K, M_l, H_l, W_l)
            heatmaps[l]  = A
            A_agg = A.max(dim=2).values                     # (B, K, H_l, W_l)
            upsampled[l] = F.interpolate(A_agg, (256,256), mode='bilinear', align_corners=False)

        if self.level_attention is not None:
            w = self.level_attention(features, T)           # (B, n_levels)
        else:
            n = len(self.proto_levels)
            w = torch.full((x.size(0), n), 1/n, device=x.device)

        logits = sum(w[:, j].view(-1,1,1,1) * upsampled[l]
                     for j, l in enumerate(self.proto_levels))
        return logits, heatmaps, w

# Stage 9a:  ProtoSegNetV2(proto_levels=(4,),       use_attention=False)
# Stage 9b:  ProtoSegNetV2(proto_levels=(3,4),      use_attention=False)
# Stage 9c:  ProtoSegNetV2(proto_levels=(1,2,3,4),  use_attention=True)
```

**Key invariant:** `logits` is computed ONLY from `heatmaps`.
No encoder features reach the output directly — the bypass is structurally impossible.

### `LevelAttentionModule` (Stage 9c only)

Adapted from `notebooks/20_attention_training.ipynb` (v5). Key changes vs v5:
- `T` accepted as runtime argument for temperature annealing
- No `lambda_ent` entropy regularisation (proven harmful in v5)
- No `feature_detach` (not needed — feedback is now correct in no-skip architecture)

```python
class LevelAttentionModule(nn.Module):
    def __init__(self, proto_levels):
        channels    = {1: 32, 2: 64, 3: 128, 4: 256}
        context_dim = sum(channels[l] for l in proto_levels)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 64), nn.ReLU(),
            nn.Linear(64, len(proto_levels))
        )
        self.proto_levels = proto_levels

    def forward(self, features, T=1.0):
        pooled  = [F.adaptive_avg_pool2d(features[l], 1).flatten(1)
                   for l in self.proto_levels]
        context = torch.cat(pooled, dim=1)
        return F.softmax(self.mlp(context) / T, dim=-1)   # (B, n_levels)
```

### Reused from existing codebase (no changes)

- `src/models/encoder.py` — `HierarchicalEncoder2D`
- `src/models/prototype_layer.py` — `PrototypeLayer`, `PrototypeProjection`, `PROTOS_PER_LEVEL`
- `src/losses/diversity_loss.py` — `prototype_diversity_loss`, `prototype_push_pull_loss`, `ProtoSegLoss`
- `src/metrics/` — all four XAI metric modules (AP, IDS, Faithfulness, Stability)

---

## Training Schedule (all three stages)

| Phase | Epochs | Encoder | Prototypes | Attention | Notes |
|-------|--------|---------|-----------|-----------|-------|
| A (Warm-up) | 1–20 | Training | **Frozen** | **Frozen** | Establish encoder before prototype attachment |
| B (Joint) | 21–80 | Training | Training | Frozen ep 21–30, then Training | 10-ep attention warmup delay (Stage 9c only) |
| C (Fine-tune) | 81–100 | Frozen | Frozen | **Training** | Refine attention weights (Stage 9c); or freeze all (9a, 9b) |

**Temperature annealing (Stage 9c, Phase B after attention unfreezes):**
```
T(epoch) = max(1.0, 5.0 × (1.0/5.0)^((epoch − 31) / 40))
→ T: 5.0 at ep 31, decays to 1.0 by ep 71
```

---

## Unit Tests (Stage 9 — before any training)

- [ ] `logits` shape = `(B, 8, 256, 256)` for all three configs
- [ ] `heatmaps[l]` shape = `(B, 8, M_l, H_l, W_l)` for each active level
- [ ] `w` shape = `(B, n_levels)`, sums to 1.0 per sample
- [ ] **Bypass invariant:** zero all `proto_layers` weights → `logits` all-zeros ← critical
- [ ] Gradients flow to `encoder`, `proto_layers`, and `level_attention` (Stage 9c)
- [ ] T=100 → w ≈ uniform; T=0.1 → w ≈ one-hot (Stage 9c)
- [ ] `PrototypeProjection` works on `ProtoSegNetV2` for all three configs

---

## Stage 9a — Single-Scale Baseline (L4 only, no skip)

### Training

```bash
python scripts/train_proto_v2.py --modality ct \
    --proto-levels 4 --no-attention \
    --lambda-div 0.001 --lambda-push 0.5 --lambda-pull 0.25 \
    --suffix _v2_l4 --max-epochs 100

python scripts/train_proto_v2.py --modality mr \
    --proto-levels 4 --no-attention \
    --lambda-div 0.001 --lambda-push 0.5 --lambda-pull 0.25 \
    --suffix _v2_l4 --max-epochs 100
```

**Checkpoints:** `checkpoints/proto_seg_{ct,mr}_v2_l4.pth`

### What to measure

- 3D Dice + ASSD (CT and MRI)
- AP, IDS, Faithfulness, Stability
- L4 prototype purity and compactness (from `scripts/proto_quality.py`)

### Expected outcome

- Dice lower than v1–v8 ProtoSegNet (no skip = less spatial detail from 16×16 → 256×256)
- AP, Faithfulness, Stability substantially better than v1–v8 (bypass removed)
- This is the "does the structural fix work?" validation

### Decision gate

If AP < 0.10 (worse than Stage 8) or Faithfulness < 0: **stop and diagnose before 9b/9c**.
Expected: AP ≥ 0.20, Faithfulness ≥ 0.20.

---

## Stage 9b — Multi-Scale Fixed (L3+L4, no skip, uniform weights)

### Training

```bash
python scripts/train_proto_v2.py --modality ct \
    --proto-levels 3 4 --no-attention \
    --lambda-div 0.001 --lambda-push 0.5 --lambda-pull 0.25 \
    --suffix _v2_l34 --max-epochs 100

python scripts/train_proto_v2.py --modality mr \
    --proto-levels 3 4 --no-attention \
    --lambda-div 0.001 --lambda-push 0.5 --lambda-pull 0.25 \
    --suffix _v2_l34 --max-epochs 100
```

**Checkpoints:** `checkpoints/proto_seg_{ct,mr}_v2_l34.pth`

### What to measure

- Same metrics as Stage 9a
- Per-level AP (L3 vs L4 independently)
- Level dominance (which level wins pixel-wise max; expected ~50/50 from v4)

### Expected outcome vs Stage 9a

- Dice: higher (L3 at 32×32 provides better boundary detail than L4 at 16×16)
- AP: similar or higher (L3 adds complementary spatial coverage)
- Faithfulness: similar (both are bypass-free)
- The gap Stage9b − Stage9a quantifies the **pure multi-scale benefit**

---

## Stage 9c — Multi-Scale with Learned Attention (all levels, no skip)

### Training

```bash
python scripts/train_proto_v2.py --modality ct \
    --proto-levels 1 2 3 4 --use-attention \
    --T-start 5.0 --T-end 1.0 --anneal-epochs 40 \
    --lambda-div 0.001 --lambda-push 0.5 --lambda-pull 0.25 \
    --suffix _v2 --max-epochs 100

python scripts/train_proto_v2.py --modality mr \
    --proto-levels 1 2 3 4 --use-attention \
    --T-start 5.0 --T-end 1.0 --anneal-epochs 40 \
    --lambda-div 0.001 --lambda-push 0.5 --lambda-pull 0.25 \
    --suffix _v2 --max-epochs 100
```

**Checkpoints:** `checkpoints/proto_seg_{ct,mr}_v2.pth`

### Attention weight monitoring

Log mean batch attention weights every epoch during Phase B:

| Expected behaviour | Flag as anomaly if |
|-------------------|-------------------|
| L1/L2 → ~0 within ep 41–50 | L2 weight > 0.10 after ep 60 |
| L4 converges to > 0.50 | L4 weight < 0.30 at ep 80 |
| Smooth monotone decay for L1/L2 | Oscillation > ±0.05 per epoch |

Log to: `results/v9/attention_weight_evolution_ct.csv`

### Expected outcome vs Stage 9b

- Dice: similar or slightly higher (attention optimally weights levels per image)
- AP: higher (attention suppresses L1/L2 noise that would otherwise dilute the heatmap)
- Stability: potentially lower (fewer active levels → simpler heatmap landscape)
- The gap Stage9c − Stage9b quantifies the **attention benefit over fixed uniform weights**

---

## Stage 10 — Final Comparison

### Full comparison table (CT)

| Model | Dice CT | AP | Faithfulness | Stability | Bypass | Levels | Attention |
|-------|---------|-----|-------------|-----------|--------|--------|-----------|
| Baseline U-Net | 0.867 | — | — | — | Yes | — | — |
| ProtoSegNet cosine (Stage 7) | 0.843 | 0.041 | −0.003 | 3.15 | Yes | L1–L4 | No |
| ProtoSegNet L2 (Stage 8 best) | 0.843 | 0.102 | +0.059 | 3.00 | Yes | L1–L4 | No |
| **Stage 9a (L4, no skip)** | TBD | TBD | TBD | TBD | **No** | L4 | No |
| **Stage 9b (L3+L4, no skip)** | TBD | TBD | TBD | TBD | **No** | L3+L4 | No |
| **Stage 9c (all levels, attention)** | TBD | TBD | TBD | TBD | **No** | L1–L4 | Yes |

### Success criteria

| Metric | Minimum (gate) | Target |
|--------|---------------|--------|
| 3D Dice CT (Stage 9c) | ≥ 0.80 | ≥ 0.83 |
| 3D Dice MRI (Stage 9c) | ≥ 0.76 | ≥ 0.79 |
| AP (Stage 9a) | ≥ 0.15 | ≥ 0.25 |
| AP (Stage 9c) | ≥ 0.25 | ≥ 0.40 |
| Faithfulness (Stage 9a) | ≥ 0.15 | ≥ 0.30 |
| Faithfulness (Stage 9c) | ≥ 0.25 | ≥ 0.40 |
| Stability (any stage) | ≤ 2.00 | ≤ 1.00 |
| L1+L2 attention weight (Stage 9c) | ≤ 0.10 | ≤ 0.05 |

### Narrative structure for report v9

The three-step progression directly answers three questions:

1. **Does removing bypass fix XAI?** → 9a vs Stage 8 (same single-scale concept, bypass removed)
2. **Does multi-scale help in the clean setting?** → 9b vs 9a (add L3, uniform weights)
3. **Does learned attention add value over fixed selection?** → 9c vs 9b (add attention)

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Dice drops > 0.07 from v1–v8 (16×16 → 256×256 is aggressive upsampling) | Medium | Stage 9b uses L3 (32×32 → 8× upsample); if still too low, add 2-layer CNN head with no skip |
| AP in Stage 9a still low (unexpected bypass) | Low | Run bypass invariant unit test before training; if logits ≠ 0 when protos zeroed, fix before proceeding |
| Attention in Stage 9c converges to L2 dominant | Low | No bypass path → L2 imprecise heatmap directly hurts loss → correct negative feedback |
| Stage 9b Dice lower than 9a (L3 uniform weight hurts L4) | Low | Uniform 0.5/0.5 split; if L3 hurts, try w_L3=0.3, w_L4=0.7 as fixed alternative |
| MRI substantially worse than CT | Medium | Run CT fully through all three stages before starting MRI |
| Phase B projection collapse (v5 bug) | Low | Collapse mechanism requires decoder; structurally impossible here |

---

## File Structure

```
src/models/
    proto_seg_net_v2.py           # ProtoSegNetV2 + LevelAttentionModule

scripts/
    train_proto_v2.py             # Adapted from train_proto.py
                                  # Args: --proto-levels, --use-attention,
                                  #       --T-start, --T-end, --anneal-epochs
    proto_quality.py              # Purity, compactness, dominance per level

results/v9/
    attention_weight_evolution_ct.csv    # Stage 9c: w per epoch
    attention_weight_evolution_mr.csv
    proto_quality_9a_ct.csv              # Stage 9a: purity, compactness
    proto_quality_9b_ct.csv              # Stage 9b
    proto_quality_9c_ct.csv              # Stage 9c
    comparison_table_v9.csv              # Stage 10: full table
    train_curve_proto_ct_v2_l4.csv       # Stage 9a
    train_curve_proto_ct_v2_l34.csv      # Stage 9b
    train_curve_proto_ct_v2.csv          # Stage 9c

checkpoints/
    proto_seg_ct_v2_l4.pth       # Stage 9a CT
    proto_seg_mr_v2_l4.pth       # Stage 9a MRI
    proto_seg_ct_v2_l34.pth      # Stage 9b CT
    proto_seg_mr_v2_l34.pth      # Stage 9b MRI
    proto_seg_ct_v2.pth          # Stage 9c CT
    proto_seg_mr_v2.pth          # Stage 9c MRI
```

---

## Key Differences from v1–v8 (Summary)

| Aspect | v1–v8 | v9 (this plan) |
|--------|-------|----------------|
| Prediction pathway | Prototype + decoder bypass | Prototype only — structurally enforced |
| Faithfulness guarantee | None | Structural: logits = f(heatmaps) only |
| Experimental structure | Ablate from full model | Build up from single-scale baseline |
| Level selection | Manual (max-gap filter in v8) | Learned (attention, correct objective in 9c) |
| Attention objective | Helps decoder → prefers L2 (wrong) | Helps prediction → prefers precise levels (correct) |
| L2 feedback loop | Present (v6 diagnosis) | Absent — no decoder to exploit |
| Stability root cause | Soft-mask bypass | Eliminated by design |
| Feature detach needed | Yes (v7 workaround) | No — feedback is now correct |
| Entropy regularisation | Counterproductive (v5) | Not used |
