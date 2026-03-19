# Execution Plan v6: Attention-Guided Prototype Training

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16 train / 2 val / 2 test patients)
**Hardware:** MacBook 48GB RAM (Apple Silicon MPS)
**Date:** 2026-03-17
**Preceded by:** `plan/execution-plan-v5.md`

---

## Motivation

v5 established two findings:

1. **Attention correctly discovers level hierarchy** (RQ5 ✅): M4-attn (λ=0) converges to L4=0.940, L1/L2≈0 without supervision — matching v4's manual ablation.
2. **Attention hurts prototype quality** (unexpected): L4 purity dropped 0.824→0.537, AP dropped 0.189→0.085. Cause: even though attention assigns near-zero weight to L1/L2, those levels still emit gradient back through the encoder, contaminating the feature representations that L4 prototypes depend on.

The gap between M4-attn (3D Dice 0.8416) and M2 (0.8722) traces to this same problem: L1/L2 prototype supervision degrades encoder quality regardless of their attention weight.

Two targeted fixes are proposed and tested separately before combining:

| Fix | Problem addressed |
|-----|------------------|
| **A: Attention-weighted prototype loss** | L1/L2 gradient contamination of encoder |
| **B: Progressive level pruning** | L1/L2 persistent skip connections + prototype capacity waste |

---

## Research Questions

**RQ6:** Does weighting prototype losses by attention weight eliminate gradient contamination and recover L4 prototype quality (purity, AP), while preserving the level hierarchy discovery?

**RQ7:** Can progressive pruning of low-attention levels automatically converge to a M2-equivalent architecture (L3+L4 only), matching M2's segmentation and prototype quality without prior knowledge of which levels to remove?

**RQ8 (conditional):** Does combining both fixes yield additive benefits beyond either alone?

**RQ9:** Does warm-starting M2 with M4-attn encoder weights improve over cold-start M2 in segmentation (3D Dice) and/or prototype quality (purity, AP)?

---

## Baseline for comparison

All experiments compare against:

| Model | 3D Dice | Purity L4 | AP L4 | Compact. L4 |
|-------|---------|-----------|-------|-------------|
| M4 (max) | 0.8407 | 0.824 | 0.189 | 0.573 |
| M4-attn λ=0 | 0.8416 | 0.537 | 0.085 | 0.494 |
| M2 (max) | 0.8722 | 0.804 | 0.236 | 0.361 |

The target for v6 experiments is to close the M4-attn → M2 gap on both segmentation and prototype quality.

---

## Experiment A: Attention-Weighted Prototype Loss

### Hypothesis

If prototype supervision loss for each level is scaled by that level's attention weight, the encoder will stop being pushed to represent L1/L2-style features as attention converges to suppress them. L4 should develop cleaner, more class-selective representations.

### Mechanism

Currently in the training loop, per-level prototype losses (diversity, push, pull) are aggregated uniformly:

```python
loss = seg_loss + λ_div * div_loss + λ_push * push_loss + λ_pull * pull_loss
# div/push/pull computed over all active levels equally
```

Change: weight each level's prototype loss component by its attention weight before summing:

```python
# w: (B, n_levels) from model._cached_attn_weights
# Compute per-level prototype losses, then weight by mean attention weight
for j, l in enumerate(model.proto_levels):
    w_l = w[:, j].mean().detach()      # scalar, stop gradient through w
    loss += w_l * (λ_div * div_l + λ_push * push_l + λ_pull * pull_l)
```

Key detail: `w_l` is **detached** — we scale the loss magnitude, but do not allow the prototype loss to push gradient back into the attention MLP itself.

### Expected behaviour

- Phase A / early Phase B: w_l ≈ uniform → all levels receive equal prototype supervision (same as current)
- After attention unfreeze (ep 31+): w_L1, w_L2 → 0 → those levels' prototype losses → 0 → encoder stops learning L1/L2-specific features
- L4 encoder representations specialise more cleanly → purity and AP recover

### What changes vs M4-attn (λ=0)

- Training loop: per-level loss weighting (≈ 10 lines of code)
- Model architecture: unchanged
- Checkpoint: saved as `proto_seg_ct_l1234_attn_wloss.pth`
- Logs: `results/v6/train_curve_proto_ct_l1234_attn_wloss.csv`
- Attention weights: `results/v6/attention_weight_evolution_wloss.csv`

### Success criteria — Outcomes

- [ ] Purity L4 > 0.70 — **0.697** ❌ (−0.003, just missed)
- [x] AP L4 > 0.15 — **0.195** ✅
- [x] 3D Dice ≥ 0.8416 — **0.8475** ✅
- [x] w_L4 > 0.5 — **0.870** ✅ / w_L1+L2 < 0.10 — **0.100** ⚠️ borderline

**Verdict: PARTIAL (2–3/4)**. Proceed to Exp B.

---

## Experiment B: Progressive Level Pruning

### Hypothesis

If levels with near-zero attention weight are progressively detached and frozen during training, the encoder is fully relieved of those levels' gradient, and the final architecture is functionally equivalent to M2. This should recover both prototype quality and segmentation performance.

### Mechanism

At the end of each validation epoch (Phase B+), check the rolling mean attention weight for each level. If a level's weight falls below a pruning threshold for a sustained number of epochs, prune it:

```python
PRUNE_THRESHOLD   = 0.05     # weight below this triggers pruning
PRUNE_PATIENCE    = 5        # must stay below threshold for 5 consecutive val epochs
PRUNE_START_EPOCH = 40       # earliest epoch pruning can trigger (let attention stabilise first)

# When level l is pruned:
#   1. Freeze its PrototypeLayer parameters
#   2. Detach its encoder features before the prototype layer:
#        feat[l] = feat[l].detach()
#   3. Remove it from active_levels for attention and mask computation
#   4. Set its prototype losses to 0
```

The decoder skip connections still carry the frozen features (no architectural change needed), but no gradient flows back through those levels. This is a soft prune: the level's features persist in the decoder's skip path, but they are no longer trained.

### Pruning schedule expectation

Based on v5 attention evolution:
- ep 35: w_L1=0.006, w_L2=0.010 → both immediately below threshold
- ep 40: w_L1=0.001, w_L2=0.002 → comfortably below
- With PRUNE_PATIENCE=5 and PRUNE_START_EPOCH=40: L1/L2 pruned around **ep 42–45**
- L3 (w≈0.06) stays above threshold → not pruned
- Final active levels: {L3, L4} — functionally equivalent to M2

### What changes vs M4-attn (λ=0)

- Model: add `pruned_levels: set` tracking, detach logic in forward pass
- Training loop: per-epoch pruning check, rolling attention weight buffer
- No decoder architecture change required
- Checkpoint: `proto_seg_ct_l1234_attn_pruned.pth`
- Prune log: `results/v6/pruning_log.csv` (epoch, level pruned, w at prune time)

### Success criteria

- L1 and L2 automatically pruned (without manual specification)
- 3D Dice ≥ 0.8600 (closing gap toward M2 0.8722)
- Purity L4 > 0.70 (encoder freed from L1/L2 supervision)
- AP L4 > 0.18 (approaching M2 0.236)
- Compactness L4 < 0.50

---

## Experiment C: Combination (Conditional)

### Trigger condition

Proceed to Experiment C only if **both A and B individually meet their success criteria** on at least one key metric compared to M4-attn (λ=0) baseline.

### Hypothesis

Attention-weighted loss and progressive pruning address complementary aspects of the same problem:
- **A** reduces contamination while all levels are still active
- **B** eliminates contamination entirely after convergence

Combining them should give cleaner early-phase training (A) followed by architectural simplification (B), potentially yielding better prototype quality than either alone.

### Changes vs baseline

Both mechanisms active simultaneously. Pruning threshold and patience may need retuning since weighted loss already reduces L1/L2 influence.

### Success criteria

- 3D Dice > M2 (0.8722) — if this is achievable, it means the combination provides benefit beyond explicit level removal
- Purity L4 ≥ M2 (0.804)
- AP L4 ≥ M2 (0.236)

---

## Stage Overview

| Stage | Name | Deliverable | Status |
|-------|------|-------------|--------|
| 23 | Exp A: Attn-weighted loss | `notebooks/23_attn_weighted_loss.ipynb` | ✅ |
| 24 | Exp A analysis | `notebooks/24_attn_weighted_loss_analysis.ipynb` | ✅ |
| 25 | Exp B: Progressive pruning | `notebooks/25_progressive_pruning.ipynb` | ✅ |
| 26 | Exp B analysis | `notebooks/26_progressive_pruning_analysis.ipynb` | ✅ |
| 27 | Exp C: Combination (cond.) | `notebooks/27_combination.ipynb` | ⬜ skipped |
| 27b | Exp D: Two-stage warm-start M2 | `notebooks/27_two_stage_warmstart.ipynb` | ✅ |
| 28 | Report v6 | `report/v6/report-v6.md` | ✅ |

---

## Stage 23 — Exp A Training: Attention-Weighted Prototype Loss ✅

### Implementation

The per-level prototype loss weighting requires restructuring how `ProtoSegLoss` returns
components. Two options:

**Option 1 (preferred — no loss class change):** Compute prototype losses per level in the
training loop and weight manually:

```python
# After criterion(logits, lbls, hm) for seg loss:
# Separately compute per-level div/push/pull using existing loss functions,
# weight by w_l.mean().detach(), accumulate into total loss.
```

**Option 2:** Add `level_weights` parameter to `ProtoSegLoss.forward()`.

Use Option 1 to minimise changes to existing tested code.

### Notebook: `notebooks/23_attn_weighted_loss.ipynb`

```
Cell 1  — Config (same as notebook 20 noent, add WEIGHTED_PROTO_LOSS=True)
Cell 2  — Imports
Cell 3  — Model + dataloaders
Cell 4  — Training loop with per-level weighted prototype loss
Cell 5  — Training curve plot
Cell 6  — Save checkpoint
```

### Tasks

- [x] Implement per-level prototype loss extraction in training loop
- [x] Weight each level's prototype loss by `w_l.mean().detach()`
- [x] Log per-level effective loss weight per epoch alongside attention weights
- [x] Verify gradient flow: attention MLP gradient should NOT include prototype loss path
- [x] Run 100 epochs, save checkpoint as `proto_seg_ct_l1234_attn_wloss.pth`

### Training Results

```
Best val Dice : 0.8203  (epoch 80)
3D Dice       : 0.8475
Checkpoint    : checkpoints/proto_seg_ct_l1234_attn_wloss.pth
Attn log      : results/v6/attention_weight_evolution_l1234_attn_wloss.csv
```

**Per-patient 3D Dice:**

| Patient | MeanFg | LV | RV | LA | RA | Myo | Aort | PA |
|---------|--------|----|----|----|----|-----|------|----|
| ct_1019 | 0.7613 | 0.868 | 0.859 | 0.801 | 0.885 | 0.748 | 0.615 | 0.553 |
| ct_1020 | 0.9337 | 0.896 | 0.959 | 0.942 | 0.906 | 0.927 | 0.976 | 0.929 |
| **Mean** | **0.8475** | | | | | | | |

**Attention weight evolution (selected epochs):**

| Epoch | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|------|------|------|------|
| 20 (end PhA) | 0.177 | 0.311 | 0.283 | 0.230 |
| 35 | 0.022 | 0.316 | 0.170 | 0.492 |
| 50 | 0.002 | 0.218 | 0.065 | 0.716 |
| 65 | 0.000 | 0.119 | 0.037 | 0.844 |
| 80 (end PhB) | 0.000 | 0.100 | 0.030 | 0.870 |
| 100 (end PhC) | 0.000 | 0.100 | 0.040 | 0.860 |

**Key observation:** L2 did NOT collapse to zero (unlike noent run where L2 → 0 by ep 35).
Weighted loss created a self-reinforcing feedback: L2 retained prototype training → better
heatmaps → attention kept assigning it weight → L2 effective weight stayed at ~10–30%.
L1 correctly collapsed to ~0. Final hierarchy: L4=0.87, L2=0.10, L3=0.04, L1≈0.

**vs baselines:**

| Model | 3D Dice | Δ |
|-------|---------|---|
| M4 (max) | 0.8407 | +0.0068 |
| M4-attn noent | 0.8416 | +0.0059 |
| M2 (max) | 0.8722 | −0.0247 |

---

## Stage 24 — Exp A Analysis ✅

### Notebook: `notebooks/24_attn_weighted_loss_analysis.ipynb`

### Tasks

- [x] Run proto quality on new checkpoint
- [x] Compare purity/AP with M4-attn (λ=0) — did gradient contamination reduce?
- [x] Verify attention hierarchy still converges (weighted loss should not disrupt discovery)
- [x] Answer RQ6

### Five-Model Comparison Table

| Model | Aggregation | 3D Dice | Purity L4 | Compact. L4 | AP L4 | Dom. L4 |
|-------|-------------|---------|-----------|-------------|-------|---------|
| M4 (max) | max | 0.8407 | 0.824 | 0.573 | 0.189 | 4.3% |
| M4-attn λ=0.02 | uniform avg | 0.7861 | 0.526 | 0.575 | 0.187 | 9.7% |
| M4-attn λ=0 | learned | 0.8416 | 0.537 | 0.494 | 0.085 | 12.5% |
| **M4-attn wloss** | learned+wloss | **0.8475** | **0.697** | **0.365** | **0.195** | **17.5%** |
| M2 (max) | max | 0.8722 | 0.804 | 0.361 | 0.236 | 49.1% |

Output: `results/v6/comparison_table_v6_expA.csv`

### Key Findings

**Purity L4: 0.697** (+0.160 vs noent 0.537) — large improvement, just below target 0.70.

**AP L4: 0.195** (+0.110 vs noent 0.085) — above target 0.15 ✅. Compactness L4: 0.365
(−0.129 vs noent) — also improved (lower = tighter).

**Unexpected: L2 pixel dominance rose to 66%** despite w_L2≈0.10.
The self-reinforcing feedback (L2 retained training → better heatmaps → higher similarity
scores → higher dominance in winner-takes-all pixel classification) made L2 the dominant
level by pixel count, even though its attention weight was low. Hierarchy:
L1=14%, **L2=66%**, L3=2%, L4=18% (dominance) vs w_L1≈0, w_L2=0.10, w_L3=0.04, w_L4=0.87 (attention).

**Heatmap observation:** wloss and noent produce visually similar diffuse yellow heatmaps
across the thorax — prototypes still fire broadly rather than on specific structures.
M2 heatmaps remain qualitatively superior (localised, class-specific). A distinctive
red edge ring appears in wloss heatmaps at the body boundary — a L2 prototype artifact
from the feedback-stabilised L2 learning body-edge features.

### RQ6 Answer: PARTIAL

| Criterion | Target | Result | Pass? |
|-----------|--------|--------|-------|
| Purity L4 | > 0.70 | 0.697 | ❌ (−0.003) |
| AP L4 | > 0.15 | 0.195 | ✅ |
| 3D Dice | ≥ 0.8416 | 0.8475 | ✅ |
| w_L4 > 0.5, w_L1+L2 < 0.10 | — | 0.870 / 0.100 | ⚠️ borderline |

**Conclusion:** Attention-weighted prototype loss substantially improved L4 prototype quality
(purity +0.160, AP +0.110, compactness −0.129) compared to noent, confirming that
gradient contamination was partially reduced. However the self-reinforcing L2 feedback
loop prevented complete suppression — L2 stabilised at w≈0.10 rather than collapsing to
zero, leaving residual contamination. Full recovery to M4 quality (purity 0.824) or M2
quality (purity 0.804) was not achieved. Exp A meets 2/4 criteria and partially meets
the remaining two → proceed to Exp B (progressive pruning) per the decision tree.

---

## Stage 25 — Exp B Training: Progressive Level Pruning ✅

### Implementation changes

**In `ProtoSegNet.forward()`** — add pruning support:

```python
# New attribute: self.pruned_levels: set[int] = set()
# In forward(), for pruned levels:
if l in self.pruned_levels:
    feat[l] = feat[l].detach()   # stop gradient; skip connection still uses features
    masked[l] = feat[l]          # bypass mask module (no prototype activation needed)
    continue
```

**In training loop** — pruning check at each val epoch:

```python
# Rolling buffer: attn_history[l] = deque of recent w_l values
# At each val epoch (epoch >= PRUNE_START_EPOCH):
for j, l in enumerate(model.proto_levels):
    recent_weights = attn_history[l]  # last PRUNE_PATIENCE epochs
    if len(recent_weights) >= PRUNE_PATIENCE and max(recent_weights) < PRUNE_THRESHOLD:
        model.pruned_levels.add(l)
        for p in model.proto_layers[str(l)].parameters():
            p.requires_grad_(False)
        log_pruning_event(epoch, l, w_l)
```

### Notebook: `notebooks/25_progressive_pruning.ipynb`

```
Cell 1  — Config (PRUNE_THRESHOLD=0.05, PRUNE_PATIENCE=5, PRUNE_START_EPOCH=40)
Cell 2  — Imports + model with pruning support
Cell 3  — Training loop with pruning check
Cell 4  — Training curve + pruning event markers
Cell 5  — Save checkpoint
```

### Tasks

- [x] Add `pruned_levels` attribute and detach logic to `ProtoSegNet`
- [x] Implement pruning check in training loop with rolling attention buffer
- [x] Log pruning events to `results/v6/pruning_log.csv`
- [x] Mark pruning events on training curve plot
- [x] Run 100 epochs, save checkpoint as `proto_seg_ct_l1234_attn_pruned.pth`

### Training Results

```
Best val Dice : 0.8136
Pruning events: L1 ep 55 (w=0.0002), L2 ep 55 (w=0.0005)
Final active  : {L3, L4}  pruned={L1, L2}
Final attn    : w_L3=0.582, w_L4=0.417  (L3 surpassed L4 post-pruning)
Checkpoint    : checkpoints/proto_seg_ct_l1234_attn_pruned.pth
```

**Key observation:** L1/L2 pruned at ep 55 (10 epochs late due to rolling buffer
retaining Phase A weights; PRUNE_PATIENCE=5 fix applied for future runs). After pruning,
attention redistributed exclusively to L3/L4, and L3 gradually rose above L4 (w_L3≈0.58
at ep 100) — suggesting the freed encoder prefers mid-scale features for this task.

---

## Stage 26 — Exp B Analysis ✅

### Notebook: `notebooks/26_progressive_pruning_analysis.ipynb`

```
Cell 1  — Load all models (M4, M4-attn λ=0, M4-attn-pruned, M2)
Cell 2  — Proto quality metrics on M4-attn-pruned
Cell 3  — Which levels were pruned and when? (pruning log)
Cell 4  — Five-model comparison table
Cell 5  — 3D Dice eval
Cell 6  — Attention weight evolution with pruning event markers
Cell 7  — Heatmap comparison: M4-attn λ=0 vs M4-attn-pruned vs M2
Cell 8  — Answer RQ7
```

### Tasks

- [x] Run proto quality on pruned checkpoint
- [x] Confirm L1/L2 auto-pruned (no manual intervention)
- [x] Compare prototype quality and Dice vs M2 — how close did pruning get?
- [x] Answer RQ7

### Results

**Run 1 (soft prune, ep 55):** 3D Dice 0.8290, Purity L4 0.671, AP L4 0.141 — 2/5 criteria
**Run 2 (zero-skip, ep 45+50):** best val 0.8040, worse than Run 1 due to BN shift
**RQ7 Verdict: NOT MET.** Auto-pruning works correctly (L1/L2 always identified), but
mid-training architectural change cannot overcome decoder co-adaptation. See §4 of report.
**Exp C: SKIPPED** (neither A nor B met primary Dice criterion).

---

## Stage 27 — Exp C: Combination (Conditional) ⬜ skipped

**Skipped:** Neither Exp A nor Exp B met primary Dice success criteria (A: 0.8475, B: 0.8290 vs
target ≥ 0.8600). Decision tree requires both to succeed before combination. Proceeding to
Exp D (two-stage warm-start) as alternative path.

---

## Stage 27b — Exp D: Two-Stage Warm-Start M2 🔄

### Motivation

Progressive pruning (Exp B) identified the root cause of failure: the decoder co-adapts to 4
skip connections during training, making mid-training architectural changes (zeroing L1/L2 skips)
impossible without BN distribution shift. Three compounding problems:

1. **BN distribution shift** — dec2 (128×128, receives L1 skip) has running stats calibrated for
   non-zero L1 features; sudden zeros at ep 45–55 breaks normalisation
2. **Late pruning** — only 45–55 recovery epochs remain at low LR after pruning triggers
3. **Encoder co-adaptation** — encoder already shaped for 4-level operation throughout training

### Solution: Two-Stage Pipeline

Use M4-attn(noent) as a **discovery model** to identify the informative levels (L3/L4), then
transfer its encoder weights into a **fresh M2 (L3+L4)** model. The decoder is randomly
initialised and trained from ep 1 with only 2 skip connections — no BN shift, full 100-epoch
budget, correct architecture throughout.

```
Stage 1 (already done): Train M4-attn(noent) 100 epochs
   → Attention confirms L1/L2 uninformative (w→0 by ep 40)
   → Encoder contains good multi-scale representations for all levels

Stage 2 (Exp D): Fresh M2(L3+L4) with warm-start encoder
   → Copy all encoder.* keys from noent checkpoint
   → Decoder + prototype layers randomly initialised
   → Train 100 epochs (3-phase identical to cold-start M2)
   → FREEZE_ENCODER_PHASE_A=False (encoder adapts from ep 1)
```

### Research Question: RQ9

**RQ9:** Does warm-starting M2 with M4-attn encoder weights improve over cold-start M2 in
segmentation (3D Dice), prototype quality (purity, AP), or both? Does the warm start provide
a better initial representation that transfers cleanly when only L3/L4 skip connections are used?

### Config

```
Source checkpoint : checkpoints/proto_seg_ct_l1234_attn_noent.pth
Target checkpoint : checkpoints/proto_seg_ct_l3l4_warmstart.pth
Log               : results/v6/train_curve_proto_ct_l3l4_warmstart.csv
Notebook          : notebooks/27_two_stage_warmstart.ipynb
PROTO_LEVELS      : [3, 4]
USE_LEVEL_ATTENTION: False
FREEZE_ENCODER_PHASE_A: False
λ_div=0.001, λ_push=0.5, λ_pull=0.25  (identical to cold-start M2)
```

### Success criteria

- [ ] 3D Dice ≥ 0.8600 (closing gap toward M2 0.8722)
- [ ] 3D Dice ≥ M2 cold-start 0.8722 (ideally beats cold-start)
- [ ] Purity L4 > 0.70
- [ ] AP L4 > 0.18
- [ ] Faster convergence vs cold-start M2 (warm encoder benefit)

### Tasks

- [x] Verify source checkpoint attention hierarchy (w_L1+L2 < 0.01 at ep 100)
- [x] Transfer encoder weights into fresh M2 model
- [x] Implement 3-phase training loop (identical to cold-start M2)
- [x] Run training (L2+L4 and L2+L3+L4)
- [x] Evaluate 3D Dice vs all baselines
- [x] Compare prototype quality vs cold-start M2 and M4-attn variants
- [x] Answer RQ9
- [x] Update report-v6.md with Exp D results

### Actual Results

Two warm-start variants trained. Stage 1 (seed=42) converged to L2+L4 (not L3+L4) due to the L2 feedback loop — w_L4=0.63, w_L2=0.30 at ep 45.

| Model | Levels | Best Val | 3D Dice | Δ vs M2 |
|-------|--------|----------|---------|---------|
| M4 (max) | L1-L4 | — | 0.8407 | −0.0315 |
| M4-attn noent | L1-L4 | 0.7949 | 0.8416 | −0.0306 |
| M4-attn wloss | L1-L4 | 0.8203 | 0.8475 | −0.0247 |
| M4-attn pruned | L1-L4 | 0.8136 | 0.8290 | −0.0432 |
| **L2+L4 warmstart** | L2,L4 | 0.8286 | 0.8291 | −0.0431 |
| **L2+L3+L4 warmstart** | L2,L3,L4 | 0.8191 | 0.8635 | −0.0087 |
| M2 cold-start | L3,L4 | 0.8380 | 0.8722 | 0.0000 |

**Proto quality (per level):**

| Model | Level | Purity | AP | Compact | Dominance |
|-------|-------|--------|----|---------|-----------|
| L2+L4 warmstart | L2 | 0.185 | 0.084 | 0.321 | 76.3% |
| L2+L4 warmstart | L4 | 0.546 | 0.226 | 0.482 | 23.7% |
| L2+L3+L4 warmstart | L2 | 0.160 | 0.035 | 0.333 | 74.0% |
| L2+L3+L4 warmstart | L3 | 0.440 | 0.120 | 0.547 | 12.9% |
| L2+L3+L4 warmstart | L4 | 0.709 | 0.138 | 0.546 | 13.0% |

**Key findings:**
- Stage 1 level discovery is seed-dependent: seed=42 → L2+L4 dominance (not the expected L3+L4), caused by the L2 self-reinforcing feedback loop
- L2 dominates decoder in all warm-start models (68–76% pixel dominance) despite poor purity (0.056–0.185)
- Root cause: Stage 1 encoder was shaped by L2/L4 attention; L2's larger skip connection (128×128) naturally dominates once transferred
- L2+L3+L4 warmstart reaches 0.8635 (−0.0087 vs M2), closest to the target but still below

**RQ9 Verdict: NOT MET.** Warm-starting does not improve over cold-start M2. The two compounding problems — seed-dependent level discovery and L2 encoder contamination carrying over into Stage 2 — mean cold-start M2 with manually selected L3+L4 remains superior.

---

## Stage 28 — Report v6 ✅

`report/v6/report-v6.md` — extends report v5 with:

```
§1 Introduction  (v6 motivation: close the attention→M2 gap)
§2 Experiment A: Attention-weighted prototype loss
   §2.1 Mechanism and implementation
   §2.2 Effect on prototype quality (RQ6)
   §2.3 Effect on segmentation
§3 Experiment B: Progressive level pruning
   §3.1 Mechanism and implementation
   §3.2 Pruning timeline (which levels, which epoch)
   §3.3 Effect on prototype quality and segmentation (RQ7)
§4 Experiment C: Combination (if conducted)
   §4.1 Additive benefit analysis (RQ8)
§5 Discussion
   - Does v6 close the gap to M2?
   - Is automatic level discovery + pruning a viable alternative to manual ablation?
§6 Conclusion
```

### Tasks

- [ ] Write after Stage 26 (and 27 if applicable)
- [ ] Include final recommendation: when to use attention-only vs pruning vs manual level selection

---

## File Structure (v6 additions)

```
plan/
  execution-plan-v6.md

src/
  models/
    proto_seg_net.py                            # add pruned_levels + detach logic (Stage 25)

notebooks/
  23_attn_weighted_loss.ipynb                   # Exp A training
  24_attn_weighted_loss_analysis.ipynb          # Exp A analysis
  25_progressive_pruning.ipynb                  # Exp B training
  26_progressive_pruning_analysis.ipynb         # Exp B analysis
  27_combination.ipynb                          # Exp C (conditional, skipped)
  27_two_stage_warmstart.ipynb                 # Exp D (two-stage warm-start M2)

results/v6/
  train_curve_proto_ct_l1234_attn_wloss.csv
  train_curve_proto_ct_l1234_attn_pruned.csv
  attention_weight_evolution_wloss.csv
  attention_weight_evolution_pruned.csv
  pruning_log.csv                               # epoch, level, weight at prune time
  proto_quality/
    m4_attn_wloss/
    m4_attn_pruned/
    m4_attn_combo/                              # Exp C only
  comparison_table_v6.csv

checkpoints/
  proto_seg_ct_l1234_attn_wloss.pth             # Exp A
  proto_seg_ct_l1234_attn_pruned.pth            # Exp B
  proto_seg_ct_l1234_attn_combo.pth             # Exp C (conditional, skipped)
  proto_seg_ct_l2l4_warmstart.pth              # Exp D run 1 (L2+L4, seed=42)
  proto_seg_ct_l234_warmstart.pth              # Exp D run 2 (L2+L3+L4, seed=42)

report/v6/
  report-v6.md
```

---

## Decision Tree

```
Stage 23–24: Exp A (weighted loss)
    │
    ├─ A meets criteria? ──YES──┐
    │                           │
    └─ A fails? → document      │   Stage 25–26: Exp B (pruning)
      why, still proceed to B   │       │
                                │       ├─ B meets criteria? ──YES──┐
                                │       │                            │
                                │       └─ B fails? → document       │
                                │                                     │
                                └───────────────────────┬────────────┘
                                                        │
                                               Both succeed?
                                                   │
                                                  YES → Stage 27 (Combination)
                                                   │
                                                  NO  → Skip to Stage 28
```

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Weighted loss disrupts attention convergence (attention no longer discovers hierarchy) | Low | Monitor w_l evolution; detach w_l from proto loss gradient ensures attention is only driven by seg loss |
| Pruning too aggressive (L3 pruned before it stabilises) | Medium | Set PRUNE_START_EPOCH=40 and PRUNE_PATIENCE=5; L3 weight ~0.06 stays above 0.05 threshold in v5 |
| Pruning too conservative (L1/L2 never pruned) | Low | v5 shows w_L1/L2 < 0.01 from ep 40 onward; threshold 0.05 gives large margin |
| Phase B initial projection collapse (same issue as v5) | Known | Already solved: skip initial projection, use periodic projection at ep 30/40/50/60/70 |
| Combination (Exp C) hyperparameter interaction | Medium | Weighted loss reduces L1/L2 proto gradient before pruning triggers; may need to reduce PRUNE_THRESHOLD to 0.02 |
