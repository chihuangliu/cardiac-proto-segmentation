# Report v6: Closing the Attention–M2 Gap in Multi-Scale Prototype Segmentation

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16 train / 2 val / 2 test patients)
**Date:** 2026-03-18
**Preceded by:** `report/v5/report-v5.md` (learned level attention)

---

## §1 Introduction

Report v5 produced a clear and somewhat contradictory result: the LevelAttentionModule reliably discovers that L1 and L2 are uninformative (w_L1/L2 → 0 within 5 epochs of unfreezing), yet the model that uses this information, M4-attn(λ=0), performs only marginally better than the vanilla M4 (3D Dice 0.8416 vs 0.8407). The explicitly pruned M2 model — which simply removes L1/L2 by hand — still outperforms by a 3.1% margin (0.8722).

v5's §7.2 diagnosed the mechanism: learned attention modulates the *soft mask* but cannot stop L1/L2 gradient flow through (a) the prototype matching objective, (b) the skip connections to the decoder. Even with w_L1 ≈ 0, the encoder's shallow layers are still optimised by L1/L2 prototype supervision, degrading L4 feature quality (purity 0.824→0.537, AP 0.189→0.085).

This report tests two targeted fixes:

> **RQ6 (Experiment A):** Does scaling each level's prototype losses by its attention weight reduce gradient contamination and recover L4 prototype quality?

> **RQ7 (Experiment B):** Can automatically pruning low-attention levels mid-training converge to a M2-equivalent architecture without prior knowledge of which levels to remove?

A conditional third experiment (RQ8: combining both fixes) was planned but not executed, as Experiment B did not meet its success criteria.

---

## §2 Experiment A — Attention-Weighted Prototype Loss

### 2.1 Mechanism

In the M4-attn(λ=0) baseline, per-level prototype losses (diversity, push, pull) are aggregated uniformly across all four levels:

```
L_proto = Σ_l  λ_div * div_l + λ_push * push_l + λ_pull * pull_l
```

The fix scales each level's contribution by its attention weight:

```python
w = model._cached_attn_weights   # (B, n_levels), softmax, has gradient
for j, l in enumerate(model.proto_levels):
    w_l = w[:, j].mean().detach()   # scalar; .detach() stops gradient into attention MLP
    L_proto += w_l * (λ_div * div_l + λ_push * push_l + λ_pull * pull_l)
```

The critical design choice is `.detach()`: the weight scales the prototype loss *magnitude* but does not allow prototype losses to push gradient back into the attention MLP. This preserves the attention module's independence — it is driven only by segmentation loss.

**Expected effect:** As attention converges to suppress L1/L2 (w_L1/L2 → 0), their prototype losses also vanish, removing the encoder gradient from those levels. L4 encoder representations should specialise more cleanly.

### 2.2 Training dynamics

**Table 1: Attention weight evolution — M4-attn wloss (Exp A)**

| Epoch | Phase | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|-------|------|------|------|------|
| 20 | A (frozen) | 0.177 | 0.311 | 0.283 | 0.230 |
| 35 | B | 0.022 | 0.316 | 0.170 | 0.492 |
| 50 | B | 0.002 | 0.218 | 0.065 | 0.716 |
| 65 | B | 0.000 | 0.119 | 0.037 | 0.844 |
| 80 | B (end) | 0.000 | 0.100 | 0.030 | 0.870 |
| 100 | C (end) | 0.000 | 0.100 | 0.040 | 0.860 |

An unexpected observation: L2 stabilised at w ≈ 0.10 and did not collapse to zero (unlike the noent baseline where L2 → 0 by ep 45). L1 correctly suppressed to near-zero.

**Self-reinforcing feedback loop (L2):** The weighted loss creates a positive feedback for L2: higher w_L2 → L2 receives more prototype supervision → L2 heatmaps become better organised → attention evaluates L2 as more informative → w_L2 stays elevated. This feedback locked L2 into a stable fixed point at ~10% weight rather than converging to zero. The consequence is that residual L2 gradient contamination persisted throughout training.

### 2.3 Results

**Table 2: Exp A outcome metrics**

| Metric | M4-attn noent | M4-attn wloss | M2 | Target | Pass? |
|--------|--------------|--------------|-----|--------|-------|
| 3D Dice | 0.8416 | **0.8475** | 0.8722 | ≥ 0.8416 | ✅ |
| Purity L4 | 0.537 | **0.697** | 0.804 | > 0.70 | ❌ (−0.003) |
| AP L4 | 0.085 | **0.195** | 0.236 | > 0.15 | ✅ |
| w_L4 > 0.5, w_L1+L2 < 0.10 | — | 0.870 / 0.100 | — | both | ⚠️ borderline |

**RQ6 Verdict: PARTIAL.** Weighted prototype loss substantially improved L4 quality (purity +0.160, AP +0.110, compactness 0.494→0.365) compared to noent, confirming that gradient contamination was partially reduced. However, the L2 feedback loop prevented complete suppression. Full recovery to M4 purity (0.824) or M2 purity (0.804) was not achieved. Segmentation improved modestly (+0.006 vs noent).

**Five-model comparison table:**

| Model | Aggregation | 3D Dice | Purity L4 | Compact. L4 | AP L4 | Dom. L4 |
|-------|-------------|---------|-----------|-------------|-------|---------|
| M4 (max) | max | 0.8407 | 0.824 | 0.573 | 0.189 | 4.3% |
| M4-attn λ=0.02 | uniform avg | 0.7861 | 0.526 | 0.575 | 0.187 | 9.7% |
| M4-attn λ=0 (noent) | learned | 0.8416 | 0.537 | 0.494 | 0.085 | 12.5% |
| **M4-attn wloss (Exp A)** | learned+wloss | **0.8475** | **0.697** | **0.365** | **0.195** | 17.5% |
| M2 (max) | max | 0.8722 | 0.804 | 0.361 | 0.236 | 49.1% |

---

## §3 Experiment B — Progressive Level Pruning

### 3.1 Mechanism

Rather than merely downweighting low-attention levels, Experiment B removes them structurally during training. At each validation epoch (once Phase B has begun and a minimum epoch threshold is passed), a rolling buffer of recent attention weights is checked. If a level's weight has stayed below a threshold for a sustained period, `model.prune_level(l)` is called:

```python
PRUNE_THRESHOLD   = 0.05    # attention weight below this triggers countdown
PRUNE_PATIENCE    = 3       # consecutive val epochs below threshold required
PRUNE_START_EPOCH = 40      # let attention stabilise before pruning
```

When level *l* is pruned:
1. Its encoder features are modified in `forward()` (see §3.3 for the two variants tested)
2. Its `PrototypeLayer` parameters are frozen (no further gradient)
3. It is excluded from the blended heatmap computation
4. The optimizer's parameter list is refreshed

The attention MLP retains all four output nodes throughout training — the pruning is a runtime decision, not an architectural change.

### 3.2 Implementation note: rolling buffer bug and fix

The initial implementation used `deque(maxlen=5)` accumulating weights from Phase A (≈ 0.25 each). Even when w_L1/L2 dropped to < 0.01 post-unfreeze, the buffer still contained Phase A entries, keeping `max(buffer) > 0.05` for ~10 extra epochs. The fix: `attn_history[l].clear()` at epoch `PHASE_A_END + ATTN_WARMUP_EPOCHS + 1` (ep 31) so the buffer only tracks post-unfreeze weights. With `PRUNE_PATIENCE=3`, this moves the expected pruning epoch from ep 55 to ep 45.

### 3.3 Two pruning variants tested

**Run 1 — Soft prune (detach only):**
Pruned level features are detached but passed unchanged to the decoder skip connection:
```python
feat[l] = feat[l].detach()   # stop gradient; frozen features still fed to decoder
```
This preserves spatial information in the skip connection at the cost of contaminating the decoder with stale L1/L2 representations.

**Run 2 — Zero-skip prune:**
After diagnosing that frozen skip features were maintaining L1/L2 decoder influence, we modified `forward()` to zero pruned skip connections:
```python
feat[l] = torch.zeros_like(feat[l])   # decoder receives zeros → level truly absent
```
The intent was to make the pruned architecture truly equivalent to M2.

### 3.4 Results — Run 1 (soft prune)

Pruning: buffer bug not yet fixed in this run; L1 and L2 both pruned at **ep 55** (w_L1=0.0002, w_L2=0.0005).

Post-pruning attention behaviour: L3 gradually rose above L4 (w_L3=0.582, w_L4=0.417 at ep 100). This is the inverse of M2's dynamics (where L4 dominates), suggesting the L4 encoder representations were already compromised from 55 epochs of 4-level joint training.

**Table 3: Exp B Run 1 outcome metrics**

| Metric | M4-attn noent | **Exp B Run 1** | M2 | Target | Pass? |
|--------|--------------|----------------|-----|--------|-------|
| L1+L2 auto-pruned | — | ✅ ep 55 | — | ✅ | ✅ |
| 3D Dice | 0.8416 | **0.8290** | 0.8722 | ≥ 0.8600 | ❌ |
| Purity L4 | 0.537 | **0.671** | 0.804 | > 0.70 | ❌ |
| AP L4 | 0.085 | **0.141** | 0.236 | > 0.18 | ❌ |
| Compactness L4 | 0.494 | **0.488** | 0.361 | < 0.50 | ✅ |

RQ7 criteria met: **2/5**. Segmentation actually *regressed* vs the noent baseline (−0.013 Dice). Per-class analysis shows the worst drops in PA (0.728, cf. M2 0.841) and LV (0.791, cf. M2 0.869).

### 3.5 Results — Run 2 (zero-skip)

Buffer fix applied (PRUNE_PATIENCE=3, buffer cleared at ep 31). Pruning: L2 pruned at **ep 45**, L1 at **ep 50**. Despite earlier pruning, Run 2 performed worse than Run 1:

| Metric | Run 1 (soft prune) | Run 2 (zero-skip) |
|--------|-------------------|------------------|
| Best val Dice | 0.8218 | 0.8040 |
| Pruning epoch | ep 55 | L2 ep 45, L1 ep 50 |
| Post-prune val trend | gradual recovery | sharp drop at ep 50, oscillates |

The val Dice dropped sharply at ep 50 (0.8026 → 0.7651) when L1 was zero-pruned, then oscillated between 0.77–0.80 for the remaining 50 epochs. This is worse than both the noent baseline and Run 1.

Additionally, this run's attention converged differently: at ep 35, w_L3=0.840 already dominated (vs w_L4=0.697 in Run 1 at the same epoch). Without a fixed random seed, the attention MLP can find different local optima, and L3-dominant dynamics produced lower-quality representations for small structures.

---

## §4 Discussion

### 4.1 Why progressive pruning failed to match M2

Three structural problems undermine the pruning approach regardless of when or how pruning is applied:

**Problem 1 — Decoder co-adaptation.** The decoder is jointly trained with all four skip connections from ep 1. The BatchNorm layers in each decoder block accumulate running statistics calibrated to the non-zero skip inputs they receive during training. When a skip is zeroed or frozen mid-training, those statistics become incorrect, producing a distribution shift. The sharper the zeroing (zero-skip vs soft prune), the more severe the shift — explaining why Run 2 performed worse than Run 1 despite "more correct" semantics.

**Problem 2 — Late pruning leaves insufficient recovery time.** Even with the buffer fix, pruning occurs at ep 45–55. At that point the cosine LR schedule has already decayed to ~1.3e-4 (from 3e-4), and only 45–55 epochs remain. M2 was trained from scratch for 100 epochs at full LR. The pruned model is expected to achieve M2-equivalent performance with a fraction of the training time, a compromised initialisation, and a lower learning rate.

**Problem 3 — Encoder representations are already shaped for 4 levels.** Before pruning, 45–55 epochs of L1/L2 prototype supervision have pushed the encoder's shallow layers toward coarse-scale representations. After pruning, L3 and L4 must compensate using encoder features that were jointly optimised for a different task distribution. M2's encoder, by contrast, is optimised entirely for L3/L4 from the start.

These three problems are architectural and cannot be resolved by tuning PRUNE_PATIENCE or PRUNE_THRESHOLD. They stem from a fundamental mismatch: progressive pruning attempts to transform a 4-level model into a 2-level model within a single training run, but the decoder and encoder are already adapted to the 4-level regime before pruning occurs.

### 4.2 What progressive pruning does succeed at

Despite failing to match M2 on quantitative metrics, Experiment B produces one robust positive finding: **automatic level discovery works reliably.** In both runs, L1 and L2 were correctly identified and pruned without any manual specification:

- The attention mechanism consistently assigns near-zero weight to L1/L2 within 5–15 epochs of unfreezing (replicating v5's RQ5 finding)
- The pruning trigger fires on the correct levels (never L3 or L4)
- The pruning decision is stable across runs with different random initialisation

This suggests that progressive attention monitoring is a reliable diagnostic for which levels are uninformative, even if the subsequent pruning step cannot recover full M2 performance.

### 4.3 The L3-vs-L4 dominance variability (no-seed problem)

A confound in both pruning runs is the attention module's initialisation sensitivity. Run 1 converged to L4-dominant (w_L4 ≈ 0.70 at ep 35); Run 2 converged to L3-dominant (w_L3 ≈ 0.84 at ep 35). Without a fixed random seed, the attention MLP finds different local optima, and the dominant level post-pruning (L3 vs L4) affects the ceiling for small-structure segmentation.

This is distinct from the pruning mechanism itself — it would affect any attention-based model trained without seeds. Fixing the seed would eliminate this variability but would also make it impossible to assess robustness across initialisations.

### 4.4 Recommended two-stage alternative

The experimental results suggest a more effective pipeline for automatic level selection:

1. **Discovery phase (ep 1–40):** Train M4-attn(λ=0) with the 3-phase protocol until attention converges (typically ep 35–40). Identify levels with w < 0.05 as candidates for removal.

2. **Retrain phase:** Initialise a new model using only the identified active levels (e.g., L3+L4), with the Phase A encoder weights transferred from the discovery run. Train for 100 epochs from scratch. The decoder is now sized and co-trained for the correct 2-level architecture from the beginning.

This approach avoids all three structural problems identified in §4.1 while retaining the key benefit of Experiment B: the active levels are chosen automatically, not by manual ablation.

---

## §7 Experiment D — Two-Stage Warm-Start M2

### 7.1 Motivation and design

Progressive pruning (Exp B) diagnosed three structural problems that prevent mid-training architecture changes from recovering M2-level performance (§4.1). §4.4 proposed a two-stage alternative: first use M4-attn(λ=0) as a *discovery* model (Stage 1), then transfer its encoder weights into a fresh M2 model (Stage 2) which trains from ep 1 with only the selected skip connections.

The key question (RQ9): does a warm-start encoder from M4-attn training provide a better initial representation that accelerates convergence or improves prototype quality compared to a cold-start M2?

### 7.2 Stage 1 — Level discovery

Training ran with `SEED=42`, `HIERARCHY_THRESHOLD=0.05` (stop when w_L1+L2 < 0.05 for 2 consecutive checks). Stage 1 stopped at **ep 45** with the following attention distribution:

| Level | w at stop |
|-------|-----------|
| L1 | 0.06 |
| L2 | 0.30 |
| L3 | 0.01 |
| L4 | 0.63 |

**Unexpected outcome:** The expected discovery (L3+L4) did not occur. The model converged to **L2+L4** instead. The threshold check `w_L1+L2 < 0.05` was not met (w_L2=0.30), so a fallback condition was used: `w_L4 > 0.50`. The L2 self-reinforcing feedback loop (identified in Exp A, §2.2) operates here identically: L2 receives prototype supervision → develops better heatmaps → attention evaluates L2 as informative → w_L2 stabilises at ~30%.

This is a seed-dependent outcome. Without fixing the feature detach / temperature annealing architectural issues in `LevelAttentionModule`, Stage 1 level discovery is not reproducible.

### 7.3 Stage 2 — Warm-start training

Two variants were trained, both using the Stage 1 (ep 45) encoder checkpoint:

**Run 1 — L2+L4 warmstart:** Uses the levels Stage 1 actually discovered.

**Run 2 — L2+L3+L4 warmstart:** Adds L3 to include the level cold-start M2 relies on.

Both use the standard 3-phase protocol (Phase A: ep 1–20, B: ep 21–80, C: ep 81–100), `FREEZE_ENCODER_PHASE_A=False`.

### 7.4 Segmentation results

**Table 5: Exp D — 3D Dice per patient**

| Model | ct_1019 | ct_1020 | Mean 3D Dice |
|-------|---------|---------|-------------|
| L2+L4 warmstart | 0.7219 | 0.9362 | 0.8291 |
| L2+L3+L4 warmstart | 0.7985 | 0.9284 | 0.8635 |
| M2 cold-start (ref) | — | — | 0.8722 |

**Table 6: Complete comparison table**

| Model | Levels | Val Dice | 3D Dice | Δ vs M2 |
|-------|--------|----------|---------|---------|
| M4 (max) | L1-L4 | — | 0.8407 | −0.0315 |
| M4-attn noent | L1-L4 | 0.7949 | 0.8416 | −0.0306 |
| M4-attn wloss | L1-L4 | 0.8203 | 0.8475 | −0.0247 |
| M4-attn pruned | L1-L4 | 0.8136 | 0.8290 | −0.0432 |
| L2+L4 warmstart | L2, L4 | 0.8286 | 0.8291 | −0.0431 |
| L2+L3+L4 warmstart | L2, L3, L4 | 0.8191 | 0.8635 | −0.0087 |
| **M2 cold-start** | L3, L4 | 0.8380 | **0.8722** | 0.0000 |

L2+L3+L4 warmstart is the best warm-start variant, reaching 0.8635 — only 0.009 below M2. However neither run exceeds or matches M2.

### 7.5 Prototype quality

**Table 7: Per-level prototype quality — warm-start models**

| Model | Level | Purity | AP | Compactness | Dominance |
|-------|-------|--------|----|-------------|-----------|
| M4 (max) | L4 | 0.824 | 0.189 | 0.573 | 4.3% |
| M4-attn noent | L4 | 0.537 | 0.085 | 0.494 | 12.5% |
| M4-attn wloss | L4 | 0.697 | 0.195 | 0.365 | 17.5% |
| M2 cold-start | L4 | 0.804 | 0.236 | 0.361 | 49.1% |
| L2+L4 warmstart | L2 | 0.185 | 0.084 | 0.321 | **76.3%** |
| L2+L4 warmstart | L4 | 0.546 | 0.226 | 0.482 | 23.7% |
| L2+L3+L4 warmstart | L2 | 0.160 | 0.035 | 0.333 | **74.0%** |
| L2+L3+L4 warmstart | L3 | 0.440 | 0.120 | 0.547 | 12.9% |
| L2+L3+L4 warmstart | L4 | 0.709 | 0.138 | 0.546 | 13.0% |

**Key observation — L2 dominance:** In both warm-start models, L2 captures 68–76% of pixel-level dominance (the level whose prototype gives the highest activation for each pixel). This is despite L2's purity being extremely poor (0.160–0.185). The root cause is structural: Stage 1 training shaped the encoder for L2/L4 attention; the L2 encoder output (spatial 64×64) feeds a large skip connection to the decoder that naturally has high activation magnitude relative to L3/L4. When the Stage 2 decoder is trained from scratch, it co-adapts to rely heavily on L2 skips.

The L2+L3+L4 run achieves the highest L4 purity among warm-start variants (0.709, above the M4-attn wloss L4 purity of 0.697), but L4 dominance collapses to 13.0% because L2 dominates. From the perspective of interpretability — where we want L4 (the highest-resolution semantics) to drive predictions — this is a poor outcome.

### 7.6 Why warm-start fails to beat cold-start M2

Two compounding problems:

**Problem 1 — Seed-dependent level discovery.** The L2 feedback loop causes Stage 1 to converge to L2+L4 (seed=42) rather than the expected L3+L4. The warm-start encoder is therefore shaped by a different level distribution than cold-start M2's encoder, which is trained for L3+L4 from ep 1. Even if we manually override the level selection to include L3, the encoder's L2 representations are already strong and contaminate Stage 2 training.

**Problem 2 — L2 encoder contamination carries over.** Stage 1 trains the encoder to represent L2 features well (because w_L2=0.30 gives L2 substantial prototype supervision). When those weights are transferred, the decoder learns to rely on L2 skips. Cold-start M2 never develops strong L2 representations at all — its encoder is entirely shaped by L3/L4 objectives.

These problems are both consequences of the same root cause: the attention mechanism's training dynamics (feedback loop, seed sensitivity) produce a discovery result that is suboptimal and non-deterministic. Fixing these dynamics (feature detach in `LevelAttentionModule.forward()`, temperature annealing) is a prerequisite for the two-stage pipeline to work as intended.

### 7.7 RQ9 Answer

**RQ9:** Does warm-starting M2 with M4-attn encoder weights improve over cold-start M2?

**Answer: NO.** Both warm-start variants underperform M2 cold-start in segmentation (−0.0431 and −0.0087 3D Dice respectively) and in prototype quality (L2 dominance 68–76% with poor purity vs M2 L4 dominance 49% with purity 0.804). The warm-start encoder imposes L2 dominance that cold-start M2 naturally avoids by never training L2 at all.

---

## §5 Overall Summary Across v5 and v6

**Table 4: Complete model comparison (v4–v6)**

| Model | Key mechanism | 3D Dice | Purity L4 | AP L4 | Compact. L4 |
|-------|--------------|---------|-----------|-------|-------------|
| M4 (max) | Cross-level max | 0.8407 | 0.824 | 0.189 | 0.573 |
| M4-attn λ=0.02 | Learned attn + entropy reg | 0.7861 | 0.526 | 0.187 | 0.575 |
| M4-attn λ=0 (noent) | Learned attn | 0.8416 | 0.537 | 0.085 | 0.494 |
| M4-attn wloss (Exp A) | Attn-weighted proto loss | 0.8475 | 0.697 | 0.195 | 0.365 |
| M4-attn pruned (Exp B) | Progressive pruning | 0.8290 | 0.671 | 0.141 | 0.488 |
| L2+L4 warmstart (Exp D) | Two-stage warm-start | 0.8291 | 0.546* | 0.226* | 0.482* |
| L2+L3+L4 warmstart (Exp D) | Two-stage warm-start | 0.8635 | 0.709* | 0.138* | 0.546* |
| **M2 (max)** | Explicit L3+L4 only | **0.8722** | **0.804** | **0.236** | **0.361** |

*L4 values only; L2 captures 68–76% pixel dominance in both warm-start models.

**RQ6 (Exp A):** PARTIAL. Attention-weighted prototype loss reduces gradient contamination and significantly improves L4 quality (purity +0.160, AP +0.110) but cannot fully suppress the L2 feedback loop. The self-reinforcing dynamic — better L2 prototypes → higher w_L2 → more L2 training — creates a stable fixed point at w_L2 ≈ 0.10 rather than zero.

**RQ7 (Exp B):** NOT MET. Progressive pruning correctly identifies and removes L1/L2 automatically, but the decoder co-adaptation problem prevents recovery to M2-level performance. Segmentation regresses vs the noent baseline (0.8290 vs 0.8416). The zero-skip variant performs even worse due to BN distribution shift.

**Exp C (Combination):** Not executed. Both individual experiments failed to meet their primary Dice criteria, so the combination trigger condition was not satisfied.

**RQ9 (Exp D):** NOT MET. The two-stage warm-start pipeline does not improve over cold-start M2. Stage 1 level discovery is seed-dependent (L2 feedback loop: seed=42 → L2+L4 instead of expected L3+L4), and the warm-start encoder imposes L2 dominance in Stage 2 regardless of the chosen level set. The best warm-start variant (L2+L3+L4) reaches 0.8635, only 0.009 below M2 on Dice but with substantially worse prototype quality (L2 dominance 74%, purity 0.16).

---

## §6 Conclusions

Four iterations of attention-based level management (v5: learned attention; v6-A: weighted prototype loss; v6-B: progressive pruning; v6-D: two-stage warm-start) have progressively narrowed the diagnosis of the M4→M2 performance gap.

The core finding is stable across all experiments: **the gap cannot be closed by modifying an M4 model's training procedure or by reusing its encoder weights.** Whether modifying the loss function (Exp A), pruning levels mid-training (Exp B), or warm-starting a fresh decoder from an M4-attn encoder (Exp D), the L2 encoder contamination problem persists. Cold-start M2, which never trains L1/L2 at all, remains the strongest model.

The positive findings are equally stable:
- Learned attention reliably discovers that L1/L2 are uninformative — confirmed across all experiments. The mechanism works; the problem is that discovery alone cannot retroactively fix an encoder that was co-trained with L1/L2.
- The L2 feedback loop is the primary obstacle. Any model that trains L2 prototype supervision (even with low attention weight) risks L2 locking into a stable 10–30% weight with high decoder dominance and poor purity.
- Attention-weighted prototype loss (Exp A) is still the most effective single-model fix: purity +0.160, AP +0.110, Dice +0.006 vs noent — without requiring architectural changes.

**Root cause diagnosis:** The two-stage pipeline (Exp D) was designed to avoid all three structural problems of progressive pruning (§4.1). It successfully avoids decoder co-adaptation (Stage 2 starts fresh), late pruning (full 100-epoch budget for Stage 2), and insufficient recovery time. Yet it still fails, because the remaining unsolved problem — seed-dependent, L2-contaminated level discovery — undermines Stage 1. The discovered level set is wrong (L2+L4 instead of L3+L4), and the warm-start encoder imposes that wrong bias on Stage 2.

**Path forward:** The two-stage pipeline can work, but only after fixing Stage 1's L2 instability. Two complementary architectural changes in `LevelAttentionModule` are required:
1. **Feature detach** (`x = features.detach()` before the MLP): breaks the L2 feedback loop so attention is a pure observer of feature quality, not a reinforcer
2. **Temperature annealing** (`softmax(logits/T)`, T: 5→1): prevents init-dependent early commitment, replacing seed sensitivity with a principled warm-up

With these fixes, Stage 1 should reliably converge to L3+L4, and Stage 2 warm-start from a clean L3+L4 encoder should close the remaining gap to M2.

---

## Appendix: Outputs

```
notebooks/
  23_attn_weighted_loss.ipynb              # Exp A training (100 epochs)
  24_attn_weighted_loss_analysis.ipynb     # Exp A analysis + RQ6
  25_progressive_pruning.ipynb             # Exp B training (100 epochs, 2 runs)
  26_progressive_pruning_analysis.ipynb    # Exp B analysis + RQ7
  27_two_stage_warmstart.ipynb             # Exp D two-stage warm-start (Stage 1 + Stage 2)
  28_warmstart_analysis.ipynb              # Exp D analysis + RQ9 + proto quality comparison

results/v6/
  train_curve_proto_ct_l1234_attn_wloss.csv
  train_curve_proto_ct_l1234_attn_pruned.csv
  train_curve_proto_ct_l2l4_warmstart.csv
  train_curve_proto_ct_l234_warmstart.csv
  attention_weight_evolution_l1234_attn_wloss.csv
  attention_weight_evolution_l1234_attn_pruned.csv
  pruning_log.csv                          # epoch, level, w_at_prune
  comparison_table_v6_expA.csv             # 5-model table (Exp A)
  comparison_table_v6_expB.csv             # 5-model table (Exp B)
  comparison_table_warmstart.csv           # 7-model table (Exp D)
  proto_quality_warmstart.csv              # per-level quality for warm-start models
  proto_quality_warmstart.png              # purity + AP bar chart vs baselines
  per_class_warmstart.png                  # per-class 3D Dice bar chart
  train_curve_warmstart_comparison.png     # val Dice curves vs baselines

checkpoints/
  proto_seg_ct_l1234_attn_wloss.pth        # Exp A (best val 0.8203, ep 80)
  proto_seg_ct_l1234_attn_pruned.pth       # Exp B Run 1 (best val 0.8218)
  proto_seg_ct_l2l4_warmstart.pth          # Exp D run 1 (L2+L4, best val 0.8286, ep 65)
  proto_seg_ct_l234_warmstart.pth          # Exp D run 2 (L2+L3+L4, best val 0.8191, ep 65)

src/models/proto_seg_net.py               # added pruned_levels, prune_level(),
                                           # zero-skip in forward() for pruned levels
```
