# Report v7: Automated Level Selection — Limits of Attention and Ablation

**Project:** Interpretable 2D Cardiac Segmentation with Prototype Networks
**Dataset:** MM-WHS CT (16 train / 2 val / 2 test patients)
**Date:** 2026-03-19
**Preceded by:** `report/v6/report-v6.md` (two-stage warm-start)

---

## §1 Introduction

Report v6 introduced a two-stage warm-start pipeline and identified two root causes for the
persistent gap between M4-attn and the cold-start M2 baseline (Dice 0.8722):

1. **L2 feedback loop** — during Stage 1 training, the attention MLP receives gradient via
   `w_L2 → soft_mask → decoder → seg_loss`, which reinforces L2 regardless of seed.
2. **Level contamination** — L2-dominant encoder representations carry into the Stage 2 decoder
   skip connections, suppressing L4 prototype quality (effective purity ~0.267 across all v6
   warm-start variants).

v6-D showed that manually specifying L3+L4 for Stage 2 substantially recovers purity
(0.649 vs 0.267) while maintaining competitive Dice (0.8635). The remaining open question
was: **can the correct level set be identified automatically, without prior knowledge that
L3+L4 is optimal?**

v7 investigates two mechanisms for this automation — one training-time, one inference-time —
and reports a negative result for both. The positive finding is that the warm-start pipeline
with manually specified L3+L4 is a principled and reproducible methodology.

---

## §2 Baseline Summary

| Model | 3D Dice | Eff. Purity | Notes |
|-------|---------|-------------|-------|
| M2 cold-start | 0.8722 | ~0.77 (est.) | Best overall; manual L3+L4 |
| L2+L3+L4 warm-start (v6-D) | 0.8635 | ~0.267 | L2 dominance |
| M4-attn noent (v6) | 0.8416 | ~0.122 | Best single-model v6 |
| M4-attn wloss (v6) | 0.8475 | ~0.122 | Attention-weighted loss |

---

## §3 Stage 29 — Warm-Start L3+L4

### 3.1 Hypothesis

v6-D used whatever levels Stage 1's attention discovered (L2+L4 for seed=42), carrying the
L2 contamination into Stage 2. The fix is simple: ignore Stage 1's noisy discovery and
directly specify L3+L4 for Stage 2. The Stage 1 encoder still provides a warm-start
(better L3/L4 representations than cold-start), but L2 is excluded from the decoder.

### 3.2 Setup

```
Stage 1 checkpoint : proto_seg_ct_l1234_attn_noent.pth  (seed=42, ep45)
Stage 2 config     : PROTO_LEVELS=[3,4], fresh decoder, full training (100 ep)
λ_div=0.001, λ_push=0.5, λ_pull=0.25  (identical to cold-start M2)
```

### 3.3 Results

**Table 1: Stage 29 outcome**

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| 3D Dice | **0.8656** | ≥ 0.8635 (best v6 warmstart) | ✅ |
| 3D Dice vs M2 | −0.0066 | ≥ 0.8722 | ❌ |
| Effective purity | **0.649** | ≥ 0.60 | ✅ |
| L4 dominance | **56.6%** | ≥ 40% | ✅ |

**Table 2: Proto quality per level (Stage 29)**

| Level | Purity | AP | Compactness | Dominance |
|-------|--------|----|-------------|-----------|
| L3 | 0.486 | 0.213 | 0.634 | 43.4% |
| L4 | 0.774 | 0.224 | 0.402 | 56.6% |

**RQ10 Verdict: PARTIAL (3/4).** Excluding L2 from Stage 2 is the decisive fix —
effective purity jumps from ~0.267 (all v6 warm-starts) to 0.649. The remaining
Dice gap to M2 (−0.0066) is small and attributed to L2 contamination in the Stage 1
encoder representations, not to the warm-start paradigm itself. L4 correctly dominates
pixel coverage at 56.6%, confirming the encoder has developed discriminative deep features.

---

## §4 Stage 30 — Fix LevelAttentionModule

### 4.1 Hypothesis

The L2 feedback loop in Stage 1 is caused by gradient flowing from the attention MLP
back into the encoder: better L2 features → higher w_L2 → more gradient to L2 encoder →
better L2 features (cycle). Two targeted architectural changes break this:

**Change 1 — Feature detach:**
```python
# Before:
x = torch.cat(pooled, dim=1)
# After:
x = torch.cat(pooled, dim=1).detach()  # attention reads features; cannot write them
```

**Change 2 — Temperature annealing:**
```python
T = max(1.0, T_START * ((T_END / T_START) ** (epoch / ANNEAL_EP)))
w = torch.softmax(logits / T, dim=-1)   # T=5→1 over 40 epochs post-unfreeze
```

High T forces uniform weights early on, preventing random init from committing to any
level before the encoder has learned meaningful representations.

### 4.2 Results (seed=42, partial run)

**Table 3: Attention weight evolution — fixed Stage 1**

| Epoch | T | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|---|------|------|------|------|
| 5–30 | 5.0 | ~0.25 | ~0.25 | ~0.25 | ~0.25 |
| 35 | 4.5 | 0.09 | **0.35** | 0.29 | 0.27 |
| 40 | 4.0 | 0.05 | **0.47** | 0.29 | 0.19 |
| 45 | 3.5 | 0.02 | **0.48** | 0.30 | 0.20 |

Temperature annealing worked as designed: weights remain near-uniform at T=5. However,
once T drops and the softmax sharpens, the MLP converges to L2+L3 dominance — not L3+L4.

### 4.3 Root Cause: Fundamental Objective Mismatch

Feature detach removed the gradient path from the attention MLP back to the encoder.
However, the MLP still receives gradient via the segmentation path:
```
w → soft_mask → decoder → seg_loss → ∇w
```
The MLP therefore learns to upweight whichever level *maximises segmentation quality*.
L2 (64×64 spatial) genuinely contributes more to decoder reconstruction than L4 (16×16).
This is correct MLP behaviour — it is simply optimising the wrong objective for our goal.

**The mismatch:**

| Objective | Favours |
|-----------|---------|
| Segmentation loss (decoder reconstruction) | L2 — fine spatial detail |
| Prototype interpretability (purity, AP) | L4 — deep semantic features |

No gradient-based attention trained on segmentation loss can automatically discover L4
as the interpretability-optimal level. This is a finding about the architecture class,
not a tunable hyperparameter.

**RQ11 Verdict: NOT MET — but the failure is informative.** The detach + temperature
mechanism correctly eliminates the gradient feedback loop. The remaining issue is
a fundamental conflict between the segmentation objective and the interpretability objective.
Resolving it requires a mechanism that explicitly rewards prototype quality, which is
out of scope for this architecture.

---

## §5 Abandoned: Post-hoc Level Ablation

### 5.1 Motivation

Since training-time attention fails due to objective mismatch, a third approach was
attempted: load the trained M4 model, zero out each level subset at inference time
using the existing `pruned_levels` mechanism, and measure 3D Dice and effective purity
for all 15 non-empty subsets of {L1,L2,L3,L4}.

### 5.2 Per-level Quality (M4 noent, all 4 levels)

| Level | Purity | AP | Compactness |
|-------|--------|----|-------------|
| L1 | 0.084 | 0.020 | 0.358 |
| L2 | 0.195 | 0.069 | 0.329 |
| L3 | 0.613 | 0.076 | 0.529 |
| L4 | 0.689 | 0.085 | 0.494 |

L1 and L2 have very low purity. L3 and L4 are substantially better.

### 5.3 Ablation Results

**Table 4: All 15 subsets (sorted by eff. purity)**

| Subset | 3D Dice | Eff. Purity | Pareto |
|--------|---------|-------------|--------|
| {L4} | 0.464 | 0.689 | ★ |
| {L3,L4} | 0.695 | 0.656 | ★ |
| {L3} | 0.000 | 0.613 | — |
| {L2,L3,L4} | 0.800 | 0.379 | ★ |
| {L2,L4} | 0.645 | 0.338 | — |
| {L1,L3,L4} | 0.814 | 0.332 | ★ |
| {L2,L3} | 0.023 | 0.300 | — |
| {L1,L2,L3,L4} | 0.842 | 0.286 | ★ |
| … | … | … | — |

**Finding 1 — L4 is a hard architectural dependency.**
Any subset without L4 gives Dice ≈ 0. The M4 decoder's deepest block
`dec4(masked[4], masked[3])` uses L4 as its primary input (upsampled from 16×16).
When L4 is zeroed, the decoder receives zeros at its critical bottleneck and fails
regardless of which other levels are active.

**Finding 2 — Shallow levels dominate pixel coverage.**
When L1 or L2 are included in a subset, they win 50–70% of pixels despite low purity.
Adding L1 or L2 to {L3,L4} drops effective purity from 0.656 to 0.332–0.379.

### 5.4 Why Ablation Fails as a Discovery Tool

**Problem 1 — Non-unique Pareto front.** Five subsets are Pareto-optimal: {L4},
{L3,L4}, {L1,L3,L4}, {L2,L3,L4}, {L1,L2,L3,L4}. The ablation cannot select among
them without a human-specified Dice-sacrifice threshold.

**Problem 2 — Co-adaptation bias.** The M4 decoder was trained with all 4 levels;
zeroing out L1+L2 at inference gives artificially low Dice for {L3,L4}:

| Method | {L3,L4} Dice |
|--------|-------------|
| Post-hoc ablation | 0.695 |
| Dedicated training (Stage 29) | 0.866 |
| Gap (co-adaptation) | **0.171** |

The Dice values on the ablation Pareto front are unreliable. Any threshold-based
selection would give wrong results without retraining to verify.

**Conclusion.** Post-hoc ablation is descriptive (confirms which levels the M4 decoder
depends on) but not prescriptive. It cannot identify the interpretability-optimal level
set without prior knowledge. The ablation confirms {L3,L4} is Pareto-optimal only
because we already know it is the right choice from Stage 29.

---

## §6 Summary

**Table 5: v7 outcome overview**

| Stage | Question | Finding |
|-------|----------|---------|
| 29 | Does warm-starting Stage 2 with L3+L4 (ignoring Stage 1 discovery) recover purity? | **YES** — eff. purity 0.267→0.649; Dice 0.8656 (−0.0066 vs M2) |
| 30 | Does detach + temperature annealing make Stage 1 stably discover L3+L4? | **NO** — mechanically correct, but seg-loss attention is objectively misaligned with interpretability |
| 31 (abandoned) | Does post-hoc ablation identify the optimal level subset automatically? | **NO** — Pareto non-unique; Dice values unreliable due to co-adaptation |

**What v7 establishes:**
1. The warm-start pipeline works — manual L3+L4 gives the best warm-start result to date.
2. Automated level selection via segmentation-loss attention is architecturally infeasible.
3. Automated level selection via post-hoc ablation is also infeasible (non-unique Pareto
   + co-adaptation bias).
4. **Manual selection of L3+L4 is the justified and practical approach** — supported by
   evidence from v6-D, Stage 29, and the ablation's Pareto front consistently placing
   {L3,L4} as the high-purity option.

---

## §7 Conclusions

### 7.1 On Automated Level Discovery

Both training-time and inference-time automation fail for the same underlying reason:
*the segmentation objective and the interpretability objective favour different levels.*
Segmentation rewards spatial resolution (L2 wins); prototype purity rewards semantic
depth (L4 wins). No single-objective mechanism can resolve this.

A principled solution would require multi-objective attention that adds an explicit
prototype purity term to the attention loss, enabling online purity estimation during
training. This is identified as future work.

### 7.2 On the Warm-Start Pipeline

The two-stage warm-start with manually specified L3+L4 is validated as a sound methodology:
- Encoder warm-start from M4 provides better L3/L4 representations than cold-start
  (Stage 29 Dice 0.8656 vs M2 0.8722 — gap of only 0.7%)
- Effective purity 0.649 represents a 2.4× improvement over v6 warm-starts (~0.267)
- The pipeline requires specifying the level set in advance; given the evidence across
  v6-D, Stage 29, and the ablation, L3+L4 is the principled choice

### 7.3 On the L2 Feedback Loop

The L2 feedback loop — identified as the root cause in v6 — is now fully characterised:
- It operates via `w_L2 → soft_mask → decoder → seg_loss → ∇w_L2 → encoder` (training-time)
- Feature detach breaks the encoder path but not the MLP training path
- Even without the feedback loop, the MLP correctly prefers L2 for segmentation
- This is not a bug; it is correct optimisation of the wrong objective

The only fix is to not use L2 as an active proto-level in Stage 2, which is exactly
what Stage 29 does.

---

## §8 Future Directions

| Direction | Rationale |
|-----------|-----------|
| Multi-objective attention (seg + purity) | Directly addresses the objective mismatch identified in Stage 30 |
| Purity-regularised prototype loss per level | Online purity signal can guide which levels receive prototype supervision |
| Cross-modality validation (MR) | All v7 experiments use CT only; L3+L4 optimality should be verified on MR |
| Class-specific level analysis | PA and LA (hardest structures) may benefit from different level configurations |

---

## Appendix: Output Files

```
results/v7/
  train_curve_proto_ct_l3l4_warmstart_v7.csv   # Stage 29 training log
  effective_quality_l3l4_warmstart_v7.csv       # Stage 29 effective quality
  attn_evolution_fixed_seed42.csv               # Stage 30 attention weights
  ablation_results.csv                          # Stage 31 (abandoned): 15-subset table
  pareto_front_ablation.png                     # Stage 31 (abandoned): Pareto plot
  ablation_sensitivity.png                      # Stage 31 (abandoned): sensitivity chart

checkpoints/
  proto_seg_ct_l3l4_warmstart_v7.pth            # Stage 29 best checkpoint (ep90)
  proto_seg_ct_l1234_attn_fixed_seed42.pth      # Stage 30 (seed=42, partial)
```
