"""
src/metrics/proto_quality.py
Stage 14 — Prototype Quality Metrics

Six metrics for evaluating what prototype networks learn:

1. compute_purity        — fraction of top-N activations on correct class
2. compute_utilization   — fraction of prototypes that are "alive"
3. compute_compactness   — fraction of image area significantly activated
4. compute_dice_sensitivity — Dice drop when one prototype is ablated
5. compute_level_dominance — which level wins the max-aggregation per pixel
6. compute_per_level_ap  — AP computed with each level's heatmap in isolation

Plus:
   build_prototype_atlas — visualise nearest training patch per prototype

All functions accept a ProtoSegNet model and a DataLoader.
Return pandas DataFrames (compatible with .to_csv()).

Note: model.forward() already returns (logits, heatmaps_dict) — no API change needed.
heatmaps_dict : {level_int: (B, K, M, H_l, W_l)}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from src.metrics.dice import dice_per_class, mean_foreground_dice

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ["BG", "LV", "RV", "LA", "RA", "Myo", "Aorta", "PA"]
FOREGROUND = list(range(1, 8))           # skip background k=0
LEVEL_STRIDES = {1: 2, 2: 4, 3: 8, 4: 16}


# ── Private helpers ───────────────────────────────────────────────────────────

def _device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _downsample_labels(labels: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Downsample integer label map (B, H, W) to (B, h, w) using nearest."""
    return (
        F.interpolate(
            labels.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        .squeeze(1)
        .long()
    )


def _ablated_forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    ablate_level: int,
    ablate_k: int,
    ablate_m: int,
) -> torch.Tensor:
    """
    Forward pass identical to ProtoSegNet.forward() but with prototype
    (ablate_level, ablate_k, ablate_m) zeroed out before masking.

    Tightly coupled to ProtoSegNet internals — update if decoder changes.
    Returns logits (B, K, H, W).
    """
    feat = model.encoder(images)

    masked = {}
    for l in [1, 2, 3, 4]:
        if str(l) in model.proto_layers:
            A = model._proto_layer(l)(feat[l])          # (B, K, M, H_l, W_l)
            if l == ablate_level:
                A = A.clone()
                A[:, ablate_k, ablate_m] = 0.0
            if model.no_soft_mask:
                masked[l] = feat[l]
            elif model.hard_mask and not model.hard_mask_active:
                masked[l] = model._soft_mask_fallback(A, feat[l])
            else:
                masked[l] = model.mask_module(A, feat[l])
        else:
            masked[l] = feat[l]

    d = model.dec4(masked[4], masked[3])
    d = model.dec3(d, masked[2])
    d = model.dec2(d, masked[1])
    d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
    d = model.dec1(d)
    return model.final_conv(d)


# ── 1. Purity ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_purity(
    model: torch.nn.Module,
    train_loader,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    For each prototype (level, class k, proto_idx m):
      Collect the peak activation position per training slice.
      Purity = fraction of top-N peaks where GT label at that position == k.

    Uses train_loader (not test) to measure what the prototype has learned
    to respond to during training.

    Returns DataFrame: level, class_idx, class_name, proto_idx, purity, n_samples
    """
    device = _device(model)
    model.eval()

    # records[(l, k, m)] = list of (activation_value, gt_label_at_peak)
    records: dict[tuple, list] = defaultdict(list)

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        feat = model.encoder(images)

        for l, pl in model.proto_layers_dict().items():
            A = pl(feat[l])                               # (B, K, M, H_l, W_l)
            B, K, M, H_l, W_l = A.shape
            lbl_l = _downsample_labels(labels, H_l, W_l)  # (B, H_l, W_l)

            for k in FOREGROUND:
                for m in range(M):
                    act = A[:, k, m, :, :]                # (B, H_l, W_l)
                    for b in range(B):
                        flat_idx = act[b].argmax().item()
                        i, j = flat_idx // W_l, flat_idx % W_l
                        val = act[b, i, j].item()
                        gt = lbl_l[b, i, j].item()
                        records[(l, k, m)].append((val, gt))

    rows = []
    for (l, k, m), rec in records.items():
        rec.sort(key=lambda x: x[0], reverse=True)
        top = rec[:top_n]
        correct = sum(1 for _, lbl in top if lbl == k)
        rows.append({
            "level": l,
            "class_idx": k,
            "class_name": CLASS_NAMES[k],
            "proto_idx": m,
            "purity": correct / len(top) if top else 0.0,
            "n_samples": len(top),
        })

    return pd.DataFrame(rows).sort_values(["level", "class_idx", "proto_idx"]).reset_index(drop=True)


# ── 2. Utilization ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_utilization(
    model: torch.nn.Module,
    test_loader,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """
    For each prototype, compute the maximum activation seen across the test set.
    A prototype is "dead" if its max activation < threshold.

    Returns DataFrame: level, class_idx, class_name, proto_idx,
                       max_activation, is_dead
    """
    device = _device(model)
    model.eval()

    max_acts: dict[tuple, float] = defaultdict(float)

    for batch in test_loader:
        images = batch["image"].to(device)
        feat = model.encoder(images)

        for l, pl in model.proto_layers_dict().items():
            A = pl(feat[l])                              # (B, K, M, H_l, W_l)
            B, K, M, H_l, W_l = A.shape
            for k in FOREGROUND:
                for m in range(M):
                    val = A[:, k, m].max().item()
                    key = (l, k, m)
                    if val > max_acts[key]:
                        max_acts[key] = val

    rows = []
    for (l, k, m), max_val in max_acts.items():
        rows.append({
            "level": l,
            "class_idx": k,
            "class_name": CLASS_NAMES[k],
            "proto_idx": m,
            "max_activation": max_val,
            "is_dead": max_val < threshold,
        })

    df = pd.DataFrame(rows).sort_values(["level", "class_idx", "proto_idx"]).reset_index(drop=True)
    return df


# ── 3. Compactness ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_compactness(
    model: torch.nn.Module,
    test_loader,
    act_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Spatial compactness: fraction of full-resolution (256×256) pixels with
    activation > act_threshold, averaged over test slices.

    Lower compactness = more focused prototype activation.

    Returns DataFrame: level, class_idx, class_name, proto_idx,
                       compactness, level_threshold
    """
    device = _device(model)
    model.eval()

    # level-relative expected max compactness (for reference only)
    level_thresholds = {1: 0.05, 2: 0.08, 3: 0.15, 4: 0.25}

    sums: dict[tuple, list] = defaultdict(list)

    for batch in test_loader:
        images = batch["image"].to(device)
        feat = model.encoder(images)

        for l, pl in model.proto_layers_dict().items():
            A = pl(feat[l])                              # (B, K, M, H_l, W_l)
            B, K, M, H_l, W_l = A.shape

            for k in FOREGROUND:
                for m in range(M):
                    act = A[:, k, m, :, :]               # (B, H_l, W_l)
                    # upsample to 256×256
                    act_up = F.interpolate(
                        act.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False
                    ).squeeze(1)                         # (B, 256, 256)
                    frac = (act_up > act_threshold).float().mean(dim=(1, 2))  # (B,)
                    sums[(l, k, m)].extend(frac.tolist())

    rows = []
    for (l, k, m), vals in sums.items():
        rows.append({
            "level": l,
            "class_idx": k,
            "class_name": CLASS_NAMES[k],
            "proto_idx": m,
            "compactness": float(np.mean(vals)),
            "level_threshold": level_thresholds[l],
        })

    return pd.DataFrame(rows).sort_values(["level", "class_idx", "proto_idx"]).reset_index(drop=True)


# ── 4. Dice Sensitivity ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_dice_sensitivity(
    model: torch.nn.Module,
    test_loader,
) -> pd.DataFrame:
    """
    For each prototype (l, k, m): zero out its activation during inference
    and measure the mean foreground Dice drop vs. baseline.

    dice_drop > 0.005 → prototype is causally important to segmentation.

    Valid only for coupled architectures (v1 soft-mask, v2 hard-mask).
    In v3 decoupled, prototype removal has minimal Dice impact by design.

    Returns DataFrame: level, class_idx, class_name, proto_idx,
                       baseline_dice, ablated_dice, dice_drop
    """
    device = _device(model)
    model.eval()

    # ── Baseline Dice ─────────────────────────────────────────────────────
    baseline_scores: list[float] = []
    all_images: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(images)
        baseline_scores.append(
            mean_foreground_dice(dice_per_class(logits, labels))
        )
        all_images.append(images.cpu())
        all_labels.append(labels.cpu())

    baseline_mean = float(np.mean(baseline_scores))

    # ── Per-prototype ablation ─────────────────────────────────────────────
    rows = []
    for l, pl in model.proto_layers_dict().items():
        K, M = pl.n_classes, pl.n_protos
        for k in FOREGROUND:
            for m in range(M):
                ablated_scores: list[float] = []
                for imgs, lbls in zip(all_images, all_labels):
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    logits_abl = _ablated_forward(model, imgs, l, k, m)
                    ablated_scores.append(
                        mean_foreground_dice(dice_per_class(logits_abl, lbls))
                    )
                ablated_mean = float(np.mean(ablated_scores))
                rows.append({
                    "level": l,
                    "class_idx": k,
                    "class_name": CLASS_NAMES[k],
                    "proto_idx": m,
                    "baseline_dice": baseline_mean,
                    "ablated_dice": ablated_mean,
                    "dice_drop": baseline_mean - ablated_mean,
                })

    return pd.DataFrame(rows).sort_values("dice_drop", ascending=False).reset_index(drop=True)


# ── 5. Level Dominance ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_level_dominance(
    model: torch.nn.Module,
    test_loader,
) -> pd.DataFrame:
    """
    For each pixel in the final aggregated heatmap, which level's contribution
    wins the cross-level max?

    Uses the max-over-K, max-over-M reduction per level, then upsamples all
    levels to 256×256 before taking argmax across levels.

    Returns DataFrame with one row: frac_l1, frac_l2, frac_l3, frac_l4
    (fraction of pixels dominated by each level, averaged over test slices)
    """
    device = _device(model)
    model.eval()

    active_levels = sorted(model.proto_layers_dict().keys())
    level_wins = {l: 0.0 for l in active_levels}
    total_pixels = 0

    for batch in test_loader:
        images = batch["image"].to(device)
        feat = model.encoder(images)
        B = images.shape[0]

        per_level = {}
        for l, pl in model.proto_layers_dict().items():
            A = pl(feat[l])                              # (B, K, M, H_l, W_l)
            A_max_k = A.max(dim=2).values.max(dim=1).values  # (B, H_l, W_l)
            per_level[l] = F.interpolate(
                A_max_k.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False
            ).squeeze(1)                                 # (B, 256, 256)

        stacked = torch.stack([per_level[l] for l in active_levels], dim=0)  # (L, B, 256, 256)
        winner_idx = stacked.argmax(dim=0)               # (B, 256, 256) — index into active_levels

        for idx, l in enumerate(active_levels):
            level_wins[l] += (winner_idx == idx).sum().item()
        total_pixels += B * 256 * 256

    row = {f"frac_l{l}": level_wins[l] / total_pixels for l in active_levels}
    # pad missing levels with 0
    for l in [1, 2, 3, 4]:
        row.setdefault(f"frac_l{l}", 0.0)

    return pd.DataFrame([row])


# ── 6. Per-level AP ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_per_level_ap(
    model: torch.nn.Module,
    test_loader,
    percentile: float = 95.0,
) -> pd.DataFrame:
    """
    Compute Activation Precision using only the heatmap from each level in
    isolation (max over M prototypes at that level only, before cross-level max).

    Reveals which scale produces the most anatomically precise activations.

    Returns DataFrame: level, class_idx, class_name, ap
    """
    device = _device(model)
    model.eval()

    active_levels = sorted(model.proto_layers_dict().keys())
    # acc[(l, k)] = list of per-slice AP values
    acc: dict[tuple, list] = defaultdict(list)

    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        feat = model.encoder(images)
        B, H, W = labels.shape

        for l, pl in model.proto_layers_dict().items():
            A = pl(feat[l])                              # (B, K, M, H_l, W_l)
            A_max_m = A.max(dim=2).values                # (B, K, H_l, W_l)
            A_up = F.interpolate(
                A_max_m, size=(H, W), mode="bilinear", align_corners=False
            )                                            # (B, K, H, W)

            for k in FOREGROUND:
                A_k = A_up[:, k, :, :]                   # (B, H, W)
                G_k = (labels == k).float()              # (B, H, W)
                for b in range(B):
                    a_flat = A_k[b].flatten()
                    thresh = torch.quantile(a_flat, percentile / 100.0)
                    M_k = (A_k[b] >= thresh).float().flatten()
                    if M_k.sum() < 1:
                        continue
                    ap = ((M_k * G_k[b].flatten()).sum() / M_k.sum()).item()
                    acc[(l, k)].append(ap)

    rows = []
    for (l, k), vals in acc.items():
        rows.append({
            "level": l,
            "class_idx": k,
            "class_name": CLASS_NAMES[k],
            "ap": float(np.mean(vals)) if vals else float("nan"),
        })

    return pd.DataFrame(rows).sort_values(["level", "class_idx"]).reset_index(drop=True)


# ── 7. Effective Quality (dominance-weighted aggregate) ───────────────────────

def compute_effective_quality(
    purity_df: pd.DataFrame,
    ap_df: pd.DataFrame,
    compactness_df: pd.DataFrame,
    dominance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dominance-weighted aggregate of purity, AP, and compactness across all active levels.

        effective_purity      = Σ_l  frac_l × mean_fg_purity_l
        effective_ap          = Σ_l  frac_l × mean_fg_ap_l
        effective_compactness = Σ_l  frac_l × mean_fg_compactness_l

    Using pixel-dominance fractions as weights makes this metric comparable across
    models with different active level sets — a model where L2 dominates 76% of
    pixels will be penalised for L2's poor purity regardless of L4's purity score.

    Parameters
    ----------
    purity_df, ap_df, compactness_df
        Outputs of compute_purity / compute_per_level_ap / compute_compactness.
        Foreground rows only (background excluded by those functions already).
    dominance_df
        Single-row output of compute_level_dominance.

    Returns
    -------
    Single-row DataFrame with columns:
        effective_purity, effective_ap, effective_compactness
        + per-level breakdown: weight_lN, purity_lN, ap_lN, compact_lN
    """
    purity_l  = purity_df.groupby("level")["purity"].mean().to_dict()
    ap_l      = ap_df.groupby("level")["ap"].mean().to_dict()
    compact_l = compactness_df.groupby("level")["compactness"].mean().to_dict()

    active_levels = sorted(
        int(col.split("_l")[1])
        for col in dominance_df.columns
        if col.startswith("frac_l") and dominance_df[col].values[0] > 0
    )

    row: dict = {}
    eff_purity = eff_ap = eff_compact = 0.0
    for l in active_levels:
        w = float(dominance_df[f"frac_l{l}"].values[0])
        p = purity_l.get(l, float("nan"))
        a = ap_l.get(l, float("nan"))
        c = compact_l.get(l, float("nan"))
        row[f"weight_l{l}"]  = w
        row[f"purity_l{l}"]  = p
        row[f"ap_l{l}"]      = a
        row[f"compact_l{l}"] = c
        if not np.isnan(p): eff_purity  += w * p
        if not np.isnan(a): eff_ap      += w * a
        if not np.isnan(c): eff_compact += w * c

    row["effective_purity"]      = eff_purity
    row["effective_ap"]          = eff_ap
    row["effective_compactness"] = eff_compact
    return pd.DataFrame([row])


# ── Atlas Builder ─────────────────────────────────────────────────────────────

@torch.no_grad()
def build_prototype_atlas(
    model: torch.nn.Module,
    train_loader,
    level: int,
) -> plt.Figure:
    """
    For each prototype at the given level, find the training slice with the
    highest peak activation. Crop the corresponding image region and overlay
    the GT label contour.

    Grid layout: rows = foreground classes (7), cols = prototypes per class (M_l).

    Returns a matplotlib Figure (call fig.savefig(...) to save).
    """
    device = _device(model)
    model.eval()

    pl = model.proto_layers_dict()[level]
    K, M = pl.n_classes, pl.n_protos
    stride = LEVEL_STRIDES[level]
    half = stride * 2        # crop half-width in original image pixels

    # best[(k, m)] = (best_val, image_crop, label_crop)
    best: dict[tuple, tuple] = {}

    for batch in train_loader:
        images_cpu = batch["image"]                      # keep on CPU for numpy crop
        labels_cpu = batch["label"]
        images_dev = images_cpu.to(device)
        feat = model.encoder(images_dev)
        A = pl(feat[level])                              # (B, K, M, H_l, W_l)
        H_l, W_l = A.shape[-2], A.shape[-1]
        B = images_cpu.shape[0]

        for k in FOREGROUND:
            for m in range(M):
                act = A[:, k, m, :, :]                   # (B, H_l, W_l)
                for b in range(B):
                    val = act[b].max().item()
                    if val > best.get((k, m), (-1, None, None))[0]:
                        flat_idx = act[b].argmax().item()
                        pi, pj = flat_idx // W_l, flat_idx % W_l
                        # map to original image coordinates
                        ci = int(pi * stride + stride // 2)
                        cj = int(pj * stride + stride // 2)
                        # crop with boundary clamping
                        img_np = images_cpu[b, 0].numpy()  # (256, 256)
                        lbl_np = labels_cpu[b].numpy()     # (256, 256)
                        r0 = max(0, ci - half); r1 = min(256, ci + half)
                        c0 = max(0, cj - half); c1 = min(256, cj + half)
                        best[(k, m)] = (
                            val,
                            img_np[r0:r1, c0:c1],
                            lbl_np[r0:r1, c0:c1],
                        )

    # ── Draw grid ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(FOREGROUND), M,
        figsize=(M * 2.5, len(FOREGROUND) * 2.5),
    )
    if M == 1:
        axes = axes[:, np.newaxis]

    for row, k in enumerate(FOREGROUND):
        for col in range(M):
            ax = axes[row, col]
            entry = best.get((k, col))
            if entry is not None:
                _, img_crop, lbl_crop = entry
                ax.imshow(img_crop, cmap="gray", aspect="auto")
                ax.contour(lbl_crop == k, levels=[0.5], colors="red", linewidths=1)
                purity_val = "—"  # filled in by notebook if purity df is available
                ax.set_title(
                    f"{CLASS_NAMES[k]} / p{col}\nact={entry[0]:.3f}",
                    fontsize=7,
                )
            else:
                ax.set_visible(False)
            ax.axis("off")

    fig.suptitle(f"Prototype Atlas — Level {level} (stride ×{stride})", fontsize=11)
    plt.tight_layout()
    return fig
