"""
src/losses/alc_loss.py
Stage 33 — Anatomical Locality Constraint (ALC)

Each prototype's activation centroid should lie near the expected anatomical
location of the class it represents. For cardiac CT/MR, structural positions
are consistent across patients.

    L_ALC = Σ_{k ∈ FG, m, l ∈ active_levels}  || centroid(A_{k,m,l}) - μ_k ||²

    centroid(A) = Σ_{x,y} A(x,y) · (x,y) / Σ A(x,y)   ← differentiable soft-argmax
    μ_k         = mean centroid of class k over training set (precomputed, fixed)

Coordinates are normalised to [0, 1] relative to image size so that the same
μ_k values apply across resolutions and (approximately) across modalities.

Usage:
    # Precompute priors once per modality
    mu = compute_anatomical_priors(train_loader, n_classes=8)  # (K, 2), CPU
    torch.save(mu, "results/v8/anatomical_priors_ct.pt")

    # During training (Phase B onwards, after prototype projection)
    loss = alc_loss(heatmaps, mu.to(device), active_levels=[3, 4], foreground=range(1, 8))
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Anatomical prior computation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_anatomical_priors(
    train_loader,
    n_classes: int = 8,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Precompute the mean centroid μ_k for each class from training labels.

    Centroid of class k in a label map is the mean (y, x) position of all
    pixels with label k, normalised to [0, 1] by dividing by (H-1) and (W-1).

    Args:
        train_loader : DataLoader yielding dicts with key "label" (B, H, W)
        n_classes    : total number of classes (including background k=0)
        device       : where to run computation (usually "cpu")

    Returns:
        mu : (n_classes, 2) float tensor — normalised (y, x) ∈ [0, 1]
             Row k=0 (background) is set to (0.5, 0.5) but unused by ALC.
    """
    sum_y = torch.zeros(n_classes, dtype=torch.float64, device=device)
    sum_x = torch.zeros(n_classes, dtype=torch.float64, device=device)
    counts = torch.zeros(n_classes, dtype=torch.float64, device=device)

    for batch in train_loader:
        labels = batch["label"]  # (B, H, W), integer
        if labels.device.type != device:
            labels = labels.to(device)
        B, H, W = labels.shape

        # Build normalised coordinate grids
        ys = torch.linspace(0.0, 1.0, H, device=device)  # (H,)
        xs = torch.linspace(0.0, 1.0, W, device=device)  # (W,)
        grid_y = ys.unsqueeze(1).expand(H, W)  # (H, W)
        grid_x = xs.unsqueeze(0).expand(H, W)  # (H, W)

        for k in range(n_classes):
            mask = (labels == k).float()  # (B, H, W)
            n_pixels = mask.sum(dim=(1, 2))  # (B,)
            valid = n_pixels > 0
            if not valid.any():
                continue

            cy = (mask * grid_y.unsqueeze(0)).sum(dim=(1, 2)) / (n_pixels + 1e-8)
            cx = (mask * grid_x.unsqueeze(0)).sum(dim=(1, 2)) / (n_pixels + 1e-8)

            sum_y[k] += cy[valid].sum().double()
            sum_x[k] += cx[valid].sum().double()
            counts[k] += valid.float().sum().double()

    # Mean centroid per class
    mu = torch.zeros(n_classes, 2, dtype=torch.float32)
    for k in range(n_classes):
        if counts[k] > 0:
            mu[k, 0] = float(sum_y[k] / counts[k])
            mu[k, 1] = float(sum_x[k] / counts[k])
        else:
            mu[k, 0] = 0.5
            mu[k, 1] = 0.5  # default if class absent

    return mu  # (K, 2)


def anatomical_priors_to_csv(mu: torch.Tensor, path: str, class_names: list[str]) -> None:
    """Save μ_k table as CSV for inspection."""
    rows = []
    for k, name in enumerate(class_names):
        rows.append({
            "class_idx": k,
            "class_name": name,
            "centroid_y": float(mu[k, 0]),
            "centroid_x": float(mu[k, 1]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable centroid (soft-argmax)
# ─────────────────────────────────────────────────────────────────────────────


def _soft_centroid(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Differentiable centroid of a 2D heatmap via soft-argmax.

    Args:
        heatmap : (..., H, W) — non-negative activation map

    Returns:
        centroid : (..., 2) — normalised (y, x) ∈ [0, 1]
    """
    *prefix, H, W = heatmap.shape

    # Normalised coordinate grids
    ys = torch.linspace(0.0, 1.0, H, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.linspace(0.0, 1.0, W, device=heatmap.device, dtype=heatmap.dtype)
    grid_y = ys.view(1, H, 1).expand(*[1] * len(prefix), H, W)
    grid_x = xs.view(1, 1, W).expand(*[1] * len(prefix), H, W)

    # Normalise heatmap to a probability distribution (sum = 1 over spatial dims)
    flat = heatmap.reshape(*prefix, H * W)
    weights = F.softmax(flat, dim=-1).reshape(*prefix, H, W)  # (..., H, W)

    cy = (weights * grid_y).sum(dim=(-2, -1))  # (...,)
    cx = (weights * grid_x).sum(dim=(-2, -1))  # (...,)

    return torch.stack([cy, cx], dim=-1)  # (..., 2)


# ─────────────────────────────────────────────────────────────────────────────
# ALC loss
# ─────────────────────────────────────────────────────────────────────────────


def alc_loss(
    heatmaps: dict[int, torch.Tensor],
    mu: torch.Tensor,
    active_levels: list[int],
    foreground: range | list[int] = range(1, 8),
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Anatomical Locality Constraint loss.

    For each active level l and foreground class k, penalises the squared
    Euclidean distance between each prototype's soft centroid and μ_k.

    Args:
        heatmaps      : {level: (B, K, M, H_l, W_l)} prototype activation maps
        mu            : (K, 2) precomputed anatomical priors, normalised [0,1]
        active_levels : list of level keys in heatmaps to apply ALC to
        foreground    : foreground class indices (default 1–7, skip background)
        eps           : guard against zero-activation heatmaps

    Returns:
        Scalar ALC loss (mean over all (level, class, proto, batch) terms).
    """
    device = mu.device
    total = torch.zeros(1, device=device, dtype=torch.float32)
    n_terms = 0

    for level in active_levels:
        if level not in heatmaps:
            continue
        A = heatmaps[level]  # (B, K, M, H_l, W_l)
        B, K, M, H_l, W_l = A.shape

        for k in foreground:
            if k >= K:
                continue
            mu_k = mu[k].to(device)  # (2,)

            # A_k: (B, M, H_l, W_l)
            A_k = A[:, k, :, :, :]

            # Clamp to non-negative (heatmaps should already be non-negative,
            # but guard against numerical issues)
            A_k = A_k.clamp(min=0.0)

            # Add small eps to avoid all-zero heatmaps
            A_k = A_k + eps

            # Soft centroid: (B, M, 2)
            centroids = _soft_centroid(A_k)  # (B, M, 2)

            # Squared distance to anatomical prior: (B, M)
            diff = centroids - mu_k.unsqueeze(0).unsqueeze(0)  # (B, M, 2)
            sq_dist = (diff ** 2).sum(dim=-1)  # (B, M)

            total = total + sq_dist.sum()
            n_terms += B * M

    if n_terms > 0:
        total = total / n_terms

    return total.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Centroid deviation metric (no gradient)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_centroid_deviation(
    model: torch.nn.Module,
    loader,
    mu: torch.Tensor,
    active_levels: list[int],
    foreground: range | list[int] = range(1, 8),
    image_size: int = 256,
) -> pd.DataFrame:
    """
    Compute mean centroid deviation (in pixels) per level on a dataset.

    centroid_deviation_l = mean over (k, m, batch) of ||centroid(A_{k,m,l}) - μ_k|| * image_size

    Args:
        model        : ProtoSegNet (eval mode)
        loader       : DataLoader with "image" key
        mu           : (K, 2) normalised anatomical priors
        active_levels: levels to evaluate
        foreground   : foreground class indices
        image_size   : pixel scale (default 256 for 256×256 images)

    Returns:
        DataFrame with columns: level, mean_deviation_px, n_terms
    """
    device = next(model.parameters()).device
    mu_dev = mu.to(device)
    model.eval()

    deviations: dict[int, list[float]] = {l: [] for l in active_levels}

    for batch in loader:
        images = batch["image"].to(device)
        feat = model.encoder(images)

        for level in active_levels:
            if str(level) not in model.proto_layers:
                continue
            pl = model.proto_layers[str(level)]
            A = pl(feat[level])  # (B, K, M, H_l, W_l)
            B, K, M, H_l, W_l = A.shape

            for k in foreground:
                if k >= K:
                    continue
                A_k = A[:, k, :, :, :].clamp(min=0.0) + 1e-8  # (B, M, H_l, W_l)
                centroids = _soft_centroid(A_k)  # (B, M, 2)
                mu_k = mu_dev[k].unsqueeze(0).unsqueeze(0)      # (1, 1, 2)
                dist_px = ((centroids - mu_k) ** 2).sum(-1).sqrt() * image_size  # (B, M)
                deviations[level].extend(dist_px.flatten().tolist())

    rows = []
    for level in active_levels:
        devs = deviations[level]
        rows.append({
            "level": level,
            "mean_deviation_px": float(np.mean(devs)) if devs else float("nan"),
            "n_terms": len(devs),
        })
    return pd.DataFrame(rows)
