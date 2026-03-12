"""
src/metrics/incremental_deletion.py
Stage 6 — Incremental Deletion Score (IDS)

IDS = AUC of mean_foreground_Dice(t) as top-t% activated pixels are zeroed,
      t ∈ {5, 10, ..., 100}%.

Lower IDS = better explanation:
  a good explanation concentrates on pixels the model truly relies on,
  so zeroing the most-activated pixels should cause the sharpest Dice drop.

AUC is normalised to [0, 1] via trapezoid rule.
"""

import torch
import torch.nn as nn
import numpy as np

from src.metrics.xai_utils import aggregate_heatmaps
from src.metrics.dice import dice_per_class, mean_foreground_dice

THRESHOLDS = list(range(5, 105, 5))  # [5, 10, ..., 100]


def _ids_single_slice(
    model: nn.Module,
    x: torch.Tensor,                        # (1, 1, H, W)
    y: torch.Tensor,                        # (1, H, W)
    heatmaps: dict[int, torch.Tensor],
    device: torch.device,
    thresholds: list[int] = THRESHOLDS,
) -> float:
    """AUC of deletion curve for one slice (must be called inside torch.no_grad)."""
    H, W = x.shape[-2], x.shape[-1]

    # Global activation: max over classes
    A = aggregate_heatmaps(heatmaps, target_size=(H, W))   # (1, K, H, W)
    A_global = A[0].max(dim=0).values                       # (H, W)
    flat_act = A_global.flatten()                           # (H*W,)
    sorted_idx = flat_act.argsort(descending=True)          # pixel indices, high → low
    n_pixels = flat_act.numel()

    x_flat = x[0, 0].flatten().clone()     # (H*W,)

    dice_curve: list[float] = []
    for t in thresholds:
        n_zero = int(n_pixels * t / 100)
        x_del = x_flat.clone()
        if n_zero > 0:
            x_del[sorted_idx[:n_zero]] = 0.0
        x_in = x_del.reshape(1, 1, H, W)
        logits_p, _ = model(x_in)
        d = dice_per_class(logits_p, y)
        dice_curve.append(mean_foreground_dice(d))

    # Normalised AUC (xs span 0.05 → 1.0)
    xs = [t / 100.0 for t in thresholds]
    auc = float(np.trapezoid(dice_curve, xs)) / (xs[-1] - xs[0])
    return auc


@torch.no_grad()
def incremental_deletion_patient(
    model: nn.Module,
    images: torch.Tensor,           # (S, 1, H, W)
    labels: torch.Tensor,           # (S, H, W)
    device: torch.device,
    max_slices: int | None = None,
) -> dict[str, float]:
    """
    Compute IDS for a full patient.

    Args:
        model      : ProtoSegNet in eval mode
        images     : (S, 1, H, W)
        labels     : (S, H, W)
        device     : inference device
        max_slices : if set, uniformly sample this many slices (speed/memory)

    Returns:
        {'ids': mean AUC,  'ids_std': std over slices}
    """
    model.eval()
    S = images.shape[0]
    indices = (
        torch.linspace(0, S - 1, max_slices).long()
        if max_slices is not None
        else torch.arange(S)
    )

    ids_vals: list[float] = []
    for i in indices.tolist():
        x = images[i : i + 1].to(device)   # (1, 1, H, W)
        y = labels[i : i + 1].to(device)   # (1, H, W)
        _, heatmaps = model(x)
        ids = _ids_single_slice(model, x, y, heatmaps, device)
        ids_vals.append(ids)

    arr = np.array(ids_vals)
    return {"ids": float(arr.mean()), "ids_std": float(arr.std())}
