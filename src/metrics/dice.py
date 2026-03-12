"""
src/metrics/dice.py
Dice coefficient computation for segmentation evaluation.
"""

import torch
from src.data.mmwhs_dataset import LABEL_NAMES


def dice_per_class(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int = 8,
    smooth: float = 1e-5,
) -> dict[str, float]:
    """
    Compute per-class Dice coefficient from logits and integer labels.

    Args:
        logits: (B, C, H, W) or (B, C, S, H, W) for patient volumes
        labels: (B, H, W) or (B, S, H, W)
        n_classes: number of segmentation classes
        smooth: Laplace smoothing to avoid zero division

    Returns:
        dict mapping class name → Dice float (NaN if class absent in GT)
    """
    preds = logits.argmax(dim=1)  # (B, H, W)

    result = {}
    for c in range(n_classes):
        pred_c = (preds == c).float()
        gt_c = (labels == c).float()
        intersection = (pred_c * gt_c).sum()
        denom = pred_c.sum() + gt_c.sum()
        if denom < smooth:
            result[LABEL_NAMES[c]] = float("nan")
        else:
            result[LABEL_NAMES[c]] = (
                (2.0 * intersection + smooth) / (denom + smooth)
            ).item()
    return result


def mean_foreground_dice(dice_dict: dict[str, float]) -> float:
    """Mean Dice over foreground classes (excludes Background), ignoring NaN."""
    vals = [v for k, v in dice_dict.items() if k != "Background" and not (v != v)]
    return sum(vals) / len(vals) if vals else 0.0
