"""
src/losses/segmentation.py
Combined segmentation loss: soft Dice (foreground only) + class-weighted Cross-Entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss averaged over foreground classes only (excludes background class 0).
    Uses one-hot encoding of labels.
    """

    def __init__(self, n_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) — raw unnormalized scores
            labels: (B, H, W) — integer class indices
        Returns:
            scalar Dice loss (mean over foreground classes)
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        B, C, H, W = probs.shape

        # One-hot encode labels: (B, H, W) → (B, C, H, W)
        labels_onehot = F.one_hot(labels, num_classes=C)        # (B, H, W, C)
        labels_onehot = labels_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dice_per_class = []
        for c in range(1, C):  # skip background (class 0)
            p = probs[:, c]          # (B, H, W)
            g = labels_onehot[:, c]  # (B, H, W)
            intersection = (p * g).sum()
            denom = p.sum() + g.sum()
            dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            dice_per_class.append(dice)

        return 1.0 - torch.stack(dice_per_class).mean()


class WeightedCELoss(nn.Module):
    """
    Cross-entropy loss with per-class weights to handle severe class imbalance.
    Weights are passed at construction time (computed from training set label distribution).
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            labels: (B, H, W)
        """
        return F.cross_entropy(logits, labels, weight=self.class_weights)


class SegmentationLoss(nn.Module):
    """
    Combined loss: 0.5 * DiceLoss + 0.5 * WeightedCELoss.
    """

    def __init__(self, class_weights: torch.Tensor, n_classes: int = 8):
        super().__init__()
        self.dice = DiceLoss(n_classes=n_classes)
        self.ce = WeightedCELoss(class_weights=class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Returns dict with 'loss', 'dice_loss', 'ce_loss' for logging.
        """
        dice_loss = self.dice(logits, labels)
        ce_loss = self.ce(logits, labels)
        total = 0.5 * dice_loss + 0.5 * ce_loss
        return {"loss": total, "dice_loss": dice_loss, "ce_loss": ce_loss}


def compute_class_weights(data_dir: str, modality: str) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training set.
    Returns a (8,) float tensor.
    """
    import numpy as np
    from pathlib import Path

    npz_dir = Path(data_dir) / f"{modality}_256" / "train" / "npz"
    counts = np.zeros(8, dtype=np.int64)
    for f in npz_dir.glob("*.npz"):
        label = np.load(f)["label"]
        for c in range(8):
            counts[c] += (label == c).sum()

    # Inverse frequency: w_c = total / (n_classes * count_c)
    total = counts.sum()
    weights = total / (8.0 * counts.clip(min=1))
    weights = weights / weights.sum() * 8  # normalize so weights sum to n_classes
    return torch.tensor(weights, dtype=torch.float32)
