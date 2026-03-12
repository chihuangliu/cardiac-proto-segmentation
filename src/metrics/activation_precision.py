"""
src/metrics/activation_precision.py
Stage 6 — Activation Precision (AP)

AP_k = |M_k ∩ G_k| / |M_k|
where  M_k = I(A_k > 95th-percentile of A_k)   (top-5% activation mask)
       G_k = binary ground-truth mask for class k

Computed per 2D slice, averaged over all patient slices.
Background class (k=0) is excluded.
"""

import torch
import torch.nn as nn

from src.data.mmwhs_dataset import LABEL_NAMES
from src.metrics.xai_utils import aggregate_heatmaps

N_CLASSES = 8
PERCENTILE = 95.0


def activation_precision_slice(
    heatmaps_dict: dict[int, torch.Tensor],
    labels: torch.Tensor,
    n_classes: int = N_CLASSES,
    percentile: float = PERCENTILE,
) -> dict[str, float]:
    """
    Per-class AP for a batch of slices.

    Args:
        heatmaps_dict : {level: (B, K, M, H_l, W_l)}
        labels        : (B, H, W)  integer ground-truth
        n_classes     : total number of classes (includes background)
        percentile    : threshold percentile for activation mask

    Returns:
        dict  class_name -> mean AP over batch  (nan if class absent in GT)
    """
    H, W = labels.shape[-2], labels.shape[-1]
    A = aggregate_heatmaps(heatmaps_dict, target_size=(H, W))  # (B, K, H, W)
    B = A.shape[0]

    result: dict[str, float] = {}
    for k in range(1, n_classes):          # skip background
        A_k = A[:, k, :, :]               # (B, H, W)
        G_k = (labels == k).float()        # (B, H, W)

        aps: list[float] = []
        for b in range(B):
            a_flat = A_k[b].flatten()
            thresh = torch.quantile(a_flat, percentile / 100.0)
            M_k = (A_k[b] >= thresh).float().flatten()
            if M_k.sum() < 1:
                continue
            ap = ((M_k * G_k[b].flatten()).sum() / M_k.sum()).item()
            aps.append(ap)

        result[LABEL_NAMES[k]] = sum(aps) / len(aps) if aps else float("nan")
    return result


@torch.no_grad()
def activation_precision_patient(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, float]:
    """
    Mean per-class AP for a full patient (stack of slices).

    Args:
        model      : ProtoSegNet in eval mode
        images     : (S, 1, H, W)
        labels     : (S, H, W)
        device     : inference device
        batch_size : slices per forward pass

    Returns:
        dict  class_name -> mean AP  (nan if class never present)
    """
    model.eval()
    S = images.shape[0]
    acc: dict[str, list[float]] = {LABEL_NAMES[k]: [] for k in range(1, N_CLASSES)}

    for start in range(0, S, batch_size):
        end = min(start + batch_size, S)
        x = images[start:end].to(device)    # (b, 1, H, W)
        y = labels[start:end].to(device)    # (b, H, W)
        _, heatmaps = model(x)
        slice_ap = activation_precision_slice(heatmaps, y)
        for cls, val in slice_ap.items():
            if val == val:                  # not nan
                acc[cls].append(val)

    return {
        cls: (sum(v) / len(v) if v else float("nan"))
        for cls, v in acc.items()
    }
