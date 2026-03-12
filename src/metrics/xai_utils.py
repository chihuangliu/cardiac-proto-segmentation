"""
src/metrics/xai_utils.py
Shared utilities for XAI metric computation.
"""
import torch
import torch.nn.functional as F


def aggregate_heatmaps(
    heatmaps_dict: dict[int, torch.Tensor],
    target_size: tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    Aggregate multi-level prototype heatmaps into a single (B, K, H, W) map.

    For each level: max over M prototypes, bilinear upsample to target_size.
    Across levels: elementwise max.

    Args:
        heatmaps_dict : {level: (B, K, M, H_l, W_l)}  log-cosine-similarity heatmaps
        target_size   : spatial resolution of output (H, W)

    Returns:
        (B, K, H, W)  per-class activation at full resolution
    """
    agg: torch.Tensor | None = None
    for A in heatmaps_dict.values():    # A: (B, K, M, H_l, W_l)
        A_max = A.max(dim=2).values     # (B, K, H_l, W_l)
        A_up = F.interpolate(A_max, size=target_size, mode="bilinear", align_corners=False)
        agg = A_up if agg is None else torch.maximum(agg, A_up)
    assert agg is not None
    return agg                          # (B, K, H, W)
