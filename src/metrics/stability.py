"""
src/metrics/stability.py
Stage 6 — Lipschitz Stability

Stability = max_{X' ∈ N_σ(X)} [ ||Φ(X) − Φ(X')||_2 / ||X − X'||_2 ]

Φ(X)  = aggregated prototype activation map (flattened to a vector)
N_σ(X) = neighbourhood sampled as X + ε, ε ~ N(0, σ²)

Estimated using N=20 i.i.d. Gaussian perturbations per slice.
All perturbations are batched for efficiency.

Lower Stability = better (explanation is more robust to input noise).
"""

import torch
import torch.nn as nn
import numpy as np

from src.metrics.xai_utils import aggregate_heatmaps

N_PERTURB = 20
SIGMA = 0.05


def _phi(heatmaps: dict[int, torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
    """Compute Φ(X) = aggregated heatmap, flattened to (B, K*H*W)."""
    A = aggregate_heatmaps(heatmaps, target_size=target_size)  # (B, K, H, W)
    return A.flatten(start_dim=1)                               # (B, K*H*W)


def _stability_single_slice(
    model: nn.Module,
    x: torch.Tensor,                    # (1, 1, H, W), on device
    heatmaps: dict[int, torch.Tensor],  # pre-computed for x
    device: torch.device,
    n_perturb: int = N_PERTURB,
    sigma: float = SIGMA,
) -> float:
    """Lipschitz stability estimate for one slice (call inside torch.no_grad)."""
    H, W = x.shape[-2], x.shape[-1]

    # Φ(X): activation vector for original input
    phi_x = _phi(heatmaps, target_size=(H, W))  # (1, K*H*W)

    # Sample N perturbations and batch them
    eps = torch.randn(n_perturb, 1, H, W, device=device) * sigma
    x_pert = (x + eps).clamp(0.0, 1.0)                         # (N, 1, H, W)

    # Single batched forward pass for all perturbations
    _, heatmaps_pert = model(x_pert)
    phi_xp = _phi(heatmaps_pert, target_size=(H, W))            # (N, K*H*W)

    # ||Φ(X) − Φ(X')||_2 for each perturbation
    phi_diff = (phi_x - phi_xp).norm(dim=1)                    # (N,)

    # ||X − X'||_2 = ||ε||_2
    x_diff = eps.flatten(start_dim=1).norm(dim=1)               # (N,)

    # Avoid zero denominator
    valid = x_diff > 1e-8
    if not valid.any():
        return float("nan")

    ratios = phi_diff[valid] / x_diff[valid]
    return ratios.max().item()


@torch.no_grad()
def stability_patient(
    model: nn.Module,
    images: torch.Tensor,           # (S, 1, H, W)
    device: torch.device,
    n_perturb: int = N_PERTURB,
    sigma: float = SIGMA,
    max_slices: int | None = None,
) -> dict[str, float]:
    """
    Compute Lipschitz Stability for a full patient.

    Args:
        model      : ProtoSegNet in eval mode
        images     : (S, 1, H, W)
        device     : inference device
        n_perturb  : Gaussian perturbations per slice
        sigma      : noise std (in normalised intensity scale)
        max_slices : if set, uniformly sample this many slices

    Returns:
        {'stability': mean max-ratio,  'stability_std': std over slices}
    """
    model.eval()
    S = images.shape[0]
    indices = (
        torch.linspace(0, S - 1, max_slices).long()
        if max_slices is not None
        else torch.arange(S)
    )

    ratios: list[float] = []
    for i in indices.tolist():
        x = images[i : i + 1].to(device)   # (1, 1, H, W)
        _, heatmaps = model(x)
        r = _stability_single_slice(model, x, heatmaps, device, n_perturb, sigma)
        if r == r:                          # not nan
            ratios.append(r)

    arr = np.array(ratios) if ratios else np.array([float("nan")])
    return {
        "stability": float(np.nanmean(arr)),
        "stability_std": float(np.nanstd(arr)),
    }
