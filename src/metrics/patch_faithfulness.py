"""
src/metrics/patch_faithfulness.py
Patch-Level Faithfulness — aligned to prototype feature map resolution.

Addresses Barrier 2 (Resolution Problem): single-pixel perturbations at 256×256
cannot affect coarse feature maps (L4: 16×16). One zeroed pixel is 1/256 of the
information in a single L4 activation — effectively zero perturbation.

Fix: zero 16×16 blocks aligned to L4's spatial grid. Each block covers exactly
one feature map location. This makes the perturbation visible to the architecture.

Formula (same as pixel Faithfulness, different granularity):
    patch_faithfulness = Pearson(E_p, Δŷ_p)  over all P patches

    E_p   = max-over-classes aggregated heatmap at feature-map position p
    Δŷ_p  = drop in predicted-class probability at center pixel of patch p
              = prob(pred_class | x_orig)[center] − prob(pred_class | x_zeroed_p)[center]

block_size argument controls granularity:
    L4 (16×16 feature map): block_size = 16  → P = 256 patches
    L3 (32×32 feature map): block_size = 8   → P = 1024 patches
    L2 (64×64 feature map): block_size = 4   → P = 4096 patches

Models must return (logits, heatmaps, w) — use ProtoSegNetV2Adapter for ProtoSegNet.
"""

import torch
import torch.nn as nn
import numpy as np

from src.metrics.xai_utils import aggregate_heatmaps

INFER_BATCH = 32  # perturbed inputs per forward pass


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _patch_faithfulness_single_slice(
    model: nn.Module,
    x: torch.Tensor,           # (1, 1, H, W) on device
    heatmaps: dict,            # already computed for x
    device: torch.device,
    block_size: int = 16,
    infer_batch: int = INFER_BATCH,
) -> float:
    """Patch-level Faithfulness for one slice (call inside torch.no_grad)."""
    H, W = x.shape[-2], x.shape[-1]
    assert H % block_size == 0 and W % block_size == 0, (
        f"Image size ({H}×{W}) must be divisible by block_size ({block_size})"
    )
    n_ph = H // block_size   # number of patch rows
    n_pw = W // block_size   # number of patch cols
    P = n_ph * n_pw          # total patches

    # --- Heatmap importance: aggregate to (H, W), then pool to patch grid ---
    A = aggregate_heatmaps(heatmaps, target_size=(H, W))   # (1, K, H, W)
    A_global = A[0].max(dim=0).values                       # (H, W)
    # Pool heatmap to patch grid by taking max in each block
    A_patch = A_global.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    # A_patch: (n_ph, n_pw, block_size, block_size)
    E_patches = A_patch.amax(dim=(-1, -2)).flatten().cpu().numpy()  # (P,)

    # --- Original predictions ---
    logits_orig, _, _w = model(x)                                    # (1, K, H, W)
    probs_orig = logits_orig[0].softmax(dim=0)                       # (K, H, W)
    pred_class_map = logits_orig[0].argmax(dim=0)                    # (H, W)

    # Center pixel of each patch
    centers_r = np.array([pi * block_size + block_size // 2 for pi in range(n_ph)
                           for _  in range(n_pw)])   # (P,)
    centers_c = np.array([pj * block_size + block_size // 2 for _  in range(n_ph)
                           for pj in range(n_pw)])   # (P,)
    orig_probs_at_center = np.array([
        probs_orig[pred_class_map[r, c].item(), r, c].item()
        for r, c in zip(centers_r, centers_c)
    ], dtype=np.float32)   # (P,)

    # --- Perturb each patch, measure Δŷ at center ---
    patch_indices = np.arange(P)
    delta_yhat = np.zeros(P, dtype=np.float32)

    for start in range(0, P, infer_batch):
        end = min(start + infer_batch, P)
        b = end - start

        x_rep = x.expand(b, -1, -1, -1).clone()   # (b, 1, H, W)

        for j_local, p_idx in enumerate(range(start, end)):
            pi = p_idx // n_pw
            pj = p_idx % n_pw
            r0, r1 = pi * block_size, (pi + 1) * block_size
            c0, c1 = pj * block_size, (pj + 1) * block_size
            x_rep[j_local, 0, r0:r1, c0:c1] = 0.0

        logits_p, _, _w = model(x_rep)             # (b, K, H, W)
        probs_p = logits_p.softmax(dim=1)           # (b, K, H, W)

        for j_local, p_idx in enumerate(range(start, end)):
            r = int(centers_r[p_idx])
            c = int(centers_c[p_idx])
            cls = pred_class_map[r, c].item()
            new_prob = probs_p[j_local, cls, r, c].item()
            delta_yhat[p_idx] = orig_probs_at_center[p_idx] - new_prob

    return _pearson(E_patches, delta_yhat)


@torch.no_grad()
def patch_faithfulness_patient(
    model: nn.Module,
    images: torch.Tensor,     # (S, 1, H, W)
    device: torch.device,
    block_size: int = 16,
    infer_batch: int = INFER_BATCH,
    max_slices: int | None = None,
) -> dict[str, float]:
    """
    Compute Patch-level Faithfulness for a full patient.

    Args:
        model      : model with 3-output interface (logits, heatmaps, w)
        images     : (S, 1, H, W)
        device     : inference device
        block_size : block size in pixels; should match the coarsest feature level
                     L4=16, L3=8, L2=4
        infer_batch: perturbed inputs per forward pass
        max_slices : if set, uniformly sample this many slices

    Returns:
        {'patch_faithfulness': mean Pearson r, 'patch_faithfulness_std': std}
    """
    model.eval()
    S = images.shape[0]
    indices = (
        torch.linspace(0, S - 1, max_slices).long()
        if max_slices is not None
        else torch.arange(S)
    )

    corrs: list[float] = []
    for i in indices.tolist():
        x = images[i : i + 1].to(device)    # (1, 1, H, W)
        _, heatmaps, _w = model(x)
        r = _patch_faithfulness_single_slice(
            model, x, heatmaps, device, block_size, infer_batch
        )
        if r == r:   # not nan
            corrs.append(r)

    arr = np.array(corrs) if corrs else np.array([float("nan")])
    return {
        "patch_faithfulness": float(np.nanmean(arr)),
        "patch_faithfulness_std": float(np.nanstd(arr)),
    }
