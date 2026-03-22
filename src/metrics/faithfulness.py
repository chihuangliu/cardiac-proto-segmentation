"""
src/metrics/faithfulness.py
Stage 6 — Faithfulness Correlation

Faithfulness = Pearson(E_i, Δŷ_i)  over N=2000 randomly sampled pixels per slice.

E_i    = prototype activation at pixel i  (max over classes of aggregated heatmap)
Δŷ_i  = drop in predicted-class probability at pixel i when pixel i is zeroed in input
         = softmax_prob(pred_class, i | x_orig) − softmax_prob(pred_class, i | x_zeroed_i)

Higher Faithfulness = better: the explanation highlights pixels that, when removed,
actually change the model's output.
"""

import torch
import torch.nn as nn
import numpy as np

from src.metrics.xai_utils import aggregate_heatmaps

N_PIXELS = 2000      # pixels sampled per slice
INFER_BATCH = 64     # perturbed inputs per forward pass


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _faithfulness_single_slice(
    model: nn.Module,
    x: torch.Tensor,                    # (1, 1, H, W), on device
    heatmaps: dict[int, torch.Tensor],  # already computed for x
    device: torch.device,
    n_pixels: int = N_PIXELS,
    infer_batch: int = INFER_BATCH,
    seed: int = 42,
) -> float:
    """Faithfulness for one slice (call inside torch.no_grad)."""
    H, W = x.shape[-2], x.shape[-1]
    n_total = H * W

    # --- Global activation map: max_k A_k at each pixel ---
    A = aggregate_heatmaps(heatmaps, target_size=(H, W))   # (1, K, H, W)
    A_global = A[0].max(dim=0).values.flatten()            # (H*W,)  on device
    E_flat = A_global.cpu().numpy()

    # --- Sample pixel indices ---
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(n_total, size=min(n_pixels, n_total), replace=False)
    E_sampled = E_flat[sampled_idx]

    # --- Original prediction ---
    logits_orig, _, _w = model(x)                               # (1, K, H, W)
    probs_orig = logits_orig[0].softmax(dim=0).reshape(logits_orig.shape[1], -1)  # (K, H*W)
    pred_class_flat = logits_orig[0].argmax(dim=0).flatten()    # (H*W,)

    # Original probabilities at sampled pixels for their predicted class
    sampled_t = torch.tensor(sampled_idx, device=device, dtype=torch.long)
    pred_cls_at_sampled = pred_class_flat[sampled_t]            # (N,)
    orig_prob_sampled = probs_orig[pred_cls_at_sampled, sampled_t].cpu().numpy()  # (N,)

    # --- Perturb each sampled pixel, measure Δŷ ---
    x_flat = x[0, 0].flatten()         # (H*W,)
    delta_yhat = np.zeros(len(sampled_idx), dtype=np.float32)

    for start in range(0, len(sampled_idx), infer_batch):
        end = min(start + infer_batch, len(sampled_idx))
        batch_pix = sampled_idx[start:end]   # numpy array of pixel indices
        b = len(batch_pix)

        # Build perturbed batch: (b, 1, H, W), each with one pixel zeroed
        x_rep = x.expand(b, -1, -1, -1).clone()            # (b, 1, H, W)
        x_rep_flat = x_rep.reshape(b, n_total)              # (b, H*W)
        pix_t = torch.tensor(batch_pix, device=device, dtype=torch.long)
        x_rep_flat[torch.arange(b, device=device), pix_t] = 0.0
        x_pert = x_rep_flat.reshape(b, 1, H, W)

        logits_p, _, _w = model(x_pert)                     # (b, K, H, W)
        probs_p = logits_p.softmax(dim=1)                   # (b, K, H, W)
        probs_p_flat = probs_p.reshape(b, logits_p.shape[1], -1)  # (b, K, H*W)

        for j in range(b):
            pix_idx = batch_pix[j]
            cls = pred_class_flat[pix_idx].item()
            new_prob = probs_p_flat[j, cls, pix_idx].item()
            delta_yhat[start + j] = orig_prob_sampled[start + j] - new_prob

    return _pearson(E_sampled, delta_yhat)


@torch.no_grad()
def faithfulness_patient(
    model: nn.Module,
    images: torch.Tensor,           # (S, 1, H, W)
    device: torch.device,
    n_pixels: int = N_PIXELS,
    infer_batch: int = INFER_BATCH,
    max_slices: int | None = None,
) -> dict[str, float]:
    """
    Compute Faithfulness for a full patient.

    Args:
        model      : ProtoSegNet in eval mode
        images     : (S, 1, H, W)
        device     : inference device
        n_pixels   : pixels sampled per slice
        infer_batch: perturbed inputs per forward pass
        max_slices : if set, uniformly sample this many slices

    Returns:
        {'faithfulness': mean Pearson r,  'faithfulness_std': std over slices}
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
        x = images[i : i + 1].to(device)   # (1, 1, H, W)
        _, heatmaps, _w = model(x)
        r = _faithfulness_single_slice(model, x, heatmaps, device, n_pixels, infer_batch)
        if r == r:                          # not nan
            corrs.append(r)

    arr = np.array(corrs) if corrs else np.array([float("nan")])
    return {
        "faithfulness": float(np.nanmean(arr)),
        "faithfulness_std": float(np.nanstd(arr)),
    }
