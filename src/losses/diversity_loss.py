"""
src/losses/diversity_loss.py
Stage 4 — Diversity Loss (Jeffrey's Divergence)

Penalizes intra-class prototype similarity to prevent prototype collapse.

Jeffrey's divergence (symmetric KL):
    D_J(P || Q) = KL(P || Q) + KL(Q || P)

Loss across all encoder levels and foreground classes:
    L_div = Σ_l Σ_{k≠0} Σ_{m≠n} [ 1 / (D_J(A_{l,k,m} || A_{l,k,n}) + eps) ]

Combined training loss (for ProtoSegNet):
    L_total = 0.5 * L_dice + 0.5 * L_wce + lambda_div * L_div
"""

import torch
import torch.nn.functional as F


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(p || q) for two probability distributions.

    Args:
        p, q : (..., N) — already softmax-normalised, summing to 1 over last dim
        eps  : small constant for log stability
    Returns:
        scalar mean over batch dimension
    """
    # Clamp to avoid log(0); both p and q are non-negative after softmax
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1).mean()


def _jeffrey_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric Jeffrey's divergence: D_J(p || q) = KL(p || q) + KL(q || p).

    Args:
        p, q : (..., N)  probability distributions
    Returns:
        scalar >= 0
    """
    return _kl_divergence(p, q, eps) + _kl_divergence(q, p, eps)


def prototype_diversity_loss(
    A_dict: dict[int, torch.Tensor],
    exclude_bg: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Diversity loss over all encoder levels and prototype pairs within each class.

    Args:
        A_dict     : { level_int : heatmap tensor (B, K, M, H_l, W_l) }
                     as returned by PrototypeLayer per level.
        exclude_bg : if True, skip class k=0 (background).
        eps        : numerical stability term added to D_J in the denominator.

    Returns:
        Scalar loss tensor (requires_grad=True when A_dict tensors do).
    """
    total_loss = torch.zeros(1, device=_first_device(A_dict))

    for level, A in A_dict.items():
        # A : (B, K, M, H, W)
        B, K, M, H, W = A.shape

        if M < 2:
            # No pairs to penalize at this level
            continue

        k_start = 1 if exclude_bg else 0

        for k in range(k_start, K):
            # A_k : (B, M, H, W)  — heatmaps for class k
            A_k = A[:, k, :, :, :]

            # Flatten spatial dims and softmax-normalise → probability distribution
            # (B, M, H*W)
            A_k_flat = A_k.reshape(B, M, H * W)
            # Softmax over spatial pixels so each heatmap sums to 1
            P_k = F.softmax(A_k_flat, dim=-1)  # (B, M, H*W)

            # Iterate over all prototype pairs (m, n) with m < n
            for m in range(M):
                for n in range(m + 1, M):
                    p_m = P_k[:, m, :]  # (B, H*W)
                    p_n = P_k[:, n, :]  # (B, H*W)
                    d_j = _jeffrey_divergence(p_m, p_n, eps=eps)
                    # Penalise small divergence (encourage large D_J)
                    total_loss = total_loss + 1.0 / (d_j + eps)

    return total_loss.squeeze()


def _first_device(A_dict: dict) -> torch.device:
    """Return the device of the first tensor in the dict."""
    for v in A_dict.values():
        return v.device
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss for ProtoSegNet training
# ─────────────────────────────────────────────────────────────────────────────


class ProtoSegLoss:
    """
    Combined loss:  L_total = 0.5 * L_dice + 0.5 * L_wce + lambda_div * L_div

    Args:
        seg_loss   : SegmentationLoss instance (Dice + WeightedCE)
        lambda_div : weight on diversity loss (default 0.01; sweep [0.001, 0.01, 0.1])
        exclude_bg : exclude background class from diversity penalty
    """

    def __init__(self, seg_loss, lambda_div: float = 0.01, exclude_bg: bool = True):
        self.seg_loss = seg_loss
        self.lambda_div = lambda_div
        self.exclude_bg = exclude_bg

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        A_dict: dict[int, torch.Tensor],
    ) -> dict:
        """
        Args:
            logits  : (B, K, H, W)  raw segmentation logits
            labels  : (B, H, W)     integer class indices
            A_dict  : {level: (B, K, M, H_l, W_l)} prototype heatmaps

        Returns:
            dict with keys: 'loss', 'dice_loss', 'ce_loss', 'div_loss'
        """
        seg_out = self.seg_loss(logits, labels)
        div_loss = prototype_diversity_loss(A_dict, exclude_bg=self.exclude_bg)
        total = seg_out["loss"] + self.lambda_div * div_loss
        return {
            "loss": total,
            "dice_loss": seg_out["dice_loss"],
            "ce_loss": seg_out["ce_loss"],
            "div_loss": div_loss,
        }
