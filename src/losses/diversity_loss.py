"""
src/losses/diversity_loss.py
Stage 4 — Diversity Loss (Jeffrey's Divergence)
Stage 8 — Push-Pull Prototype Alignment Loss

Penalizes intra-class prototype similarity to prevent prototype collapse.

Jeffrey's divergence (symmetric KL):
    D_J(P || Q) = KL(P || Q) + KL(Q || P)

Loss across all encoder levels and foreground classes:
    L_div = Σ_l Σ_{k≠0} Σ_{m≠n} [ 1 / (D_J(A_{l,k,m} || A_{l,k,n}) + eps) ]

Push-pull prototype alignment:
    L_push = -mean_{fg pixels of class k} max_m A_{l,k,m}   (push activation up over GT)
    L_pull = +mean_{bg pixels of class k} max_m A_{l,k,m}   (pull activation down elsewhere)

Combined training loss (for ProtoSegNet):
    L_total = 0.5*L_dice + 0.5*L_wce + lambda_div*L_div + lambda_push*L_push + lambda_pull*L_pull
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

    Uses exp(-D_J) per pair (bounded in [0, 1]) — identical to the ProtoSeg
    repo formulation.  Minimising exp(-D_J) maximises Jeffrey's divergence
    between prototype heatmap distributions, encouraging diverse prototypes.

    Compared to the previous 1/(D_J+eps) formulation:
      - Bounded: when D_J=0, loss=1.0  (not 1/eps = 1e8)
      - Safe with larger λ_div values (e.g. 0.01)

    Args:
        A_dict     : { level_int : heatmap tensor (B, K, M, H_l, W_l) }
                     as returned by PrototypeLayer per level.
        exclude_bg : if True, skip class k=0 (background).
        eps        : numerical stability for KL divergence internals.

    Returns:
        Scalar loss tensor in [0, n_pairs], requires_grad=True when inputs do.
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
                    # exp(-D_J) ∈ [0,1]: minimising pushes D_J → large
                    total_loss = total_loss + torch.exp(-d_j)

    return total_loss.squeeze()


def _first_device(A_dict: dict) -> torch.device:
    """Return the device of the first tensor in the dict."""
    for v in A_dict.values():
        return v.device
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8 — Push-pull prototype alignment loss
# ─────────────────────────────────────────────────────────────────────────────


def prototype_push_pull_loss(
    A_dict: dict[int, torch.Tensor],
    labels: torch.Tensor,
    exclude_bg: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Push-pull prototype alignment loss.

    For each level l and foreground class k:
        act_{l,k} = max_m A_{l,k,m}  upsampled to input resolution
        L_push += -mean(act_{l,k} over GT foreground pixels of class k)
        L_pull += +mean(act_{l,k} over GT background pixels of class k)

    Minimising L_push drives heatmaps to activate strongly over the GT region.
    Minimising L_pull drives heatmaps to suppress activation elsewhere.

    Args:
        A_dict     : {level: (B, K, M, H_l, W_l)} prototype heatmaps
        labels     : (B, H, W) integer GT class labels
        exclude_bg : if True, skip class k=0
        eps        : guard against empty foreground/background

    Returns:
        (push_loss, pull_loss) — both scalar tensors, each already averaged over
        the number of (level, class) terms.
    """
    device = labels.device
    B, H_in, W_in = labels.shape
    K = next(iter(A_dict.values())).shape[1]
    k_start = 1 if exclude_bg else 0

    push_total = torch.zeros(1, device=device)
    pull_total = torch.zeros(1, device=device)
    n_terms = 0

    for A in A_dict.values():
        # A: (B, K, M, H_l, W_l)
        # Max over prototype dim → (B, K, H_l, W_l)
        act, _ = A.max(dim=2)
        # Upsample to input resolution
        act_up = F.interpolate(
            act, size=(H_in, W_in), mode="bilinear", align_corners=False
        )  # (B, K, H_in, W_in)

        for k in range(k_start, K):
            fg = (labels == k).float()   # (B, H_in, W_in)
            bg = 1.0 - fg
            fg_count = fg.sum() + eps
            bg_count = bg.sum() + eps

            act_k = act_up[:, k, :, :]  # (B, H_in, W_in)

            # Push: minimise → increase activation over fg (loss is negative)
            push_total = push_total - (act_k * fg).sum() / fg_count
            # Pull: minimise → decrease activation over bg (loss is positive)
            pull_total = pull_total + (act_k * bg).sum() / bg_count
            n_terms += 1

    if n_terms > 0:
        push_total = push_total / n_terms
        pull_total = pull_total / n_terms

    return push_total.squeeze(), pull_total.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss for ProtoSegNet training
# ─────────────────────────────────────────────────────────────────────────────


class ProtoSegLoss:
    """
    Combined loss:
        L_total = 0.5*L_dice + 0.5*L_wce
                + lambda_div  * L_div
                + lambda_push * L_push
                + lambda_pull * L_pull
                + lambda_alc  * L_ALC   (optional, default 0.0 = off)

    Args:
        seg_loss      : SegmentationLoss instance (Dice + WeightedCE)
        lambda_div    : weight on diversity loss (default 0.01)
        lambda_push   : weight on push alignment loss (default 0.0 = off)
        lambda_pull   : weight on pull alignment loss (default 0.0 = off)
        exclude_bg    : exclude background class from diversity and push-pull
        lambda_alc    : weight on ALC loss (default 0.0 = off)
        alc_mu        : (K, 2) anatomical priors for ALC; required if lambda_alc > 0
        alc_levels    : list of level keys to apply ALC to (e.g. [3, 4])
    """

    def __init__(
        self,
        seg_loss,
        lambda_div: float = 0.01,
        lambda_push: float = 0.0,
        lambda_pull: float = 0.0,
        exclude_bg: bool = True,
        lambda_alc: float = 0.0,
        alc_mu=None,
        alc_levels=None,
    ):
        self.seg_loss = seg_loss
        self.lambda_div = lambda_div
        self.lambda_push = lambda_push
        self.lambda_pull = lambda_pull
        self.exclude_bg = exclude_bg
        self.lambda_alc = lambda_alc
        self.alc_mu = alc_mu
        self.alc_levels = alc_levels or []

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
            dict with keys: 'loss', 'dice_loss', 'ce_loss', 'div_loss',
                            'push_loss', 'pull_loss', 'alc_loss'
        """
        seg_out = self.seg_loss(logits, labels)
        div_loss = prototype_diversity_loss(A_dict, exclude_bg=self.exclude_bg)
        push_loss, pull_loss = prototype_push_pull_loss(
            A_dict, labels, exclude_bg=self.exclude_bg
        )
        total = (
            seg_out["loss"]
            + self.lambda_div  * div_loss
            + self.lambda_push * push_loss
            + self.lambda_pull * pull_loss
        )

        alc_l = torch.zeros(1, device=logits.device)
        if self.lambda_alc > 0.0 and self.alc_mu is not None and self.alc_levels:
            from src.losses.alc_loss import alc_loss as _alc_loss
            alc_l = _alc_loss(
                A_dict,
                self.alc_mu,
                self.alc_levels,
                foreground=list(range(1, logits.shape[1])),
            )
            total = total + self.lambda_alc * alc_l

        return {
            "loss": total,
            "dice_loss": seg_out["dice_loss"],
            "ce_loss": seg_out["ce_loss"],
            "div_loss": div_loss,
            "push_loss": push_loss,
            "pull_loss": pull_loss,
            "alc_loss": alc_l,
        }
