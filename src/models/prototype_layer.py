"""
src/models/prototype_layer.py
Stage 3 — Prototype Layer
Stage 8 — L2 distance similarity (replaces log-cosine)

PrototypeLayer       : learnable K×M prototype matrices + L2 similarity heatmaps
SoftMaskModule       : aggregate per-class heatmap → spatial attention mask on Z_l
PrototypeProjection  : push each prototype to the nearest real training feature vector

Similarity (Stage 8):
    sim(z, p) = 1 / (||z - p||² / C + 1)   ∈ (0, 1]

    where C = feature_dim normalises the distance to be scale-invariant across levels.
    sim = 1.0 when z = p (perfect match).
    sim → 0 as ||z - p||² → ∞.
    Sharp spatial falloff: background features far from the prototype → sim ≈ 0.
    Contrast with log-cosine which saturated near log(2) everywhere (poor localisation).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Prototype counts per level (M_l) ─────────────────────────────────────────
PROTOS_PER_LEVEL: dict[int, int] = {1: 4, 2: 3, 3: 2, 4: 2}
# Total prototypes per class = 4+3+2+2 = 11  →  88 vectors (K=8 classes)


# ─────────────────────────────────────────────────────────────────────────────
# PrototypeLayer
# ─────────────────────────────────────────────────────────────────────────────


class PrototypeLayer(nn.Module):
    """
    Learnable prototype layer for one encoder level l.

    Stores K × M prototype vectors  p_{k,m} ∈ ℝ^C.
    Computes L2-distance similarity heatmaps between Z_l and every prototype.

    Similarity:  S = 1 / ( ||z - p||² / C + 1 )   →  range (0, 1]
        S = 1.0 when z = p (exact match).
        S → 0 as z diverges from p (sharp spatial falloff).
        Division by C makes the range level-invariant across encoder depths.

    Args
    ----
    n_classes   K : number of output classes (default 8)
    n_protos    M : prototypes per class at this level
    feature_dim C : channel depth of the input feature map Z_l

    Input  : Z_l  (B, C, H, W)
    Output : A    (B, K, M, H, W)   L2-similarity heatmap  ∈ (0, 1]
    """

    def __init__(self, n_classes: int, n_protos: int, feature_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_protos = n_protos
        self.feature_dim = feature_dim
        # Learnable prototype bank  (K, M, C)
        self.prototypes = nn.Parameter(torch.randn(n_classes, n_protos, feature_dim))

    # ------------------------------------------------------------------
    def forward(self, Z_l: torch.Tensor) -> torch.Tensor:
        B, C, H, W = Z_l.shape
        K, M = self.n_classes, self.n_protos
        assert C == self.feature_dim, (
            f"feature_dim mismatch: expected {self.feature_dim}, got {C}"
        )

        # Flatten spatial  →  (B, H*W, C)
        z_flat = Z_l.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)

        # Prototype bank  →  (K*M, C)
        p_flat = self.prototypes.reshape(K * M, C)  # (KM, C)

        # Squared L2 distance  ||z - p||²  for every (z, p) pair  →  (B, HW, KM)
        # Expanded via: ||z-p||² = ||z||² + ||p||² - 2·z·p
        z_sq = (z_flat ** 2).sum(dim=-1, keepdim=True)          # (B, HW, 1)
        p_sq = (p_flat ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, KM)
        zp   = torch.einsum("bnc,kc->bnk", z_flat, p_flat)      # (B, HW, KM)
        dist_sq = (z_sq + p_sq - 2.0 * zp).clamp(min=0.0)      # (B, HW, KM)

        # L2 similarity: 1 / (normalised_dist² + 1)  →  (0, 1]
        sim = 1.0 / (dist_sq / C + 1.0)  # (B, HW, KM)

        # Reshape to  (B, K, M, H, W)
        sim = sim.reshape(B, H, W, K, M)
        sim = sim.permute(0, 3, 4, 1, 2).contiguous()  # (B, K, M, H, W)
        return sim


# ─────────────────────────────────────────────────────────────────────────────
# SoftMaskModule
# ─────────────────────────────────────────────────────────────────────────────


class SoftMaskModule(nn.Module):
    """
    Aggregates prototype heatmaps into a spatial attention mask and applies it
    to the feature map Z_l.

    Steps
    -----
    1. Max over prototypes M      →  per-class map  (B, K, H, W)
    2. Sum over classes K         →  aggregate mask  (B, 1, H, W)
    3. Element-wise multiply Z_l  →  masked features (B, C, H, W)

    Input  : A    (B, K, M, H, W)   heatmap from PrototypeLayer
             Z_l  (B, C, H, W)      raw feature map from encoder
    Output : masked_Z  (B, C, H, W) — same spatial shape as Z_l
    """

    def forward(self, A: torch.Tensor, Z_l: torch.Tensor) -> torch.Tensor:
        # max over M  →  (B, K, H, W)
        A_max = A.max(dim=2).values
        # sum over K  →  (B, 1, H, W)
        mask = A_max.sum(dim=1, keepdim=True)
        # broadcast multiply
        return Z_l * mask  # (B, C, H, W)


class HardMaskModule(nn.Module):
    """
    Hard spatial gate with Straight-Through Estimator (STE).

    Replaces SoftMaskModule to break the information bypass path identified in
    Stage 8: the soft-mask allowed the decoder to ignore prototype heatmaps via
    skip connections. The hard binary gate forces all spatial locations below the
    quantile threshold to exactly zero, blocking the bypass.

    Forward pass  : binary mask  (0 or 1)  — decoder sees only prototype-activated regions
    Backward pass : gradient flows through the continuous heatmap W (STE approximation)

    Steps
    -----
    1. Max over prototypes M → per-class map   (B, K, H, W)
    2. Max over classes K   → aggregate weight (B, 1, H, W)   [W ∈ (0, 1]]
    3. Threshold at sample-wise quantile q     → binary mask   (B, 1, H, W)
    4. STE: value = hard mask; gradient = W
    5. Multiply Z_l by mask_ste

    Args
    ----
    quantile : float  spatial percentile threshold in [0, 1).
               0.5 zeros the bottom 50% of spatial locations per sample.
               Lower → less aggressive (closer to soft-mask behaviour).
               Higher → more aggressive (sparser decoder input).

    Input  : A    (B, K, M, H, W)   heatmap from PrototypeLayer
             Z_l  (B, C, H, W)      raw feature map from encoder
    Output : masked_Z  (B, C, H, W) — same spatial shape as Z_l
    """

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        assert 0.0 <= quantile < 1.0, f"quantile must be in [0, 1), got {quantile}"
        self.quantile = quantile

    def forward(self, A: torch.Tensor, Z_l: torch.Tensor) -> torch.Tensor:
        # max over M  →  (B, K, H, W)
        A_max = A.max(dim=2).values
        # max over K  →  (B, 1, H, W)  peak prototype activation per spatial location
        W = A_max.max(dim=1, keepdim=True).values  # ∈ (0, 1]

        # Per-sample threshold: shape (B, 1, 1, 1)
        with torch.no_grad():
            tau = W.flatten(1).quantile(self.quantile, dim=1).view(-1, 1, 1, 1)

        # Hard binary mask (non-differentiable, forward value)
        mask_hard = (W >= tau).float()

        # STE: forward uses mask_hard; backward gradient flows through W
        # mask_ste == mask_hard numerically; dL/dW flows in backward
        mask_ste = mask_hard.detach() + W - W.detach()

        return Z_l * mask_ste  # (B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# PrototypeProjection
# ─────────────────────────────────────────────────────────────────────────────


class PrototypeProjection:
    """
    Pushes each prototype p_{l,k,m} to the nearest real training feature vector.

    For each (level, class k, prototype m):
      - Iterate over training DataLoader on CPU (MPS-safe)
      - Collect all feature vectors where GT label == k at that level's resolution
      - Find the training vector with highest cosine similarity to p_{l,k,m}
      - Replace the prototype in-place with that real vector
      - Record feat_idx for later visualization

    Saves: ``checkpoints/projected_prototypes.pt``
    Runtime target: < 2 min on CPU (2D feature bank is small).

    Nearest-neighbour search uses L2 distance (consistent with the L2 similarity
    used in PrototypeLayer.forward).

    Args
    ----
    encoder       : HierarchicalEncoder2D
    proto_layers  : dict { level_int → PrototypeLayer }
    device        : 'cpu' (recommended — avoids MPS re-allocations during projection)
    """

    def __init__(
        self,
        encoder: nn.Module,
        proto_layers: dict,
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.proto_layers = {k: v.to(device) for k, v in proto_layers.items()}
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def project(
        self,
        dataloader,
        save_path: str = "checkpoints/projected_prototypes.pt",
    ) -> dict:
        """
        Run one projection pass over the entire dataloader.

        The dataloader must yield ``(image, label)`` batches where
          image : (B, 1, H, W)  float32
          label : (B, H, W)     long  (class indices 0-K)

        Returns
        -------
        metadata : dict  {(level, k, m): {'feat_idx': int}}
        """
        self.encoder.eval()
        for pl in self.proto_layers.values():
            pl.eval()

        device = self.device
        encoder = self.encoder

        # ── Pass 1: collect feature bank and downsampled labels per level ──
        feat_bank: dict[int, list[torch.Tensor]] = {l: [] for l in self.proto_layers}
        lbl_bank: dict[int, list[torch.Tensor]] = {l: [] for l in self.proto_layers}

        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)  # (B, H, W)

            feats = encoder(images)  # {level: (B,C,Hl,Wl)}

            for level, Z_l in feats.items():
                if level not in self.proto_layers:
                    continue
                B, C, Hl, Wl = Z_l.shape

                # Downsample GT labels to this level's spatial resolution
                lbl_down = (
                    F.interpolate(
                        labels.float().unsqueeze(1),
                        size=(Hl, Wl),
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                )  # (B, Hl, Wl)

                # Flatten spatial  →  (B*Hl*Wl, C) and (B*Hl*Wl,)
                z_flat = Z_l.permute(0, 2, 3, 1).reshape(-1, C)
                l_flat = lbl_down.reshape(-1)

                feat_bank[level].append(z_flat.cpu())
                lbl_bank[level].append(l_flat.cpu())

        # Concatenate per level
        for level in self.proto_layers:
            feat_bank[level] = torch.cat(feat_bank[level], dim=0)  # (N, C)
            lbl_bank[level] = torch.cat(lbl_bank[level], dim=0)  # (N,)

        # ── Pass 2: per-prototype nearest-neighbour search (L2 distance) ──
        metadata: dict = {}

        for level, pl in self.proto_layers.items():
            K, M, C = pl.n_classes, pl.n_protos, pl.feature_dim

            bank_raw = feat_bank[level].to(device)  # (N, C)
            lbls = lbl_bank[level].to(device)        # (N,)

            for k in range(K):
                class_mask = lbls == k  # (N,) bool
                n_k = class_mask.sum().item()
                if n_k == 0:
                    # No training positions for this class — leave prototype unchanged
                    continue

                bank_k = bank_raw[class_mask]  # (N_k, C)
                orig_indices = class_mask.nonzero(as_tuple=True)[0]  # (N_k,)

                for m in range(M):
                    p_vec = pl.prototypes[k, m].unsqueeze(0)  # (1, C)

                    # Squared L2 distance with class-filtered bank  →  (N_k,)
                    dists_sq = ((bank_k - p_vec) ** 2).sum(dim=-1)
                    best_local = dists_sq.argmin().item()
                    best_global = orig_indices[best_local].item()

                    # Replace prototype with the nearest real feature vector
                    pl.prototypes.data[k, m] = bank_raw[best_global]
                    metadata[(level, k, m)] = {"feat_idx": int(best_global)}

        # ── Save ──────────────────────────────────────────────────────────
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "proto_state": {
                    level: pl.prototypes.data.clone()
                    for level, pl in self.proto_layers.items()
                },
                "metadata": metadata,
            },
            save_path,
        )
        print(f"Projected prototypes saved → {save_path}")
        return metadata
