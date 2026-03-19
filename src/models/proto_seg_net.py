"""
src/models/proto_seg_net.py
Stage 5  — Full Prototype Segmentation Network
Stage 9  — Hard Mask (HardMaskModule + STE) replaces SoftMaskModule
Stage 19 — LevelAttentionModule: learned weighted-sum aggregation over levels

ProtoSegNet: end-to-end pipeline
    Input X  (B, 1, 256, 256)
    → HierarchicalEncoder2D  → {Z_l}  per-level feature maps
    → PrototypeLayer (per level)      → {A_{l,k,m}} heatmaps (B, K, M, H_l, W_l)
    → SoftMaskModule | HardMaskModule → masked feature maps  (B, C_l, H_l, W_l)
    → 2D Decoder (bilinear upsample + skip connections from masked features)
    → 1×1 Conv                        → logits (B, K, 256, 256)

forward() returns (logits, heatmaps_dict):
    logits        : (B, K, H, W)  raw segmentation logits (no softmax)
    heatmaps_dict : {level: (B, K, M, H_l, W_l)}  for XAI metrics

use_level_attention=True replaces the cross-level max aggregation (used to build the
soft mask) with a learned weighted sum.  The mask module then receives a blended
feature map instead of the max-activated one.  Backward compatible: existing
checkpoints load unchanged with use_level_attention=False (default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import HierarchicalEncoder2D
from src.models.prototype_layer import (
    PrototypeLayer, SoftMaskModule, HardMaskModule, PROTOS_PER_LEVEL
)

N_CLASSES = 8


# ── Level Attention Module ─────────────────────────────────────────────────────

class LevelAttentionModule(nn.Module):
    """
    Learns soft weights over active prototype levels conditioned on encoder context.

    For each input image, produces a (B, n_levels) softmax weight vector that is
    used to blend per-level prototype heatmaps before the soft-mask step:

        heatmap_blended[k] = Σ_l  w[:,l] * upsample(max_m A[l,k,m])

    This replaces the cross-level max aggregation with a learned, input-conditioned
    weighted sum, allowing the model to discover that deep levels (L3, L4) are more
    semantically informative than shallow ones (L1, L2).

    Architecture:
        Global avg pool each active level → concat → Linear → ReLU → Linear → softmax

    Args:
        active_levels : sorted list of level ints, e.g. [1, 2, 3, 4] or [3, 4]
        channels      : dict {level_int: n_channels}, e.g. {1:32, 2:64, 3:128, 4:256}
        hidden_dim    : MLP hidden size (default 64)
    """

    def __init__(self, active_levels: list[int], channels: dict[int, int],
                 hidden_dim: int = 64):
        super().__init__()
        self.active_levels = active_levels
        in_dim = sum(channels[l] for l in active_levels)
        n_levels = len(active_levels)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_levels),
        )

    def forward(self, feat: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feat : {level: (B, C_l, H_l, W_l)}
        Returns:
            w    : (B, n_levels)  softmax attention weights
        """
        pooled = [feat[l].mean(dim=(2, 3)) for l in self.active_levels]  # [(B, C_l)]
        x = torch.cat(pooled, dim=1)                                      # (B, sum_C_l)
        return torch.softmax(self.mlp(x), dim=1)                          # (B, n_levels)


# ── Decoder building block ─────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    One decoder upsampling step:
        Upsample(×2, bilinear) → concat skip connection → Conv3×3 → BN → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── ProtoSegNet ────────────────────────────────────────────────────────────────

class ProtoSegNet(nn.Module):
    """
    Multi-scale prototype segmentation network.

    Encoder channels  : {1: 32, 2: 64, 3: 128, 4: 256}
    Prototype counts  : {l1: 4, l2: 3, l3: 2, l4: 2} per class

    Ablation flags:
        proto_levels       : list of levels to place prototypes on, e.g. [4] or [3, 4].
                             Levels not in the list pass raw encoder features to the
                             decoder unchanged. Default None = [1, 2, 3, 4] (all levels).
                             Supersedes single_scale when provided.
        single_scale       : legacy flag — equivalent to proto_levels=[4]. Ignored when
                             proto_levels is explicitly set.
        no_soft_mask       : bypass mask module entirely; raw encoder features fed to decoder
        hard_mask          : use HardMaskModule (STE binary gate) instead of SoftMaskModule
        mask_quantile      : spatial quantile threshold for HardMaskModule (default 0.5)
        use_level_attention: replace cross-level max aggregation with LevelAttentionModule
                             weighted sum. Default False (backward compatible).

    Decoder (deep → shallow):
        masked_Z4  (B, 256, 16,  16)  ─→  DecoderBlock(256+128, 128)  →  (B, 128, 32, 32)
        masked_Z3  (B, 128, 32,  32)  ─┘
        ↓
        DecoderBlock(128+64, 64)  →  (B, 64, 64, 64)
        masked_Z2  (B,  64, 64,  64)  ─┘
        ↓
        DecoderBlock(64+32, 32)   →  (B, 32, 128, 128)
        masked_Z1  (B,  32, 128, 128) ─┘
        ↓
        Upsample(×2) → Conv3×3 → BN → ReLU  →  (B, 32, 256, 256)
        ↓
        1×1 Conv  →  (B, K, 256, 256)  logits
    """

    def __init__(self, n_classes: int = N_CLASSES,
                 proto_levels: list[int] | None = None,
                 single_scale: bool = False,
                 no_soft_mask: bool = False,
                 hard_mask: bool = False,
                 mask_quantile: float = 0.5,
                 use_level_attention: bool = False):
        super().__init__()
        self.n_classes = n_classes
        self.no_soft_mask = no_soft_mask
        self.hard_mask = hard_mask
        self.mask_quantile = mask_quantile
        self.use_level_attention = use_level_attention

        # Resolve active levels: proto_levels takes priority over single_scale
        if proto_levels is not None:
            active_levels = sorted(proto_levels)
        elif single_scale:
            active_levels = [4]
        else:
            active_levels = [1, 2, 3, 4]
        self.proto_levels = active_levels
        # Keep single_scale for checkpoint backward compat
        self.single_scale = (active_levels == [4])

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = HierarchicalEncoder2D(in_channels=1)
        ch = HierarchicalEncoder2D.CHANNELS  # {1:32, 2:64, 3:128, 4:256}

        # ── Prototype layers ──────────────────────────────────────────────────
        self.proto_layers = nn.ModuleDict({
            str(l): PrototypeLayer(n_classes, PROTOS_PER_LEVEL[l], ch[l])
            for l in active_levels
        })

        # ── Mask module ───────────────────────────────────────────────────────
        # hard_mask=True stores both modules; hard gating is activated only when
        # hard_mask_active is set to True (done at Phase A→B transition in trainer).
        # During Phase A (random prototypes), hard_mask_active=False so the decoder
        # learns with soft-mask inputs, avoiding disruption from random binary gating.
        if hard_mask:
            self.mask_module = HardMaskModule(quantile=mask_quantile)
            self._soft_mask_fallback = SoftMaskModule()
        else:
            self.mask_module = SoftMaskModule()
            self._soft_mask_fallback = None
        self.hard_mask_active: bool = False   # set True by trainer at Phase B start

        # ── Level attention ───────────────────────────────────────────────────
        if use_level_attention and len(active_levels) > 1:
            self.level_attention = LevelAttentionModule(active_levels, ch)
        else:
            self.level_attention = None
        self._cached_attn_weights: torch.Tensor | None = None  # set during forward()
        self.pruned_levels: set[int] = set()   # levels detached from gradient (Stage 25)

        # ── Decoder ───────────────────────────────────────────────────────────
        # dec4: upsample Z4 (16→32)  +  skip Z3  →  128 ch
        self.dec4 = DecoderBlock(ch[4] + ch[3], 128)
        # dec3: upsample (32→64)    +  skip Z2  →  64 ch
        self.dec3 = DecoderBlock(128 + ch[2], 64)
        # dec2: upsample (64→128)   +  skip Z1  →  32 ch
        self.dec2 = DecoderBlock(64 + ch[1], 32)
        # dec1: upsample (128→256), no more skip
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Final 1×1 conv: 32 → K logits
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    # ── Helpers: expose proto_layers with int keys ─────────────────────────────

    def _proto_layer(self, level: int) -> PrototypeLayer:
        return self.proto_layers[str(level)]

    def proto_layers_dict(self) -> dict[int, PrototypeLayer]:
        """Return {level_int: PrototypeLayer} for use with PrototypeProjection."""
        return {int(k): v for k, v in self.proto_layers.items()}

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """
        Args:
            x : (B, 1, 256, 256)
        Returns:
            logits        : (B, K, 256, 256)   raw (pre-softmax) segmentation scores
            heatmaps_dict : {level: (B, K, M, H_l, W_l)}
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        feat = self.encoder(x)  # {1: Z1, 2: Z2, 3: Z3, 4: Z4}

        # Detach pruned levels so no gradient flows back through those encoder stages.
        # The decoder skip connection receives a zero tensor so the pruned architecture
        # is equivalent to M2 (levels truly absent), not just gradient-stopped.
        for l in self.pruned_levels:
            if l in feat:
                feat[l] = torch.zeros_like(feat[l])

        # ── Prototype heatmaps ────────────────────────────────────────────────
        heatmaps: dict[int, torch.Tensor] = {}
        for l in [1, 2, 3, 4]:
            if str(l) in self.proto_layers and l not in self.pruned_levels:
                heatmaps[l] = self._proto_layer(l)(feat[l])  # (B, K, M, H_l, W_l)

        # ── Level attention weights (optional) ───────────────────────────────
        # w: (B, n_active_levels) — only computed when level_attention is active
        if self.level_attention is not None:
            w = self.level_attention(feat)   # (B, n_levels), softmax
            self._cached_attn_weights = w    # cached for entropy regularisation in training loop
        else:
            w = None
            self._cached_attn_weights = None

        # ── Soft masks ────────────────────────────────────────────────────────
        masked: dict[int, torch.Tensor] = {}
        for l in [1, 2, 3, 4]:
            if str(l) in self.proto_layers:
                if l in self.pruned_levels:
                    # Pruned: feat[l] is already zeros (set above); pass zeros as skip
                    # so the decoder sees this level as fully absent (equivalent to M2).
                    masked[l] = feat[l]
                else:
                    A = heatmaps[l]                              # (B, K, M, H_l, W_l)
                    if w is not None:
                        # Weighted-sum over non-pruned levels only; renormalise so
                        # pruning a level does not shrink the blended signal magnitude.
                        A_blended = torch.zeros_like(A[:, :, 0, :, :])  # (B, K, H_l, W_l)
                        w_sum = torch.zeros(A.shape[0], 1, 1, 1, device=A.device)
                        for j, l2 in enumerate(self.proto_levels):
                            if l2 in self.pruned_levels:
                                continue
                            A_l2_max = heatmaps[l2].max(dim=2).values    # (B, K, H_l2, W_l2)
                            A_l2_up  = F.interpolate(
                                A_l2_max, size=A.shape[-2:],
                                mode="bilinear", align_corners=False
                            )                                             # (B, K, H_l, W_l)
                            A_blended = A_blended + w[:, j:j+1, None, None] * A_l2_up
                            w_sum     = w_sum + w[:, j:j+1, None, None]
                        # Renormalise; when no pruning w_sum ≈ 1 so behaviour unchanged.
                        A_for_mask = (A_blended / (w_sum + 1e-8)).unsqueeze(2)
                    else:
                        A_for_mask = A

                    if self.no_soft_mask:
                        masked[l] = feat[l]
                    elif self.hard_mask and not self.hard_mask_active:
                        masked[l] = self._soft_mask_fallback(A_for_mask, feat[l])
                    else:
                        masked[l] = self.mask_module(A_for_mask, feat[l])
            else:
                masked[l] = feat[l]

        # ── Decoder ──────────────────────────────────────────────────────────
        d = self.dec4(masked[4], masked[3])  # (B, 128, 32, 32)
        d = self.dec3(d, masked[2])          # (B,  64, 64, 64)
        d = self.dec2(d, masked[1])          # (B,  32, 128, 128)
        # Upsample to full resolution
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec1(d)                     # (B, 32, 256, 256)
        logits = self.final_conv(d)          # (B, K, 256, 256)

        return logits, heatmaps

    # ── Parameter groups for phase-based training ─────────────────────────────

    def freeze_prototypes(self):
        """Phase A: freeze prototype parameters and level attention (if present)."""
        for pl in self.proto_layers.values():
            for p in pl.parameters():
                p.requires_grad_(False)
        if self.level_attention is not None:
            for p in self.level_attention.parameters():
                p.requires_grad_(False)

    def unfreeze_prototypes(self):
        """Phase B: unfreeze prototype parameters and level attention (if present)."""
        for pl in self.proto_layers.values():
            for p in pl.parameters():
                p.requires_grad_(True)
        if self.level_attention is not None:
            for p in self.level_attention.parameters():
                p.requires_grad_(True)

    def freeze_encoder_and_prototypes(self):
        """Phase C: freeze encoder and prototypes; decoder + attention train."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for pl in self.proto_layers.values():
            for p in pl.parameters():
                p.requires_grad_(False)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor | None:
        """
        Return level attention weights for input x, or None if not using attention.

        Args:
            x : (B, 1, 256, 256)
        Returns:
            w : (B, n_active_levels) softmax weights, or None
        """
        if self.level_attention is None:
            return None
        with torch.no_grad():
            feat = self.encoder(x)
            return self.level_attention(feat)

    def unfreeze_all(self):
        """Unfreeze every parameter."""
        for p in self.parameters():
            p.requires_grad_(True)

    def prune_level(self, level: int) -> None:
        """
        Hard-soft prune a prototype level:
          - Add to pruned_levels so forward() zeroes its encoder features
            (decoder skip connection receives zeros → truly absent, like M2).
          - Freeze its PrototypeLayer parameters (no further gradient).
        """
        self.pruned_levels.add(level)
        if str(level) in self.proto_layers:
            for p in self.proto_layers[str(level)].parameters():
                p.requires_grad_(False)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
