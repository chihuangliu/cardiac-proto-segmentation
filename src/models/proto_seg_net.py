"""
src/models/proto_seg_net.py
Stage 5 — Full Prototype Segmentation Network
Stage 9 — Hard Mask (HardMaskModule + STE) replaces SoftMaskModule

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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import HierarchicalEncoder2D
from src.models.prototype_layer import (
    PrototypeLayer, SoftMaskModule, HardMaskModule, PROTOS_PER_LEVEL
)

N_CLASSES = 8


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
        single_scale : only level-4 prototypes; levels 1-3 skip prototype & mask
        no_soft_mask : bypass mask module entirely; raw encoder features fed to decoder
        hard_mask    : use HardMaskModule (STE binary gate) instead of SoftMaskModule
        mask_quantile: spatial quantile threshold for HardMaskModule (default 0.5)

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
                 single_scale: bool = False,
                 no_soft_mask: bool = False,
                 hard_mask: bool = False,
                 mask_quantile: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.single_scale = single_scale
        self.no_soft_mask = no_soft_mask
        self.hard_mask = hard_mask
        self.mask_quantile = mask_quantile

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = HierarchicalEncoder2D(in_channels=1)
        ch = HierarchicalEncoder2D.CHANNELS  # {1:32, 2:64, 3:128, 4:256}

        # ── Prototype layers ──────────────────────────────────────────────────
        # single_scale: only level 4 has prototypes
        active_levels = [4] if single_scale else [1, 2, 3, 4]
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

        # ── Prototype heatmaps + soft masks ──────────────────────────────────
        heatmaps: dict[int, torch.Tensor] = {}
        masked: dict[int, torch.Tensor] = {}
        for l in [1, 2, 3, 4]:
            if str(l) in self.proto_layers:
                A = self._proto_layer(l)(feat[l])          # (B, K, M, H_l, W_l)
                heatmaps[l] = A
                if self.no_soft_mask:
                    masked[l] = feat[l]
                elif self.hard_mask and not self.hard_mask_active:
                    # Phase A: prototypes random → use soft mask to avoid random binary gating
                    masked[l] = self._soft_mask_fallback(A, feat[l])
                else:
                    masked[l] = self.mask_module(A, feat[l])
            else:
                # single_scale: no prototype/mask for this level; pass raw features
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
        """Phase A: freeze all prototype parameters."""
        for pl in self.proto_layers.values():
            for p in pl.parameters():
                p.requires_grad_(False)

    def unfreeze_prototypes(self):
        """Phase B: unfreeze prototype parameters."""
        for pl in self.proto_layers.values():
            for p in pl.parameters():
                p.requires_grad_(True)

    def freeze_encoder_and_prototypes(self):
        """Phase C: freeze encoder and prototypes; only decoder trains."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for pl in self.proto_layers.values():
            for p in pl.parameters():
                p.requires_grad_(False)

    def unfreeze_all(self):
        """Unfreeze every parameter."""
        for p in self.parameters():
            p.requires_grad_(True)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
