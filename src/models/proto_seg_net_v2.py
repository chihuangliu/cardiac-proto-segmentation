"""
src/models/proto_seg_net_v2.py
Stage 9 — ProtoSegNetV2: Prototype-Only Prediction Path (no skip connections, no decoder)

Architecture: encoder → prototype heatmaps → upsample → weighted sum → logits

No decoder, no skip connections.  Structurally enforces Faithfulness:
    logits = f(heatmaps)  only — encoder features never bypass the prototype layer.

Stage 9a: ProtoSegNetV2(proto_levels=(4,),      use_attention=False)
Stage 9b: ProtoSegNetV2(proto_levels=(3,4),     use_attention=False)
Stage 9c: ProtoSegNetV2(proto_levels=(1,2,3,4), use_attention=True)

Unit-test bypass invariant:
    # Replace proto_layers with a zero-output module; logits must be zero.
    # (Zeroing prototype weights is NOT equivalent — L2 sim with p=0 is non-zero.)
    class _ZeroHeatmap(nn.Module):
        def forward(self, z):
            B = z.size(0)
            return torch.zeros(B, 8, 2, z.size(2), z.size(3), device=z.device)
    m = ProtoSegNetV2(proto_levels=(4,), use_attention=False)
    m.proto_layers['4'] = _ZeroHeatmap()
    assert m(x)[0].abs().max() < 1e-6    # no bypass path → logits = 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import HierarchicalEncoder2D
from src.models.prototype_layer import PrototypeLayer, PROTOS_PER_LEVEL

CHANNELS: dict[int, int] = {1: 32, 2: 64, 3: 128, 4: 256}


# ─────────────────────────────────────────────────────────────────────────────
# LevelAttentionModule
# ─────────────────────────────────────────────────────────────────────────────


class LevelAttentionModule(nn.Module):
    """
    Shared MLP that produces per-level attention weights from global-average-
    pooled encoder features.

    No entropy regularisation (proven harmful in v5).
    No feature detach (feedback is correct in no-skip architecture).
    Temperature T accepted at runtime for annealing (Stage 9c).

    Args
    ----
    proto_levels : tuple of int   active encoder levels, e.g. (1, 2, 3, 4)

    Input  : features  {level: (B, C_l, H_l, W_l)}
             T         softmax temperature (default 1.0)
    Output : w         (B, n_levels)  sums to 1.0 per sample
    """

    def __init__(self, proto_levels: tuple):
        super().__init__()
        self.proto_levels = tuple(proto_levels)
        context_dim = sum(CHANNELS[l] for l in self.proto_levels)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(self.proto_levels)),
        )

    def forward(self, features: dict, T: float = 1.0) -> torch.Tensor:
        pooled  = [F.adaptive_avg_pool2d(features[l], 1).flatten(1)
                   for l in self.proto_levels]
        context = torch.cat(pooled, dim=1)           # (B, context_dim)
        logits  = self.mlp(context)                  # (B, n_levels)
        return F.softmax(logits / T, dim=-1)         # (B, n_levels)


# ─────────────────────────────────────────────────────────────────────────────
# ProtoSegNetV2
# ─────────────────────────────────────────────────────────────────────────────


class ProtoSegNetV2(nn.Module):
    """
    Prototype-only segmentation network — no decoder, no skip connections.

    Prediction path (structurally enforced):
        features[l]   = encoder(x)[l]                     (B, C_l, H_l, W_l)
        A[l]          = proto_layers[l](features[l])       (B, K, M_l, H_l, W_l)
        A_agg[l]      = A[l].max(dim=2).values             (B, K, H_l, W_l)
        up[l]         = bilinear_upsample(A_agg[l], 256)   (B, K, 256, 256)
        logits        = Σ_l  w_l * up[l]                   (B, K, 256, 256)

    Because logits depend ONLY on heatmaps, Faithfulness is guaranteed by
    construction — no bypass path exists.

    Args
    ----
    n_classes     K  : number of output classes (default 8)
    proto_levels     : tuple of encoder levels to activate (e.g. (4,), (3,4))
    use_attention    : use LevelAttentionModule; if False, uniform weights
    """

    def __init__(
        self,
        n_classes: int = 8,
        proto_levels: tuple = (4,),
        use_attention: bool = False,
    ):
        super().__init__()
        self.n_classes    = n_classes
        self.proto_levels = tuple(proto_levels)
        self.use_attention = use_attention

        self.encoder = HierarchicalEncoder2D()
        self.proto_layers = nn.ModuleDict({
            str(l): PrototypeLayer(n_classes, PROTOS_PER_LEVEL[l], CHANNELS[l])
            for l in self.proto_levels
        })
        self.level_attention = (
            LevelAttentionModule(self.proto_levels) if use_attention else None
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, T: float = 1.0
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Args
        ----
        x : (B, 1, 256, 256)
        T : softmax temperature for LevelAttentionModule (default 1.0)

        Returns
        -------
        logits   : (B, K, 256, 256)
        heatmaps : {level_int: (B, K, M_l, H_l, W_l)}
        w        : (B, n_levels)  attention weights (uniform when use_attention=False)
        """
        features  = self.encoder(x)
        heatmaps  = {}
        upsampled = {}

        for l in self.proto_levels:
            A = self.proto_layers[str(l)](features[l])      # (B, K, M_l, H_l, W_l)
            heatmaps[l] = A
            A_agg = A.max(dim=2).values                     # (B, K, H_l, W_l)
            upsampled[l] = F.interpolate(
                A_agg, size=(256, 256), mode="bilinear", align_corners=False
            )                                               # (B, K, 256, 256)

        if self.level_attention is not None:
            w = self.level_attention(features, T)           # (B, n_levels)
        else:
            n = len(self.proto_levels)
            w = torch.full(
                (x.size(0), n), 1.0 / n, device=x.device, dtype=x.dtype
            )

        logits = sum(
            w[:, j].view(-1, 1, 1, 1) * upsampled[l]
            for j, l in enumerate(self.proto_levels)
        )
        return logits, heatmaps, w

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_prototypes(self) -> None:
        for p in self.proto_layers.parameters():
            p.requires_grad_(False)

    def unfreeze_prototypes(self) -> None:
        for p in self.proto_layers.parameters():
            p.requires_grad_(True)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def freeze_attention(self) -> None:
        if self.level_attention is not None:
            for p in self.level_attention.parameters():
                p.requires_grad_(False)

    def unfreeze_attention(self) -> None:
        if self.level_attention is not None:
            for p in self.level_attention.parameters():
                p.requires_grad_(True)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def freeze_encoder_and_prototypes(self) -> None:
        """Phase C (Stage 9c): freeze encoder + prototypes, leave attention trainable."""
        self.freeze_encoder()
        self.freeze_prototypes()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def proto_layers_dict(self) -> dict:
        """Return {level_int: PrototypeLayer} for use with PrototypeProjection."""
        return {int(k): v for k, v in self.proto_layers.items()}

    def count_parameters(self) -> dict:
        enc   = sum(p.numel() for p in self.encoder.parameters())
        proto = sum(p.numel() for p in self.proto_layers.parameters())
        attn  = (sum(p.numel() for p in self.level_attention.parameters())
                 if self.level_attention else 0)
        return {"encoder": enc, "proto": proto, "attention": attn,
                "total": enc + proto + attn}
