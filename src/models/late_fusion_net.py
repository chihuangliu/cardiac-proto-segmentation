"""
src/models/late_fusion_net.py
Stage 9LF — Late Fusion ProtoSegNet

Each level is independently trained (9a, 9L2, 9L3, 9L1).
All encoders + prototype layers are frozen.
Only the attention MLP is trained — on a fair playing field.

Architecture:
    for each level l:
        feat_l    = model_l.encoder(x)[l]          frozen
        A_l       = model_l.proto_layers[l](feat_l) frozen
        up_l      = upsample(A_l.max(M), 256)
        pooled_l  = GAP(feat_l)

    context = cat(pooled_l for all l)
    w       = softmax(MLP(context))                trainable
    logits  = Σ_l  w_l * up_l
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.proto_seg_net_v2 import ProtoSegNetV2

CHANNELS: dict[int, int] = {1: 32, 2: 64, 3: 128, 4: 256}


class LateFusionProtoNet(nn.Module):
    """
    Late Fusion: independently trained per-level ProtoSegNetV2 models,
    all frozen, combined by a learned attention MLP.

    Args
    ----
    level_models : {level_int: ProtoSegNetV2}  — each is a single-level model
    """

    def __init__(self, level_models: dict):
        super().__init__()
        self.proto_levels = sorted(level_models.keys())
        # Register sub-models so state_dict / .to(device) work
        self.level_models = nn.ModuleDict(
            {str(l): m for l, m in level_models.items()}
        )

        # Freeze all sub-models
        for m in self.level_models.values():
            for p in m.parameters():
                p.requires_grad_(False)

        # Attention MLP: context = concatenated GAP features from each level
        context_dim = sum(CHANNELS[l] for l in self.proto_levels)
        self.attn_mlp = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(self.proto_levels)),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, T: float = 1.0
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Args
        ----
        x : (B, 1, 256, 256)
        T : softmax temperature (default 1.0, no annealing needed)

        Returns
        -------
        logits   : (B, K, 256, 256)
        heatmaps : {level_int: (B, K, M_l, H_l, W_l)}
        w        : (B, n_levels)
        """
        pooled_list = []
        upsampled   = {}
        heatmaps    = {}

        for l in self.proto_levels:
            model  = self.level_models[str(l)]
            feats  = model.encoder(x)                            # frozen
            feat_l = feats[l]
            A      = model.proto_layers[str(l)](feat_l)          # (B,K,M,H,W) frozen
            heatmaps[l] = A

            A_agg = A.max(dim=2).values                          # (B, K, H, W)
            upsampled[l] = F.interpolate(
                A_agg, size=(256, 256), mode="bilinear", align_corners=False
            )
            pooled_list.append(
                F.adaptive_avg_pool2d(feat_l, 1).flatten(1)      # (B, C_l)
            )

        context = torch.cat(pooled_list, dim=1)                  # (B, context_dim)
        w = F.softmax(self.attn_mlp(context) / T, dim=-1)        # (B, n_levels)

        logits = sum(
            w[:, j].view(-1, 1, 1, 1) * upsampled[l]
            for j, l in enumerate(self.proto_levels)
        )
        return logits, heatmaps, w

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        frozen    = sum(p.numel() for m in self.level_models.values()
                        for p in m.parameters())
        trainable = sum(p.numel() for p in self.attn_mlp.parameters())
        return {"frozen": frozen, "trainable_attn": trainable, "total": frozen + trainable}
