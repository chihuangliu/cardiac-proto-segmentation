#!/usr/bin/env python
"""
scripts/visualize_prototypes.py
Stage 8 — Prototype Atlas Visualization

For each encoder level, generates a grid showing what each prototype has "learned":
  rows    = 8 cardiac classes (BG, LV, RV, LA, RA, Myo, Aorta, PA)
  columns = M_l prototypes at that level

Each cell:
  - Left:  64×64 image crop centred on the prototype's projected source position
  - Right: L2 similarity heatmap for that prototype (jet, upsampled to 256×256, then cropped)

The prototype source position is recovered from:
    feat_idx  (stored in projected_prototypes_ct/mr.pt metadata)
    → slice_idx = feat_idx // (H_l × W_l)
    → (feat_h, feat_w) in the feature map grid
    → pixel (cy, cx) in the 256×256 original image via bilinear-equivalent mapping

Requires: a trained checkpoint + projected_prototypes_{modality}.pt

Usage:
    python scripts/visualize_prototypes.py --modality ct
    python scripts/visualize_prototypes.py --modality ct \\
        --checkpoint checkpoints/proto_seg_ct_l2.pth \\
        --proj-checkpoint checkpoints/projected_prototypes_ct.pt \\
        --output-dir results/prototype_atlas
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from src.data.mmwhs_dataset import MMWHSSliceDataset, LABEL_NAMES, NUM_CLASSES
from src.models.proto_seg_net import ProtoSegNet
from src.models.prototype_layer import PROTOS_PER_LEVEL
from src.models.encoder import HierarchicalEncoder2D

DATA_DIR = ROOT / "data" / "pack" / "processed_data"
CKPT_DIR = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"

# Encoder level → spatial size (from 256×256 input)
LEVEL_SPATIAL = {1: 128, 2: 64, 3: 32, 4: 16}

# Class color map for borders
CLASS_COLORS = {
    0: "#555555",  # BG
    1: "#e63946",  # LV
    2: "#2a9d8f",  # RV
    3: "#457b9d",  # LA
    4: "#f4a261",  # RA
    5: "#8338ec",  # Myo
    6: "#06d6a0",  # Aorta
    7: "#ffb703",  # PA
}

CROP_HALF = 48      # half-size of image crop (96×96 → displayed at cell size)
CELL_PX   = 128     # rendered cell size in pixels


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def get_prototype_heatmaps(model, img_tensor, device):
    """Run forward pass; return {level: (K, M, H_l, W_l)} heatmaps for single image."""
    model.eval()
    img = img_tensor.unsqueeze(0).to(device)   # (1,1,256,256)
    _, hm_dict = model(img)
    return {l: hm[0].cpu() for l, hm in hm_dict.items()}   # {l: (K,M,H,W)}


def feat_idx_to_pixel(feat_idx, level):
    """Convert flat feature bank index → pixel (cy, cx) in 256×256 image."""
    S = LEVEL_SPATIAL[level]
    spatial = feat_idx % (S * S)
    feat_h  = spatial // S
    feat_w  = spatial % S
    cy = int((feat_h + 0.5) * 256 / S)
    cx = int((feat_w + 0.5) * 256 / S)
    return cy, cx


def crop_patch(img_np, cy, cx, half=CROP_HALF):
    """Crop a square patch centred at (cy, cx); pad with zeros if near border."""
    H, W = img_np.shape
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    # Padding amounts
    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - W)
    y1c, y2c = max(0, y1), min(H, y2)
    x1c, x2c = max(0, x1), min(W, x2)
    crop = img_np[y1c:y2c, x1c:x2c]
    crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    return crop   # (2*half, 2*half)


def blend_heatmap(img_np, heatmap_np, alpha=0.45):
    """
    Overlay a jet heatmap on a grayscale image.
    img_np     : (H, W) float, normalised 0-1
    heatmap_np : (H, W) float, any range → normalised internally
    Returns (H, W, 3) float RGB in [0,1].
    """
    rgb = np.stack([img_np, img_np, img_np], axis=-1)
    hmin, hmax = heatmap_np.min(), heatmap_np.max()
    if hmax > hmin:
        hmap_norm = (heatmap_np - hmin) / (hmax - hmin)
    else:
        hmap_norm = np.zeros_like(heatmap_np)
    jet = plt.cm.jet(hmap_norm)[..., :3]   # (H, W, 3)
    blended = (1 - alpha) * rgb + alpha * jet
    return np.clip(blended, 0, 1)


def make_level_atlas(level, model, dataset, metadata, device, title_prefix=""):
    """
    Build a matplotlib figure for one encoder level.
    Rows = classes (BG … PA), Cols = prototypes M_l.
    Returns the figure.
    """
    S = LEVEL_SPATIAL[level]
    active_levels = list(model.proto_layers.keys())   # string keys

    if str(level) not in active_levels:
        return None   # single_scale model doesn't have this level

    pl = model.proto_layers[str(level)]
    K, M = pl.n_classes, pl.n_protos

    fig, axes = plt.subplots(K, M, figsize=(M * 2.2, K * 2.2), squeeze=False)
    fig.suptitle(f"{title_prefix}Level {level}  (spatial {S}×{S},  {M} proto/class)",
                 fontsize=12, fontweight="bold", y=1.01)

    # Cache: slice_idx → (img_np, heatmap_dict)
    slice_cache: dict[int, tuple] = {}

    for k in range(K):
        for m in range(M):
            ax = axes[k][m]
            ax.axis("off")

            # Header label on leftmost col
            if m == 0:
                ax.text(-0.08, 0.5, LABEL_NAMES[k], transform=ax.transAxes,
                        fontsize=8, va="center", ha="right",
                        color=CLASS_COLORS[k], fontweight="bold")

            key = (level, k, m)
            if key not in metadata:
                ax.set_facecolor("#1a1a1a")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6, color="gray")
                continue

            feat_idx = metadata[key]["feat_idx"]
            slice_idx = feat_idx // (S * S)
            cy, cx = feat_idx_to_pixel(feat_idx, level)

            # Load + cache
            if slice_idx not in slice_cache:
                sample = dataset[slice_idx]
                img_t = sample["image"]          # (1, 256, 256)
                img_np = img_t[0].numpy()        # (256, 256)
                # Normalize 0→1 for display
                imin, imax = img_np.min(), img_np.max()
                img_norm = (img_np - imin) / (imax - imin + 1e-8)
                hm_dict = get_prototype_heatmaps(model, img_t, device)
                slice_cache[slice_idx] = (img_norm, hm_dict)

            img_norm, hm_dict = slice_cache[slice_idx]

            # Prototype similarity heatmap → upsample to 256×256
            hm_km = hm_dict[level][k, m]         # (H_l, W_l)
            hm_up = F.interpolate(
                hm_km.unsqueeze(0).unsqueeze(0).float(),
                size=(256, 256), mode="bilinear", align_corners=False
            ).squeeze().numpy()

            # Crop patch (image) and matching heatmap crop
            img_crop = crop_patch(img_norm, cy, cx)
            hm_crop  = crop_patch(hm_up,   cy, cx)

            blended = blend_heatmap(img_crop, hm_crop)
            ax.imshow(blended, interpolation="bilinear")

            # Mark prototype centre with a white crosshair
            hw = img_crop.shape[0] // 2
            ax.axhline(hw, color="white", lw=0.6, alpha=0.7)
            ax.axvline(hw, color="white", lw=0.6, alpha=0.7)

            # Coloured border per class
            for spine in ax.spines.values():
                spine.set_edgecolor(CLASS_COLORS[k])
                spine.set_linewidth(1.5)
                spine.set_visible(True)

            ax.set_title(f"m={m}", fontsize=6, pad=2)

    # Column header: prototype index
    for m in range(M):
        axes[0][m].set_title(f"proto {m}", fontsize=7, pad=3, color="#cccccc")

    fig.patch.set_facecolor("#111111")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#111111")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualise prototype atlas")
    parser.add_argument("--modality", required=True, choices=["ct", "mr"])
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Model checkpoint (default: checkpoints/proto_seg_{modality}_l2.pth)")
    parser.add_argument("--proj-checkpoint", type=str, default="",
                        help="Projected prototypes (default: checkpoints/projected_prototypes_{modality}.pt)")
    parser.add_argument("--output-dir", type=str, default=str(RESULT_DIR / "prototype_atlas"))
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split used during projection (default: train)")
    args = parser.parse_args()

    mod = args.modality
    ckpt_path = args.checkpoint or str(CKPT_DIR / f"proto_seg_{mod}_l2.pth")
    proj_path = args.proj_checkpoint or str(CKPT_DIR / f"projected_prototypes_{mod}.pt")
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device   : {device}")
    print(f"Modality : {mod.upper()}")
    print(f"Checkpoint     : {ckpt_path}")
    print(f"Proj checkpoint: {proj_path}")
    print(f"Output dir     : {out_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ProtoSegNet(
        n_classes=NUM_CLASSES,
        single_scale=ckpt.get("single_scale", False),
        no_soft_mask=ckpt.get("no_soft_mask", False),
    ).to(device)
    state = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  single_scale={model.single_scale}  no_soft_mask={model.no_soft_mask}")

    # ── Load projection metadata ───────────────────────────────────────────────
    proj_ckpt = torch.load(proj_path, map_location="cpu", weights_only=False)
    metadata = proj_ckpt["metadata"]   # {(level, k, m): {'feat_idx': int}}
    print(f"Projection metadata: {len(metadata)} entries")

    # ── Load training dataset (same order as during projection) ───────────────
    dataset = MMWHSSliceDataset(DATA_DIR, mod, args.split, augment=False, preload=True)
    print(f"Dataset: {len(dataset)} slices ({args.split})")

    # ── Generate one atlas per level ──────────────────────────────────────────
    active_levels = [int(k) for k in model.proto_layers.keys()]
    suffix = f"_{mod}"
    if model.single_scale:
        suffix += "_single_scale"

    for level in sorted(active_levels):
        print(f"\nGenerating level {level} atlas…")
        fig = make_level_atlas(level, model, dataset, metadata, device,
                               title_prefix=f"{mod.upper()} — ")
        if fig is None:
            print(f"  Skipped (level {level} not in this model)")
            continue
        out_path = out_dir / f"{mod}_level{level}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
