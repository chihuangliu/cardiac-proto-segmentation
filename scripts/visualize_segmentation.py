#!/usr/bin/env python
"""
scripts/visualize_segmentation.py
Stage 8 — Segmentation & Attention Visualization

For each test patient, selects representative slices (one per foreground class:
the slice where that class occupies the most GT pixels) and generates a 4-panel figure:

  ┌──────────────┬──────────────┐
  │  Input CT/MR │  GT Labels   │
  ├──────────────┼──────────────┤
  │  Prediction  │  Attention   │
  └──────────────┴──────────────┘

Attention panel: aggregate of all-level prototype similarity heatmaps for the target class,
summed over prototypes and levels, upsampled to 256×256, overlaid on the input image.

Output structure:
    results/segmentation_viz/{modality}/{patient_id}_slice{idx:04d}_{classname}.png

Usage:
    python scripts/visualize_segmentation.py --modality ct
    python scripts/visualize_segmentation.py --modality ct \\
        --checkpoint checkpoints/proto_seg_ct_l2.pth \\
        --output-dir results/segmentation_viz
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
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from src.data.mmwhs_dataset import MMWHSSliceDataset, LABEL_NAMES, NUM_CLASSES
from src.models.proto_seg_net import ProtoSegNet

DATA_DIR = ROOT / "data" / "pack" / "processed_data"
CKPT_DIR = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"

# Per-class RGB colours (0-1 float) — intentionally distinct
CLASS_COLORS_F = {
    0: (0.0,  0.0,  0.0),   # BG: black
    1: (0.90, 0.22, 0.27),  # LV: red
    2: (0.17, 0.61, 0.56),  # RV: teal
    3: (0.27, 0.48, 0.62),  # LA: steel blue
    4: (0.96, 0.63, 0.38),  # RA: orange
    5: (0.51, 0.22, 0.93),  # Myo: purple
    6: (0.02, 0.84, 0.63),  # Aorta: green
    7: (1.00, 0.72, 0.02),  # PA: gold
}


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def label_to_rgb(label_np):
    """Convert (H,W) int label map → (H,W,3) float RGB."""
    rgb = np.zeros((*label_np.shape, 3), dtype=np.float32)
    for k, c in CLASS_COLORS_F.items():
        mask = label_np == k
        rgb[mask] = c
    return rgb


def aggregate_attention(hm_dict, class_idx):
    """
    Aggregate prototype heatmaps across all levels for a single class.
    hm_dict : {level: (K, M, H_l, W_l)} CPU tensors (single sample, batch dim removed)
    Returns  : (256, 256) numpy float, max-normalised
    """
    agg = torch.zeros(1, 1, 256, 256)
    n = 0
    for level, hm in hm_dict.items():
        # hm: (K, M, H_l, W_l) — take max over M for the target class
        hm_k = hm[class_idx].max(dim=0).values   # (H_l, W_l)
        hm_up = F.interpolate(
            hm_k.unsqueeze(0).unsqueeze(0).float(),
            size=(256, 256), mode="bilinear", align_corners=False
        )
        agg += hm_up
        n += 1
    agg /= max(n, 1)
    arr = agg.squeeze().numpy()
    vmax = arr.max()
    if vmax > 0:
        arr = arr / vmax
    return arr


def blend_attention(img_norm, attn_np, alpha=0.45):
    """Overlay a hot-colourmap attention on grayscale image. Returns (H,W,3)."""
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    hot = plt.cm.hot(attn_np)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * hot, 0, 1)


@torch.no_grad()
def run_inference(model, img_tensor, device):
    """
    Single-image forward pass.
    Returns (pred_np, hm_dict) where
        pred_np  : (256, 256) int argmax prediction
        hm_dict  : {level: (K, M, H_l, W_l)} CPU tensors
    """
    model.eval()
    img = img_tensor.unsqueeze(0).to(device)
    logits, hm_dict = model(img)
    pred = logits[0].argmax(dim=0).cpu().numpy()
    hm_cpu = {l: hm[0].cpu() for l, hm in hm_dict.items()}
    return pred, hm_cpu


def make_4panel(img_norm, gt_np, pred_np, attn_np, title="", gt_class=None):
    """
    Build a 2×2 figure with:
      [input]  [GT]
      [pred]   [attention]
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.patch.set_facecolor("#111111")
    if title:
        fig.suptitle(title, fontsize=10, color="white", y=1.01)

    def imshow(ax, img, cmap=None):
        ax.imshow(img, cmap=cmap, interpolation="bilinear")
        ax.axis("off")

    # Panel 1: input
    imshow(axes[0, 0], img_norm, cmap="gray")
    axes[0, 0].set_title("Input", color="white", fontsize=9)

    # Panel 2: GT
    imshow(axes[0, 1], label_to_rgb(gt_np))
    axes[0, 1].set_title("Ground Truth", color="white", fontsize=9)

    # Panel 3: prediction
    imshow(axes[1, 0], label_to_rgb(pred_np))
    axes[1, 0].set_title("Prediction", color="white", fontsize=9)

    # Panel 4: attention for target class
    imshow(axes[1, 1], blend_attention(img_norm, attn_np))
    label_name = LABEL_NAMES.get(gt_class, "?") if gt_class is not None else "all"
    axes[1, 1].set_title(f"Attention ({label_name})", color="white", fontsize=9)

    # Class legend (bottom)
    patches = [
        mpatches.Patch(color=CLASS_COLORS_F[k], label=LABEL_NAMES[k])
        for k in range(1, NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=7,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, -0.04))

    for ax in axes.flat:
        ax.set_facecolor("#111111")

    plt.tight_layout(pad=0.5)
    return fig


def select_representative_slices(patient_slices):
    """
    For each foreground class k=1..7, find the slice index (within the patient's
    slice list) where GT label k has the most pixels.
    Returns list of (slice_local_idx, class_k) — deduplicated by local_idx,
    keeping the class with the highest count on that slice.
    """
    best: dict[int, tuple[int, int]] = {}   # class_k → (local_idx, count)
    for local_idx, sample in enumerate(patient_slices):
        gt = sample["label"].numpy()   # (256, 256)
        for k in range(1, NUM_CLASSES):
            count = int((gt == k).sum())
            if count > best.get(k, (-1, -1))[1]:
                best[k] = (local_idx, count)

    # Build output: unique slices, each tagged with the primary class
    seen: dict[int, int] = {}   # local_idx → best class (highest pixel count)
    for k, (local_idx, count) in best.items():
        if count == 0:
            continue
        if local_idx not in seen or count > best[seen.get(local_idx, 0)][1]:
            seen[local_idx] = k

    return sorted(seen.items())   # [(local_idx, class_k), ...]


def process_patient(patient_id, patient_slices, model, device, out_dir):
    """Generate visualizations for all representative slices of one patient."""
    patient_dir = out_dir
    patient_dir.mkdir(parents=True, exist_ok=True)

    reps = select_representative_slices(patient_slices)
    print(f"  {patient_id}: {len(reps)} representative slices")

    for local_idx, class_k in reps:
        sample = patient_slices[local_idx]
        img_t  = sample["image"]           # (1, 256, 256) float32
        gt_np  = sample["label"].numpy()   # (256, 256) int64
        fname  = sample["filename"]

        # Normalise image for display
        img_np = img_t[0].numpy()
        imin, imax = img_np.min(), img_np.max()
        img_norm = (img_np - imin) / (imax - imin + 1e-8)

        # Inference
        pred_np, hm_dict = run_inference(model, img_t, device)

        # Attention for the representative class
        if class_k in hm_dict or len(hm_dict) > 0:
            attn = aggregate_attention(hm_dict, class_k)
        else:
            attn = np.zeros((256, 256), dtype=np.float32)

        # Compute Dice for this slice / class (informational)
        gt_k   = (gt_np == class_k).sum()
        pred_k = (pred_np == class_k).sum()
        tp_k   = ((gt_np == class_k) & (pred_np == class_k)).sum()
        dice_k = 2 * tp_k / (gt_k + pred_k + 1e-8)

        title = (f"{patient_id}  |  slice {fname}  |  "
                 f"class={LABEL_NAMES[class_k]}  "
                 f"Dice={dice_k:.3f}")

        fig = make_4panel(img_norm, gt_np, pred_np, attn,
                          title=title, gt_class=class_k)

        class_name = LABEL_NAMES[class_k].lower().replace(" ", "_")
        out_path = patient_dir / f"{patient_id}_{fname.replace('.npz','')}_{class_name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved → {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Segmentation + attention visualization")
    parser.add_argument("--modality", required=True, choices=["ct", "mr"])
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Model checkpoint (default: checkpoints/proto_seg_{modality}_l2.pth)")
    parser.add_argument("--output-dir", type=str, default=str(RESULT_DIR / "segmentation_viz"))
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to visualize (default: test)")
    args = parser.parse_args()

    mod = args.modality
    ckpt_path = args.checkpoint or str(CKPT_DIR / f"proto_seg_{mod}_l2.pth")
    out_dir   = Path(args.output_dir) / mod
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device    : {device}")
    print(f"Modality  : {mod.upper()}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output    : {out_dir}")

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

    # ── Load test dataset ──────────────────────────────────────────────────────
    dataset = MMWHSSliceDataset(DATA_DIR, mod, args.split, augment=False, preload=True)
    print(f"Dataset: {len(dataset)} slices ({args.split} split)")

    # Group slices by patient
    from collections import defaultdict
    patient_slices: dict[str, list] = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        patient_slices[sample["patient"]].append(sample)

    patients = sorted(patient_slices.keys())
    print(f"Patients: {patients}\n")

    for patient_id in patients:
        process_patient(patient_id, patient_slices[patient_id],
                        model, device, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
