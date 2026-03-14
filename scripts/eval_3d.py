#!/usr/bin/env python
"""
scripts/eval_3d.py
Stage 7 — 3D Volume Reconstruction Evaluation

Stacks per-patient 2D predictions along Z-axis, then computes:
  - 3D Dice per class (volumetric TP/FP/FN)
  - ASSD (Average Symmetric Surface Distance) using scipy distance transform
  - Saves per-patient 3D predictions as NIfTI (.nii.gz) for ITK-SNAP rendering

Usage:
    python scripts/eval_3d.py --modality ct --checkpoint checkpoints/proto_seg_ct.pth
    python scripts/eval_3d.py --modality mr --checkpoint checkpoints/proto_seg_mr.pth
    python scripts/eval_3d.py --modality ct --checkpoint ... --baseline  # use baseline U-Net
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.data.mmwhs_dataset import MMWHSPatientDataset, LABEL_NAMES, NUM_CLASSES
from src.models.proto_seg_net import ProtoSegNet
from src.models.unet import UNet2D as UNet

DATA_DIR   = ROOT / "data" / "pack" / "processed_data"
RESULT_DIR = ROOT / "results"

FG_NAMES = [LABEL_NAMES[k] for k in range(1, NUM_CLASSES)]


# ── Device ────────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_3d(pred_vol: np.ndarray, gt_vol: np.ndarray, n_classes: int) -> dict[str, float]:
    """
    Compute volumetric Dice per class.
    pred_vol, gt_vol: integer arrays of shape (S, H, W).
    """
    results = {}
    for k in range(1, n_classes):
        pred_k = pred_vol == k
        gt_k   = gt_vol   == k
        tp = (pred_k & gt_k).sum()
        fp = (pred_k & ~gt_k).sum()
        fn = (~pred_k & gt_k).sum()
        denom = 2 * tp + fp + fn
        results[LABEL_NAMES[k]] = float(2 * tp / denom) if denom > 0 else float("nan")
    return results


def assd_3d(pred_vol: np.ndarray, gt_vol: np.ndarray, n_classes: int,
            voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> dict[str, float]:
    """
    Compute ASSD (Average Symmetric Surface Distance) per class.
    voxel_spacing: (z, y, x) spacing in mm. Defaults to 1.0mm isotropic.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    results = {}
    for k in range(1, n_classes):
        pred_k = (pred_vol == k).astype(bool)
        gt_k   = (gt_vol   == k).astype(bool)

        # Skip classes absent from both volumes
        if not pred_k.any() and not gt_k.any():
            results[LABEL_NAMES[k]] = float("nan")
            continue

        # Surface extraction: erode by 1 voxel and XOR
        struct = np.ones((3, 3, 3), dtype=bool)
        pred_surface = pred_k ^ binary_erosion(pred_k, structure=struct)
        gt_surface   = gt_k   ^ binary_erosion(gt_k,   structure=struct)

        # Distance transforms (distance from each voxel to the opposite surface)
        if gt_surface.any():
            dt_gt  = distance_transform_edt(~gt_surface,   sampling=voxel_spacing)
        else:
            # Degenerate: entire volume is foreground for GT
            dt_gt = distance_transform_edt(~gt_k, sampling=voxel_spacing)

        if pred_surface.any():
            dt_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
        else:
            dt_pred = distance_transform_edt(~pred_k, sampling=voxel_spacing)

        # Mean surface distances
        d_pred_to_gt = dt_gt[pred_surface].mean()   if pred_surface.any() else 0.0
        d_gt_to_pred = dt_pred[gt_surface].mean()   if gt_surface.any()   else 0.0

        # ASSD = symmetric mean
        n_pred = pred_surface.sum()
        n_gt   = gt_surface.sum()
        if n_pred + n_gt > 0:
            assd = (d_pred_to_gt * n_pred + d_gt_to_pred * n_gt) / (n_pred + n_gt)
        else:
            assd = float("nan")

        results[LABEL_NAMES[k]] = float(assd)

    return results


# ── NIfTI export ──────────────────────────────────────────────────────────────

def save_nifti(volume: np.ndarray, out_path: Path) -> None:
    """Save integer volume (S, H, W) as NIfTI .nii.gz."""
    try:
        import nibabel as nib
        affine = np.eye(4)
        img = nib.Nifti1Image(volume.astype(np.uint8), affine)
        nib.save(img, str(out_path))
        print(f"  Saved NIfTI: {out_path.relative_to(ROOT)}")
    except ImportError:
        print("  [Warning] nibabel not installed — skipping NIfTI export")


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_patient(model, sample: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run slice-by-slice inference and return (pred_vol, gt_vol) as (S, H, W) int arrays.
    """
    imgs   = sample["image"]   # (S, 1, H, W)
    labels = sample["label"]   # (S, H, W)

    pred_slices = []
    for s in range(imgs.shape[0]):
        x = imgs[s : s + 1].to(device)           # (1, 1, H, W)
        if hasattr(model, "proto_layers"):
            logits, _ = model(x)
        else:
            logits = model(x)
        pred = logits.squeeze(0).argmax(0).cpu().numpy()  # (H, W)
        pred_slices.append(pred)

    pred_vol = np.stack(pred_slices, axis=0).astype(np.int32)
    gt_vol   = labels.numpy().astype(np.int32)
    return pred_vol, gt_vol


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="3D Dice + ASSD evaluation for ProtoSegNet")
    parser.add_argument("--modality",   required=True, choices=["ct", "mr"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--baseline",   action="store_true",
                        help="Load checkpoint as baseline U-Net (UNet) instead of ProtoSegNet")
    parser.add_argument("--no-nifti",   action="store_true",
                        help="Skip NIfTI export")
    parser.add_argument("--spacing",    nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        metavar=("Z", "Y", "X"),
                        help="Voxel spacing in mm for ASSD (default: 1.0 1.0 1.0)")
    args = parser.parse_args()

    device = pick_device()
    nifti_dir = RESULT_DIR / "nifti"
    nifti_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {device}")
    print(f"Modality  : {args.modality.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Spacing   : {args.spacing} mm")

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.baseline:
        model = UNet(n_classes=NUM_CLASSES).to(device)
    else:
        model = ProtoSegNet(
            n_classes=NUM_CLASSES,
            single_scale=ckpt.get("single_scale", False),
            no_soft_mask=ckpt.get("no_soft_mask", False),
            hard_mask=ckpt.get("hard_mask", False),
            mask_quantile=ckpt.get("mask_quantile", 0.5),
        ).to(device)
        model.hard_mask_active = ckpt.get("hard_mask_active", ckpt.get("hard_mask", False))

    # Handle multiple possible key names
    state = (ckpt.get("model_state_dict")
             or ckpt.get("model_state")
             or ckpt)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    model_type = "U-Net" if args.baseline else "ProtoSegNet"
    print(f"Model     : {model_type} ({n_params:,} params)\n")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    test_ds = MMWHSPatientDataset(DATA_DIR, modality=args.modality, split="test")

    all_dice: list[dict] = []
    all_assd: list[dict] = []

    for i in range(len(test_ds)):
        sample  = test_ds[i]
        patient = sample["patient"]
        S       = sample["image"].shape[0]
        print(f"[{i+1}/{len(test_ds)}] {patient}  ({S} slices)")

        pred_vol, gt_vol = infer_patient(model, sample, device)

        # 3D Dice
        dice = dice_3d(pred_vol, gt_vol, NUM_CLASSES)
        all_dice.append(dice)

        # ASSD
        assd = assd_3d(pred_vol, gt_vol, NUM_CLASSES, voxel_spacing=tuple(args.spacing))
        all_assd.append(assd)

        # Print per-patient table
        mean_dice = np.nanmean([v for v in dice.values() if v == v])
        mean_assd = np.nanmean([v for v in assd.values() if v == v])
        print(f"  3D Dice (mean fg) = {mean_dice:.4f}  |  ASSD (mean fg) = {mean_assd:.2f} mm")
        print(f"  {'Class':<14} {'3D Dice':>8}  {'ASSD (mm)':>10}")
        print(f"  {'-'*36}")
        for name in FG_NAMES:
            d = dice.get(name, float("nan"))
            a = assd.get(name, float("nan"))
            d_str = f"{d:.4f}" if d == d else "  nan "
            a_str = f"{a:8.2f}" if a == a else "     nan"
            print(f"  {name:<14} {d_str:>8}  {a_str:>10}")
        print()

        # NIfTI export
        if not args.no_nifti:
            save_nifti(pred_vol, nifti_dir / f"{patient}_pred.nii.gz")
            save_nifti(gt_vol,   nifti_dir / f"{patient}_gt.nii.gz")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f" Aggregate results — {args.modality.upper()} ({len(test_ds)} test patients)")
    print("=" * 55)
    print(f"  {'Class':<14} {'3D Dice':>8}  {'ASSD (mm)':>10}")
    print(f"  {'-'*36}")

    agg_dice_vals = []
    agg_assd_vals = []
    for name in FG_NAMES:
        ds = [d[name] for d in all_dice if d.get(name) == d.get(name)]
        as_ = [a[name] for a in all_assd if a.get(name) == a.get(name)]
        md = np.nanmean(ds) if ds else float("nan")
        ma = np.nanmean(as_) if as_ else float("nan")
        agg_dice_vals.append(md)
        agg_assd_vals.append(ma)
        d_str = f"{md:.4f}" if md == md else "  nan "
        a_str = f"{ma:8.2f}" if ma == ma else "     nan"
        print(f"  {name:<14} {d_str:>8}  {a_str:>10}")

    print(f"  {'-'*36}")
    print(f"  {'Mean (fg)':<14} {np.nanmean(agg_dice_vals):>8.4f}  "
          f"{np.nanmean(agg_assd_vals):>10.2f}")

    # ── Target check ──────────────────────────────────────────────────────────
    targets = {"ct": (0.72, 6.0), "mr": (0.67, 7.0)}
    dice_target, assd_target = targets[args.modality]
    mean_dice_all = np.nanmean(agg_dice_vals)
    mean_assd_all = np.nanmean(agg_assd_vals)

    print()
    print("  Stage 7 Success Criteria")
    print(f"  3D Dice ≥ {dice_target}: {mean_dice_all:.4f}  "
          f"{'✅' if mean_dice_all >= dice_target else '❌'}")
    print(f"  ASSD   ≤ {assd_target} mm: {mean_assd_all:.2f}  "
          f"{'✅' if mean_assd_all <= assd_target else '❌'}")


if __name__ == "__main__":
    main()
