"""
scripts/train_baseline.py
Train a baseline 2D U-Net on MM-WHS CT or MRI data.

Usage:
    .venv/bin/python scripts/train_baseline.py --modality ct
    .venv/bin/python scripts/train_baseline.py --modality mr
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from src.data.mmwhs_dataset import (
    MMWHSSliceDataset,
    MMWHSPatientDataset,
    make_dataloaders,
    LABEL_NAMES,
)
from src.losses.segmentation import SegmentationLoss, compute_class_weights
from src.metrics.dice import dice_per_class, mean_foreground_dice
from src.models.unet import UNet2D

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = "data/pack/processed_data"
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

N_CLASSES = 8
BASE_CH = 32
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
VAL_EVERY = 5
PATIENCE = 20  # early stopping


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_slices(model, loader, device) -> dict:
    """Quick slice-level Dice on the val loader (fast, runs every VAL_EVERY epochs)."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        lbls = batch["label"]
        logits = model(imgs).cpu()
        all_logits.append(logits)
        all_labels.append(lbls)
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    return dice_per_class(logits_cat, labels_cat)


@torch.no_grad()
def validate_patients(model, modality: str, split: str, device) -> dict:
    """Patient-level volumetric Dice (slice stack). Slower but clinically meaningful."""
    model.eval()
    patient_ds = MMWHSPatientDataset(DATA_DIR, modality, split)
    patient_results = {}
    for i in range(len(patient_ds)):
        sample = patient_ds[i]
        imgs = sample["image"].to(device)   # (S, 1, H, W)
        lbls = sample["label"]              # (S, H, W)
        # Infer slice by slice to keep memory low
        logits_list = []
        for s in range(imgs.shape[0]):
            logits_list.append(model(imgs[s:s+1]).cpu())
        logits_vol = torch.cat(logits_list, dim=0)  # (S, C, H, W)
        dice = dice_per_class(logits_vol, lbls)
        patient_results[sample["patient"]] = dice
    return patient_results


# ── Training loop ─────────────────────────────────────────────────────────────

def train(modality: str):
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Training Baseline 2D U-Net — {modality.upper()}")
    print(f"  Device: {device}  |  Batch: {BATCH_SIZE}  |  Epochs: {MAX_EPOCHS}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = make_dataloaders(DATA_DIR, modality, batch_size=BATCH_SIZE)
    print(f"  Train: {len(loaders['train'].dataset)} slices  "
          f"Val: {len(loaders['val'].dataset)} slices  "
          f"Test: {len(loaders['test'].dataset)} slices")

    # ── Class weights ─────────────────────────────────────────────────────────
    weights_path = Path(f"data/class_weights_{modality}.pt")
    if weights_path.exists():
        class_weights = torch.load(weights_path, weights_only=True)
        print(f"  Loaded class weights from {weights_path}")
    else:
        print("  Computing class weights from training set…")
        class_weights = compute_class_weights(DATA_DIR, modality)
        torch.save(class_weights, weights_path)
        print(f"  Saved class weights to {weights_path}")
    print(f"  Class weights: { {LABEL_NAMES[i]: f'{class_weights[i]:.3f}' for i in range(N_CLASSES)} }")

    # ── Model & loss ──────────────────────────────────────────────────────────
    model = UNet2D(in_channels=1, n_classes=N_CLASSES, base_ch=BASE_CH).to(device)
    print(f"  Model params: {model.count_parameters():,}")

    criterion = SegmentationLoss(class_weights=class_weights.to(device), n_classes=N_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = RESULTS_DIR / f"train_log_baseline_{modality}.csv"
    csv_file = open(log_path, "w", newline="")
    fieldnames = ["epoch", "train_loss", "train_dice_loss", "train_ce_loss",
                  "val_mean_fg_dice", "lr", "epoch_time_s"] + \
                 [f"val_dice_{LABEL_NAMES[c]}" for c in range(1, N_CLASSES)]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_val_dice = 0.0
    best_epoch = 0
    no_improve = 0
    ckpt_path = CHECKPOINT_DIR / f"baseline_unet_{modality}.pth"

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = total_dice_loss = total_ce_loss = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            losses = criterion(logits, lbls)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses["loss"].item()
            total_dice_loss += losses["dice_loss"].item()
            total_ce_loss += losses["ce_loss"].item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = total_loss / n_batches
        avg_dice_loss = total_dice_loss / n_batches
        avg_ce_loss = total_ce_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        # ── Validation ────────────────────────────────────────────────────────
        row = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "train_dice_loss": round(avg_dice_loss, 5),
            "train_ce_loss": round(avg_ce_loss, 5),
            "lr": round(current_lr, 7),
            "epoch_time_s": round(epoch_time, 1),
        }

        if epoch % VAL_EVERY == 0 or epoch == 1:
            val_dice = validate_slices(model, loaders["val"], device)
            val_mean = mean_foreground_dice(val_dice)
            row["val_mean_fg_dice"] = round(val_mean, 5)
            for c in range(1, N_CLASSES):
                name = LABEL_NAMES[c]
                v = val_dice.get(name, float("nan"))
                row[f"val_dice_{name}"] = round(v, 4) if v == v else "nan"

            # Checkpoint on improvement
            if val_mean > best_val_dice:
                best_val_dice = val_mean
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "class_weights": class_weights,
                }, ckpt_path)
                flag = " ← best"
            else:
                no_improve += 1
                flag = ""

            print(f"  Ep {epoch:3d}/{MAX_EPOCHS} | "
                  f"loss={avg_loss:.4f} (D={avg_dice_loss:.4f} CE={avg_ce_loss:.4f}) | "
                  f"val_Dice={val_mean:.4f}{flag} | "
                  f"lr={current_lr:.2e} | {epoch_time:.1f}s")
        else:
            row["val_mean_fg_dice"] = ""
            for c in range(1, N_CLASSES):
                row[f"val_dice_{LABEL_NAMES[c]}"] = ""
            if epoch % 10 == 0:
                print(f"  Ep {epoch:3d}/{MAX_EPOCHS} | "
                      f"loss={avg_loss:.4f} | lr={current_lr:.2e} | {epoch_time:.1f}s")

        writer.writerow(row)
        csv_file.flush()

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} val checks)")
            break

    csv_file.close()
    print(f"\n  Best val mean fg Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint saved: {ckpt_path}")
    print(f"  Log saved: {log_path}")

    # ── Final patient-level evaluation ────────────────────────────────────────
    print(f"\n  Loading best checkpoint for patient-level evaluation…")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    for split in ("val", "test"):
        print(f"\n  [{split.upper()}] Patient-level Dice:")
        patient_results = validate_patients(model, modality, split, device)
        for patient, dice in patient_results.items():
            mean_fg = mean_foreground_dice(dice)
            per_class = "  ".join(
                f"{k[:3]}={v:.3f}" for k, v in dice.items() if k != "Background" and v == v
            )
            print(f"    {patient}: mean_fg={mean_fg:.4f}  [{per_class}]")

    print(f"\n  Stage 1 complete for {modality.upper()}.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    train(args.modality)
