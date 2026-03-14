#!/usr/bin/env python
"""
scripts/train_proto.py
Stage 7 — ProtoSegNet Training
Stage 8 — Push-Pull XAI Fix (--lambda-push / --lambda-pull)
Stage 9 — Hard Mask (--hard-mask / --mask-quantile)

3-phase training schedule:
  Phase A (ep 1–20):   backbone + decoder; prototypes frozen
  Phase B (ep 21–80):  all params; diversity + push-pull loss; projection every 10 ep
  Phase C (ep 81–100): decoder fine-tuning only

Usage:
    python scripts/train_proto.py --modality ct
    python scripts/train_proto.py --modality mr
    python scripts/train_proto.py --modality ct --lambda-push 0.1 --lambda-pull 0.05 --suffix _pp
    python scripts/train_proto.py --modality ct --hard-mask --mask-quantile 0.5 --suffix _hm
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.data.mmwhs_dataset import (
    MMWHSSliceDataset,
    make_dataloaders,
    LABEL_NAMES,
    NUM_CLASSES,
)
from src.models.proto_seg_net import ProtoSegNet
from src.models.prototype_layer import PrototypeProjection, PROTOS_PER_LEVEL
from src.losses.segmentation import SegmentationLoss, compute_class_weights
from src.losses.diversity_loss import ProtoSegLoss
from src.metrics.dice import dice_per_class, mean_foreground_dice

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = ROOT / "data" / "pack" / "processed_data"
CKPT_DIR    = ROOT / "checkpoints"
RESULT_DIR  = ROOT / "results"

BATCH_SIZE          = 16
LR                  = 3e-4
WEIGHT_DECAY        = 1e-5
LAMBDA_DIV          = 0.01  # override via --lambda-div
LAMBDA_PUSH         = 0.0   # override via --lambda-push
LAMBDA_PULL         = 0.0   # override via --lambda-pull
PHASE_A_END         = 20
PHASE_B_END         = 80
PHASE_C_END         = 100
MAX_EPOCHS          = PHASE_C_END  # override via --max-epochs
VAL_EVERY           = 5
PATIENCE            = 15
PROJECTION_INTERVAL = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def validate_slices(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits, _ = model(batch["image"].to(device))
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
    model.train()
    return dice_per_class(torch.cat(all_logits), torch.cat(all_labels))


def run_projection(model, modality, device, proj_path):
    print("  [Projection] Building feature bank on CPU…", flush=True)
    t0 = time.time()
    proj_ds = MMWHSSliceDataset(DATA_DIR, modality, "train", augment=False, preload=True)
    proj_loader = torch.utils.data.DataLoader(proj_ds, batch_size=32, shuffle=False)
    wrapped = [(b["image"], b["label"]) for b in proj_loader]

    projector = PrototypeProjection(
        encoder=model.encoder,
        proto_layers=model.proto_layers_dict(),
        device="cpu",
    )
    projector.project(wrapped, save_path=str(proj_path))

    # Restore model to training device — PrototypeProjection.to('cpu') mutates
    # the shared encoder/proto_layers in-place; move everything back.
    model.to(device)

    ckpt = torch.load(proj_path, weights_only=True)
    for level, proto_data in ckpt["proto_state"].items():
        model.proto_layers[str(level)].prototypes.data.copy_(proto_data)
    print(f"  [Projection] Done in {time.time()-t0:.1f}s", flush=True)


def set_phase(model, epoch, optimizer, phase_b_end=PHASE_B_END):
    if epoch <= PHASE_A_END:
        model.unfreeze_all()
        model.freeze_prototypes()
        phase = "A"
    elif epoch <= phase_b_end:
        model.unfreeze_all()
        phase = "B"
    else:
        model.freeze_encoder_and_prototypes()
        phase = "C"
    # Refresh optimizer param groups
    optimizer.param_groups[0]["params"] = [p for p in model.parameters() if p.requires_grad]
    return phase


# ── Main ──────────────────────────────────────────────────────────────────────

def train(modality: str, lambda_div: float = LAMBDA_DIV, lambda_push: float = LAMBDA_PUSH,
          lambda_pull: float = LAMBDA_PULL, suffix: str = "", start_epoch: int = 1,
          init_checkpoint: str = "", max_epochs: int = MAX_EPOCHS,
          single_scale: bool = False, no_soft_mask: bool = False,
          hard_mask: bool = False, mask_quantile: float = 0.5) -> None:
    device = pick_device()
    CKPT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    phase_b_end = min(PHASE_B_END, max_epochs)
    phase_c_end = max_epochs

    print(f"\n{'='*60}")
    print(f"  Training ProtoSegNet — {modality.upper()}  (suffix='{suffix}')")
    print(f"  Device: {device}  |  Batch: {BATCH_SIZE}  |  Max epochs: {max_epochs}")
    print(f"  λ_div={lambda_div}  λ_push={lambda_push}  λ_pull={lambda_pull}")
    print(f"  single_scale={single_scale}  no_soft_mask={no_soft_mask}")
    print(f"  hard_mask={hard_mask}  mask_quantile={mask_quantile}")
    print(f"  Projection every {PROJECTION_INTERVAL} epochs (Phase B)")
    print(f"{'='*60}\n")

    # Data
    loaders = make_dataloaders(DATA_DIR, modality, batch_size=BATCH_SIZE)
    print(f"  Train: {len(loaders['train'].dataset)} slices  "
          f"Val: {len(loaders['val'].dataset)} slices  "
          f"Test: {len(loaders['test'].dataset)} slices")

    # Class weights
    weights_path = ROOT / "data" / f"class_weights_{modality}.pt"
    if weights_path.exists():
        class_weights = torch.load(weights_path, weights_only=True)
        print(f"  Loaded class weights from {weights_path.name}")
    else:
        print("  Computing class weights…")
        class_weights = compute_class_weights(DATA_DIR, modality)
        torch.save(class_weights, weights_path)

    # Model + loss
    model = ProtoSegNet(n_classes=NUM_CLASSES,
                        single_scale=single_scale,
                        no_soft_mask=no_soft_mask,
                        hard_mask=hard_mask,
                        mask_quantile=mask_quantile).to(device)
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded weights from {init_checkpoint}  (was ep {ckpt['epoch']}, val {ckpt['best_val_dice']:.4f})")
    seg_loss = SegmentationLoss(class_weights=class_weights.to(device), n_classes=NUM_CLASSES)
    criterion = ProtoSegLoss(
        seg_loss=seg_loss,
        lambda_div=lambda_div,
        lambda_push=lambda_push,
        lambda_pull=lambda_pull,
    )
    total_params = model.count_parameters()["total"]
    print(f"  Model params: {total_params:,}")

    # Optimizer + scheduler
    model.freeze_prototypes()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Logging
    log_path  = RESULT_DIR / f"train_curve_proto_{modality}{suffix}.csv"
    ckpt_path = CKPT_DIR   / f"proto_seg_{modality}{suffix}.pth"
    proj_path = CKPT_DIR   / f"projected_prototypes_{modality}.pt"

    fieldnames = (
        ["epoch", "phase", "train_loss", "train_dice_loss", "train_ce_loss",
         "train_div_loss", "train_push_loss", "train_pull_loss",
         "val_mean_fg_dice", "lr", "epoch_time_s"]
        + [f"val_dice_{LABEL_NAMES[c]}" for c in range(1, NUM_CLASSES)]
    )
    # Append if resuming mid-run, otherwise start fresh
    csv_mode = "a" if start_epoch > 1 else "w"
    csv_file = open(log_path, csv_mode, newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if csv_mode == "w":
        writer.writeheader()

    best_val_dice, best_epoch, no_improve = 0.0, 0, 0
    current_phase = "A"
    if start_epoch == 1:
        print(f"  Phase A: backbone + decoder; prototypes frozen (epochs 1–{PHASE_A_END})\n")
    else:
        print(f"  Resuming from epoch {start_epoch}\n")

    for epoch in range(start_epoch, max_epochs + 1):

        # Phase transitions
        new_phase = set_phase(model, epoch, optimizer, phase_b_end)
        if new_phase != current_phase:
            current_phase = new_phase
            if current_phase == "B":
                print(f"\n  → Phase B: all params; diversity loss active "
                      f"(epochs {PHASE_A_END+1}–{phase_b_end})")
                if hard_mask:
                    model.hard_mask_active = True
                    print(f"  [Hard mask] Activated (quantile={mask_quantile})")
                run_projection(model, modality, device, proj_path)
                # Reset best so Phase B/C find their own best checkpoint
                # (Phase A val Dice is not comparable once diversity loss is active)
                best_val_dice, best_epoch, no_improve = 0.0, 0, 0
            elif current_phase == "C":
                print(f"\n  → Phase C: decoder fine-tuning only "
                      f"(epochs {phase_b_end+1}–{phase_c_end})")

        # Periodic projection in Phase B
        if (current_phase == "B"
                and epoch > PHASE_A_END + 1
                and (epoch - PHASE_A_END) % PROJECTION_INTERVAL == 0):
            run_projection(model, modality, device, proj_path)

        # Train epoch
        t0 = time.time()
        model.train()
        total_loss = total_dice = total_ce = total_div = total_push = total_pull = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            logits, hm = model(imgs)

            if current_phase == "A":
                out = seg_loss(logits, lbls)
                out["div_loss"]  = torch.zeros(1, device=device)
                out["push_loss"] = torch.zeros(1, device=device)
                out["pull_loss"] = torch.zeros(1, device=device)
            else:
                out = criterion(logits, lbls, hm)

            out["loss"].backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            total_loss += out["loss"].item()
            total_dice += out["dice_loss"].item()
            total_ce   += out["ce_loss"].item()
            total_div  += out["div_loss"].item()
            total_push += out["push_loss"].item()
            total_pull += out["pull_loss"].item()
            n_batches  += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg = lambda s: s / n_batches
        current_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch, "phase": current_phase,
            "train_loss":      round(avg(total_loss), 5),
            "train_dice_loss": round(avg(total_dice), 5),
            "train_ce_loss":   round(avg(total_ce),   5),
            "train_div_loss":  round(avg(total_div),  5),
            "train_push_loss": round(avg(total_push), 6),
            "train_pull_loss": round(avg(total_pull), 6),
            "lr": round(current_lr, 7),
            "epoch_time_s": round(epoch_time, 1),
        }

        # Validation
        if epoch % VAL_EVERY == 0 or epoch == 1:
            val_dice = validate_slices(model, loaders["val"], device)
            val_mean = mean_foreground_dice(val_dice)
            row["val_mean_fg_dice"] = round(val_mean, 5)
            for c in range(1, NUM_CLASSES):
                name = LABEL_NAMES[c]
                v = val_dice.get(name, float("nan"))
                row[f"val_dice_{name}"] = round(v, 4) if v == v else "nan"

            if val_mean > best_val_dice:
                best_val_dice, best_epoch, no_improve = val_mean, epoch, 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": best_val_dice,
                    "class_weights": class_weights,
                    "lambda_div": lambda_div,
                    "lambda_push": lambda_push,
                    "lambda_pull": lambda_pull,
                    "single_scale": single_scale,
                    "no_soft_mask": no_soft_mask,
                    "hard_mask": hard_mask,
                    "mask_quantile": mask_quantile,
                    "hard_mask_active": model.hard_mask_active,
                }, ckpt_path)
                flag = " ← best"
            else:
                no_improve += 1
                flag = ""

            pp_str = (f" push={avg(total_push):.4f} pull={avg(total_pull):.4f}"
                      if lambda_push > 0 or lambda_pull > 0 else "")
            print(f"  [{current_phase}] Ep {epoch:3d}/{MAX_EPOCHS} | "
                  f"loss={avg(total_loss):.4f} "
                  f"(D={avg(total_dice):.4f} CE={avg(total_ce):.4f} div={avg(total_div):.4f}{pp_str}) | "
                  f"val={val_mean:.4f}{flag} | lr={current_lr:.2e} | {epoch_time:.1f}s",
                  flush=True)
        else:
            row["val_mean_fg_dice"] = ""
            for c in range(1, NUM_CLASSES):
                row[f"val_dice_{LABEL_NAMES[c]}"] = ""
            if epoch % 10 == 0:
                print(f"  [{current_phase}] Ep {epoch:3d}/{MAX_EPOCHS} | "
                      f"loss={avg(total_loss):.4f} | lr={current_lr:.2e} | {epoch_time:.1f}s",
                      flush=True)

        writer.writerow(row)
        csv_file.flush()

        if current_phase != "A" and no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} val checks)")
            break

    csv_file.close()
    print(f"\n  Best val mean fg Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Log        : {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ProtoSegNet")
    parser.add_argument("--modality", required=True, choices=["ct", "mr"])
    parser.add_argument("--lambda-div",  type=float, default=LAMBDA_DIV,
                        help="Weight on diversity loss (default 0.01)")
    parser.add_argument("--lambda-push", type=float, default=LAMBDA_PUSH,
                        help="Weight on push alignment loss (default 0.0 = off)")
    parser.add_argument("--lambda-pull", type=float, default=LAMBDA_PULL,
                        help="Weight on pull alignment loss (default 0.0 = off)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to checkpoint and log filenames (e.g. _pp)")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="Resume training from this epoch (skips earlier phases)")
    parser.add_argument("--init-checkpoint", type=str, default="",
                        help="Load model weights from this checkpoint before training")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help=f"Total epochs to train (default {MAX_EPOCHS}; use 50 for ablations)")
    parser.add_argument("--single-scale", action="store_true",
                        help="Ablation: use only level-4 prototypes (no multi-scale)")
    parser.add_argument("--no-soft-mask", action="store_true",
                        help="Ablation: skip mask module entirely; raw encoder features to decoder")
    parser.add_argument("--hard-mask", action="store_true",
                        help="Stage 9: use HardMaskModule (STE binary gate) instead of SoftMaskModule")
    parser.add_argument("--mask-quantile", type=float, default=0.5,
                        help="Spatial quantile threshold for HardMaskModule (default 0.5)")
    args = parser.parse_args()
    train(args.modality, lambda_div=args.lambda_div, lambda_push=args.lambda_push,
          lambda_pull=args.lambda_pull, suffix=args.suffix,
          start_epoch=args.start_epoch, init_checkpoint=args.init_checkpoint,
          max_epochs=args.max_epochs, single_scale=args.single_scale,
          no_soft_mask=args.no_soft_mask, hard_mask=args.hard_mask,
          mask_quantile=args.mask_quantile)


if __name__ == "__main__":
    main()
