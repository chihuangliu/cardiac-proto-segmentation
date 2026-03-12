"""
scripts/explore_data.py
Explore the MM-WHS dataset: statistics, label distribution, sample visualization.
Run: .venv/bin/python scripts/explore_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.data.mmwhs_dataset import (
    MMWHSSliceDataset,
    MMWHSPatientDataset,
    make_dataloaders,
    save_splits_json,
    LABEL_NAMES,
    NUM_CLASSES,
)

DATA_DIR = "data/pack/processed_data"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def print_split_summary():
    print("\n=== Dataset Split Summary ===")
    for modality in ("ct", "mr"):
        print(f"\n  [{modality.upper()}]")
        for split in ("train", "val", "test"):
            ds = MMWHSSliceDataset(DATA_DIR, modality, split)
            patients = ds.get_patients()
            print(f"    {split:5s}: {len(ds):4d} slices, {len(patients)} patients → {patients}")


def compute_label_distribution():
    print("\n=== Label Class Distribution (train set) ===")
    for modality in ("ct", "mr"):
        ds = MMWHSSliceDataset(DATA_DIR, modality, "train")
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for sample in ds:
            lbl = sample["label"].numpy()
            for c in range(NUM_CLASSES):
                counts[c] += (lbl == c).sum()
        total = counts.sum()
        print(f"\n  [{modality.upper()}] total voxels: {total:,}")
        for c, name in LABEL_NAMES.items():
            pct = 100 * counts[c] / total
            print(f"    Class {c} {name:12s}: {counts[c]:10,} ({pct:.2f}%)")
    return counts


def visualize_samples():
    print("\n=== Generating sample visualizations ===")
    cmap_label = plt.get_cmap("tab10", NUM_CLASSES)
    legend_patches = [
        mpatches.Patch(color=cmap_label(i), label=f"{i}: {LABEL_NAMES[i]}")
        for i in range(NUM_CLASSES)
    ]

    for modality in ("ct", "mr"):
        ds = MMWHSSliceDataset(DATA_DIR, modality, "train")
        # Pick slices with >1 foreground class for interesting visualization
        samples_to_show = []
        for i, sample in enumerate(ds):
            lbl = sample["label"].numpy()
            if len(np.unique(lbl)) >= 4:
                samples_to_show.append(sample)
            if len(samples_to_show) == 4:
                break

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"{modality.upper()} — Sample Slices", fontsize=14)
        for col, sample in enumerate(samples_to_show):
            img = sample["image"].squeeze().numpy()
            lbl = sample["label"].numpy()
            patient = sample["patient"]
            fname = sample["filename"]

            axes[0, col].imshow(img, cmap="gray")
            axes[0, col].set_title(f"{patient}\n{fname}", fontsize=7)
            axes[0, col].axis("off")

            axes[1, col].imshow(img, cmap="gray")
            label_rgb = cmap_label(lbl / NUM_CLASSES)
            label_rgb[..., 3] = (lbl > 0).astype(float) * 0.6
            axes[1, col].imshow(label_rgb)
            axes[1, col].set_title("Overlay", fontsize=7)
            axes[1, col].axis("off")

        fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8)
        out_path = RESULTS_DIR / f"sample_{modality}.png"
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  Saved: {out_path}")


def test_dataloader():
    print("\n=== Testing DataLoader ===")
    import time
    import torch
    for modality in ("ct", "mr"):
        loaders = make_dataloaders(DATA_DIR, modality, batch_size=8, num_workers=0)
        loader = loaders["train"]
        t0 = time.time()
        batch = next(iter(loader))
        elapsed = time.time() - t0
        img = batch["image"]
        lbl = batch["label"]
        print(f"  [{modality.upper()}] Batch image: {img.shape} {img.dtype} "
              f"range=[{img.min():.3f}, {img.max():.3f}] "
              f"| label: {lbl.shape} {lbl.dtype} unique={lbl.unique().tolist()} "
              f"| time: {elapsed:.3f}s")


def test_patient_dataset():
    print("\n=== Testing PatientDataset ===")
    for modality in ("ct", "mr"):
        ds = MMWHSPatientDataset(DATA_DIR, modality, "test")
        sample = ds[0]
        print(f"  [{modality.upper()}] Patient '{sample['patient']}': "
              f"volume shape={sample['image'].shape}, "
              f"n_slices={sample['n_slices']}")


def save_splits():
    out = "data/splits.json"
    save_splits_json(DATA_DIR, out)


if __name__ == "__main__":
    print_split_summary()
    test_dataloader()
    test_patient_dataset()
    visualize_samples()
    save_splits()
    print("\n✓ Stage 0 verification complete.")
