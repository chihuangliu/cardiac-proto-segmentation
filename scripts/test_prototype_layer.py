"""
scripts/test_prototype_layer.py
Unit tests for Stage 3 — PrototypeLayer, SoftMaskModule, PrototypeProjection.

Tests
-----
1. Output heatmap shape  →  (B, K, M, H_l, W_l)
2. Similarity scores in  [0, log(2)]
3. Gradients flow to self.prototypes
4. SoftMask output shape matches input feature map shape
5. PrototypeProjection runs in < 2 min and updates prototype values

Usage
-----
    .venv/bin/python scripts/test_prototype_layer.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.encoder import HierarchicalEncoder2D
from src.models.prototype_layer import (
    PROTOS_PER_LEVEL,
    PrototypeLayer,
    PrototypeProjection,
    SoftMaskModule,
)

PASS = "✅"
FAIL = "❌"

# ── Shared constants ──────────────────────────────────────────────────────────
N_CLASSES   = 8
BATCH_SIZE  = 4
IMG_SIZE    = 256
ENCODER_CH  = HierarchicalEncoder2D.CHANNELS   # {1:32, 2:64, 3:128, 4:256}
SPATIAL     = {1: 128, 2: 64, 3: 32, 4: 16}
LOG2        = math.log(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Output heatmap shape
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shapes():
    print("\n[1] Heatmap shape assertions")
    all_ok = True
    for level, M in PROTOS_PER_LEVEL.items():
        C   = ENCODER_CH[level]
        H   = SPATIAL[level]
        pl  = PrototypeLayer(N_CLASSES, M, C)
        Z_l = torch.randn(BATCH_SIZE, C, H, H)

        with torch.no_grad():
            A = pl(Z_l)

        expected = (BATCH_SIZE, N_CLASSES, M, H, H)
        ok = tuple(A.shape) == expected
        all_ok = all_ok and ok
        status = PASS if ok else FAIL
        print(f"  {status}  Level {level}: expected {expected}  got {tuple(A.shape)}")

    assert all_ok, "Shape mismatch — see above"
    print(f"  {PASS} All heatmap shapes correct")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Similarity scores in [0, log(2)]
# ─────────────────────────────────────────────────────────────────────────────

def test_similarity_range():
    print("\n[2] Similarity score range in [0, log(2)]")
    all_ok = True
    for level, M in PROTOS_PER_LEVEL.items():
        C  = ENCODER_CH[level]
        H  = SPATIAL[level]
        pl = PrototypeLayer(N_CLASSES, M, C)
        Z_l = torch.randn(BATCH_SIZE, C, H, H)

        with torch.no_grad():
            A = pl(Z_l)

        lo = A.min().item()
        hi = A.max().item()
        ok = (lo >= -1e-6) and (hi <= LOG2 + 1e-6)
        all_ok = all_ok and ok
        status = PASS if ok else FAIL
        print(f"  {status}  Level {level}: min={lo:.6f}  max={hi:.6f}  log2={LOG2:.6f}")

    assert all_ok, "Scores out of [0, log(2)]"
    print(f"  {PASS} All similarity scores in [0, log(2)]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gradient flow to prototypes
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow():
    print("\n[3] Gradient flow to prototype parameters")
    all_ok = True
    for level, M in PROTOS_PER_LEVEL.items():
        C  = ENCODER_CH[level]
        H  = SPATIAL[level]
        pl = PrototypeLayer(N_CLASSES, M, C)
        Z_l = torch.randn(BATCH_SIZE, C, H, H)

        A = pl(Z_l)
        A.sum().backward()

        has_grad = pl.prototypes.grad is not None
        if has_grad:
            max_grad = pl.prototypes.grad.abs().max().item()
            ok = max_grad > 0.0
        else:
            ok = False
        all_ok = all_ok and ok
        status = PASS if ok else FAIL
        info   = f"max_grad={max_grad:.2e}" if has_grad else "grad is None"
        print(f"  {status}  Level {level}: {info}")
        pl.zero_grad()

    assert all_ok, "Prototype gradients missing or zero"
    print(f"  {PASS} Prototypes receive gradients at all levels")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SoftMaskModule output shape matches Z_l shape
# ─────────────────────────────────────────────────────────────────────────────

def test_soft_mask_shape():
    print("\n[4] SoftMaskModule output shape matches Z_l")
    soft_mask = SoftMaskModule()
    all_ok = True
    for level, M in PROTOS_PER_LEVEL.items():
        C  = ENCODER_CH[level]
        H  = SPATIAL[level]
        pl  = PrototypeLayer(N_CLASSES, M, C)
        Z_l = torch.randn(BATCH_SIZE, C, H, H)

        with torch.no_grad():
            A       = pl(Z_l)
            masked  = soft_mask(A, Z_l)

        ok = tuple(masked.shape) == tuple(Z_l.shape)
        all_ok = all_ok and ok
        status = PASS if ok else FAIL
        print(f"  {status}  Level {level}: masked shape {tuple(masked.shape)} vs Z_l {tuple(Z_l.shape)}")

    assert all_ok, "SoftMask output shape mismatch"
    print(f"  {PASS} SoftMask output shapes match Z_l at all levels")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SoftMask mask values are non-negative
# ─────────────────────────────────────────────────────────────────────────────

def test_soft_mask_nonneg():
    print("\n[5] SoftMask mask values non-negative (A ≥ 0 → mask ≥ 0)")
    soft_mask = SoftMaskModule()
    level, M = 3, PROTOS_PER_LEVEL[3]
    C, H     = ENCODER_CH[level], SPATIAL[level]
    pl       = PrototypeLayer(N_CLASSES, M, C)
    Z_l      = torch.abs(torch.randn(BATCH_SIZE, C, H, H))  # non-neg features

    with torch.no_grad():
        A      = pl(Z_l)
        masked = soft_mask(A, Z_l)

    ok = masked.min().item() >= -1e-6
    status = PASS if ok else FAIL
    print(f"  {status}  masked.min() = {masked.min().item():.4f}")
    assert ok, "Masked features contain unexpected negatives"
    print(f"  {PASS} Mask is non-negative")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PrototypeProjection updates values and completes in < 2 min
# ─────────────────────────────────────────────────────────────────────────────

def test_projection():
    print("\n[6] PrototypeProjection — updates prototypes + runtime < 2 min")

    # Build a tiny synthetic dataloader (50 slices, all classes present)
    N_SLICES = 50
    imgs   = torch.randn(N_SLICES, 1, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, N_CLASSES, (N_SLICES, IMG_SIZE, IMG_SIZE))
    ds     = TensorDataset(imgs, labels)
    dl     = DataLoader(ds, batch_size=8, shuffle=False)

    encoder = HierarchicalEncoder2D()
    proto_layers = {
        level: PrototypeLayer(N_CLASSES, M, ENCODER_CH[level])
        for level, M in PROTOS_PER_LEVEL.items()
    }

    # Record prototype values before projection
    before = {
        (level, k, m): proto_layers[level].prototypes.data[k, m].clone()
        for level, M in PROTOS_PER_LEVEL.items()
        for k in range(N_CLASSES)
        for m in range(M)
    }

    projector = PrototypeProjection(encoder, proto_layers, device="cpu")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        save_path = tmp.name

    t0 = time.time()
    meta = projector.project(dl, save_path=save_path)
    elapsed = time.time() - t0

    # Check runtime
    ok_time = elapsed < 120.0
    status = PASS if ok_time else FAIL
    print(f"  {status}  Projection time: {elapsed:.1f}s  (target < 120s)")

    # Check that prototypes actually changed for classes present in labels
    n_changed = 0
    for (level, k, m), old_val in before.items():
        new_val = proto_layers[level].prototypes.data[k, m]
        if not torch.allclose(old_val, new_val, atol=1e-6):
            n_changed += 1

    ok_changed = n_changed > 0
    status = PASS if ok_changed else FAIL
    print(f"  {status}  Prototypes updated: {n_changed} / {len(before)}")

    # Check save file exists and is loadable
    data = torch.load(save_path, weights_only=False)
    ok_save = "proto_state" in data and "metadata" in data
    status = PASS if ok_save else FAIL
    print(f"  {status}  Save file loadable with keys: {list(data.keys())}")
    os.unlink(save_path)

    assert ok_time and ok_changed and ok_save
    print(f"  {PASS} Projection complete")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integration: encoder → PrototypeLayer → SoftMask end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_shapes():
    print("\n[7] End-to-end: encoder → PrototypeLayer → SoftMask (all levels)")
    encoder    = HierarchicalEncoder2D()
    soft_mask  = SoftMaskModule()
    proto_layers = {
        level: PrototypeLayer(N_CLASSES, M, ENCODER_CH[level])
        for level, M in PROTOS_PER_LEVEL.items()
    }

    x = torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    feats = encoder(x)

    all_ok = True
    for level, Z_l in feats.items():
        pl     = proto_layers[level]
        A      = pl(Z_l)
        masked = soft_mask(A, Z_l)

        # Shapes
        M  = PROTOS_PER_LEVEL[level]
        C  = ENCODER_CH[level]
        H  = SPATIAL[level]

        shape_A      = tuple(A.shape)
        shape_masked = tuple(masked.shape)
        exp_A        = (BATCH_SIZE, N_CLASSES, M, H, H)
        exp_masked   = (BATCH_SIZE, C, H, H)

        ok = (shape_A == exp_A) and (shape_masked == exp_masked)
        all_ok = all_ok and ok
        status = PASS if ok else FAIL
        print(f"  {status}  Level {level}: A={shape_A}  masked={shape_masked}")

    assert all_ok
    print(f"  {PASS} All levels pass end-to-end shape check")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Stage 3 — PrototypeLayer / SoftMaskModule / Projection Tests")
    print("=" * 65)

    test_output_shapes()
    test_similarity_range()
    test_gradient_flow()
    test_soft_mask_shape()
    test_soft_mask_nonneg()
    test_projection()
    test_end_to_end_shapes()

    print("\n" + "=" * 65)
    print("  All Stage 3 tests passed ✅")
    print("=" * 65)
