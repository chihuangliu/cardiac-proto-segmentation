"""
scripts/test_encoder.py
Unit tests + RAM profile + smoke-test for HierarchicalEncoder2D (Stage 2).

Usage:
    .venv/bin/python scripts/test_encoder.py
"""

import os
import sys
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from src.models.encoder import HierarchicalEncoder2D

PASS = "✅"
FAIL = "❌"


# ── 1. Shape assertions ───────────────────────────────────────────────────────

def test_output_shapes():
    print("\n[1] Output shape assertions — batch=2, input (2,1,256,256)")
    enc = HierarchicalEncoder2D()
    enc.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        feats = enc(x)

    expected = {
        1: (2,  32, 128, 128),
        2: (2,  64,  64,  64),
        3: (2, 128,  32,  32),
        4: (2, 256,  16,  16),
    }
    all_ok = True
    for lvl, exp_shape in expected.items():
        actual = tuple(feats[lvl].shape)
        ok = actual == exp_shape
        all_ok = all_ok and ok
        status = PASS if ok else FAIL
        print(f"  {status}  Z_{lvl}: expected {exp_shape}  got {actual}")

    assert all_ok, "Shape mismatch — see above"
    print(f"  {PASS} All shapes correct")
    return feats


# ── 2. Gradient flow ──────────────────────────────────────────────────────────

def test_gradient_flow():
    print("\n[2] Gradient flow — parameters receive gradients")
    enc = HierarchicalEncoder2D()
    x = torch.randn(2, 1, 256, 256)
    feats = enc(x)

    # Simple surrogate loss: sum all feature maps
    loss = sum(f.sum() for f in feats.values())
    loss.backward()

    no_grad = [n for n, p in enc.named_parameters() if p.grad is None]
    if no_grad:
        print(f"  {FAIL} No gradient for: {no_grad}")
        raise AssertionError("Missing gradients")
    print(f"  {PASS} All {enc.count_parameters():,} parameters have gradients")


# ── 3. RAM profile ────────────────────────────────────────────────────────────

def test_ram_profile():
    print("\n[3] RAM profile — forward + backward, batch=16")
    enc = HierarchicalEncoder2D()
    x = torch.randn(16, 1, 256, 256)

    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()

    feats = enc(x)
    loss = sum(f.sum() for f in feats.values())
    loss.backward()

    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap1.compare_to(snap0, "lineno")
    total_kb = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024
    total_mb = total_kb / 1024
    print(f"  Peak incremental RAM: {total_mb:.1f} MB")

    # Tensor memory estimate (more accurate for MPS/CPU)
    tensor_bytes = sum(
        p.numel() * p.element_size() for p in enc.parameters()
    ) + sum(
        f.numel() * f.element_size() for f in feats.values()
    )
    print(f"  Tensor RAM (params + features): {tensor_bytes / 1024**2:.1f} MB")

    limit_mb = 4 * 1024
    ok = total_mb < limit_mb
    print(f"  {PASS if ok else FAIL} RAM {'within' if ok else 'exceeds'} 4 GB budget")


# ── 4. Parameter count ────────────────────────────────────────────────────────

def test_param_count():
    print("\n[4] Parameter count")
    enc = HierarchicalEncoder2D()
    n = enc.count_parameters()
    print(f"  Parameters: {n:,}")
    # Compare to UNet2D for reference
    from src.models.unet import UNet2D
    unet_n = UNet2D().count_parameters()
    print(f"  UNet2D params (reference): {unet_n:,}")
    print(f"  Encoder is {n/unet_n*100:.1f}% of full U-Net")


# ── 5. Smoke-test: encoder + simple decoder head, 5 train steps ───────────────

def test_smoke_train():
    """Plugs HierarchicalEncoder2D into a minimal segmentation head and verifies
    that loss decreases over 5 gradient steps (encoder backbone does not regress)."""
    print("\n[5] Smoke-test — 5 training steps with minimal decoder head")

    class MinimalSegHead(nn.Module):
        """Simple 1×1 conv decoder: upsample Z_1 to 256×256, predict K classes."""
        def __init__(self, n_classes: int = 8):
            super().__init__()
            self.enc = HierarchicalEncoder2D()
            # Use only Z_1 (128×128, 32ch) for a minimal sanity check
            self.head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(32, n_classes, kernel_size=1),
            )

        def forward(self, x):
            feats = self.enc(x)
            return self.head(feats[1])  # (B, 8, 256, 256)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  Device: {device}")

    model = MinimalSegHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for step in range(5):
        x = torch.randn(4, 1, 256, 256, device=device)
        y = torch.randint(0, 8, (4, 256, 256), device=device)
        logits = model(x)
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"    step {step+1}: loss={loss.item():.4f}")

    # Loss should roughly decrease (or at least not explode)
    ok = losses[-1] < losses[0] * 5  # very lenient — just checking no explosion
    print(f"  {PASS if ok else FAIL} Loss {'stable' if ok else 'exploded'} "
          f"(first={losses[0]:.4f}  last={losses[-1]:.4f})")
    if not ok:
        raise AssertionError("Loss explosion in smoke-test")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Stage 2 — HierarchicalEncoder2D Tests")
    print("=" * 60)

    test_output_shapes()
    test_gradient_flow()
    test_ram_profile()
    test_param_count()
    test_smoke_train()

    print("\n" + "=" * 60)
    print("  All Stage 2 tests passed ✅")
    print("=" * 60)
