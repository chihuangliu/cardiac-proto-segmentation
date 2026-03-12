"""
scripts/test_diversity_loss.py
Stage 4 unit tests — Diversity Loss (Jeffrey's Divergence)

Tests:
  1. Identical heatmaps → high loss (prototype pair is useless)
  2. Orthogonal heatmaps → low loss (prototype pair is diverse)
  3. Background exclusion: modifying k=0 heatmaps doesn't change loss
  4. Gradient flows to prototype parameters through A_dict
  5. Loss is non-negative
  6. ProtoSegLoss combines seg + div correctly
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.losses.diversity_loss import prototype_diversity_loss, ProtoSegLoss
from src.losses.segmentation import SegmentationLoss


# ── helpers ───────────────────────────────────────────────────────────────────


def make_A_dict(B=2, K=8, pattern="random") -> dict[int, torch.Tensor]:
    """Build a fake A_dict mimicking PrototypeLayer output."""
    specs = {1: (4, 128, 128), 2: (3, 64, 64), 3: (2, 32, 32), 4: (2, 16, 16)}
    A_dict = {}
    for level, (M, H, W) in specs.items():
        if pattern == "random":
            A_dict[level] = torch.rand(B, K, M, H, W)
        elif pattern == "identical":
            # All M heatmaps identical within each (B,K)
            base = torch.rand(B, K, 1, H, W)
            A_dict[level] = base.expand(B, K, M, H, W).clone()
        elif pattern == "orthogonal":
            # Each prototype heatmap concentrated at a different spatial pixel
            A_dict[level] = torch.zeros(B, K, M, H, W)
            for m in range(M):
                row = (m * 10) % H
                col = (m * 15) % W
                A_dict[level][:, :, m, row, col] = 100.0  # near-one-hot after softmax
    return A_dict


# ── tests ─────────────────────────────────────────────────────────────────────


def test_identical_heatmaps_high_loss():
    A_dict = make_A_dict(pattern="identical")
    loss = prototype_diversity_loss(A_dict, exclude_bg=True)
    assert loss.item() > 10.0, f"Expected high loss for identical heatmaps, got {loss.item():.4f}"
    print(f"[PASS] identical heatmaps → loss = {loss.item():.4f} (high)")


def test_orthogonal_heatmaps_low_loss():
    A_dict = make_A_dict(pattern="orthogonal")
    loss = prototype_diversity_loss(A_dict, exclude_bg=True)
    assert loss.item() < 50.0, f"Expected lower loss for orthogonal heatmaps, got {loss.item():.4f}"
    print(f"[PASS] orthogonal heatmaps → loss = {loss.item():.4f} (lower than identical)")


def test_orthogonal_less_than_identical():
    loss_identical = prototype_diversity_loss(make_A_dict(pattern="identical"), exclude_bg=True)
    loss_ortho = prototype_diversity_loss(make_A_dict(pattern="orthogonal"), exclude_bg=True)
    assert loss_ortho.item() < loss_identical.item(), (
        f"Orthogonal loss ({loss_ortho.item():.4f}) should be < identical loss ({loss_identical.item():.4f})"
    )
    print(f"[PASS] orthogonal ({loss_ortho.item():.4f}) < identical ({loss_identical.item():.4f})")


def test_background_exclusion():
    """Modifying k=0 heatmaps must not change loss when exclude_bg=True."""
    A_base = make_A_dict(pattern="random")

    # Compute loss with original bg
    loss_base = prototype_diversity_loss(A_base, exclude_bg=True)

    # Make bg heatmaps identical (worst case for bg)
    A_modified = {l: A.clone() for l, A in A_base.items()}
    for level, A in A_modified.items():
        B, K, M, H, W = A.shape
        base_bg = A[:, 0:1, 0:1, :, :].expand(B, 1, M, H, W)
        A_modified[level][:, 0, :, :, :] = base_bg[:, 0, :, :, :]

    loss_modified = prototype_diversity_loss(A_modified, exclude_bg=True)

    # Losses should be equal (bg excluded from both)
    assert abs(loss_base.item() - loss_modified.item()) < 1e-3, (
        f"Background should be excluded: base={loss_base.item():.6f}, modified={loss_modified.item():.6f}"
    )
    print(f"[PASS] background exclusion: base={loss_base.item():.4f}, bg-modified={loss_modified.item():.4f}")


def test_gradient_flows():
    """Gradient must flow back through A_dict to prototype parameters."""
    from src.models.encoder import HierarchicalEncoder2D
    from src.models.prototype_layer import PrototypeLayer, PROTOS_PER_LEVEL

    encoder = HierarchicalEncoder2D()
    proto_layers = nn.ModuleDict({
        str(l): PrototypeLayer(n_classes=8, n_protos=m, feature_dim=[32, 64, 128, 256][l - 1])
        for l, m in PROTOS_PER_LEVEL.items()
    })

    x = torch.randn(2, 1, 256, 256)
    feats = encoder(x)

    A_dict = {}
    for l_str, pl in proto_layers.items():
        l = int(l_str)
        A_dict[l] = pl(feats[l])

    loss = prototype_diversity_loss(A_dict, exclude_bg=True)
    loss.backward()

    for l_str, pl in proto_layers.items():
        assert pl.prototypes.grad is not None, f"No gradient for level {l_str} prototypes"
        assert pl.prototypes.grad.abs().sum().item() > 0, f"Zero gradient for level {l_str}"
    print(f"[PASS] gradients flow to all prototype parameters, loss={loss.item():.4f}")


def test_non_negative():
    for pattern in ["random", "identical", "orthogonal"]:
        A_dict = make_A_dict(pattern=pattern)
        loss = prototype_diversity_loss(A_dict, exclude_bg=True)
        assert loss.item() >= 0.0, f"Loss negative ({loss.item()}) for pattern '{pattern}'"
    print("[PASS] loss is non-negative for all patterns")


def test_proto_seg_loss_combines():
    """ProtoSegLoss: total ≈ seg + lambda_div * div."""
    from src.models.encoder import HierarchicalEncoder2D
    from src.models.prototype_layer import PrototypeLayer, PROTOS_PER_LEVEL

    encoder = HierarchicalEncoder2D()
    proto_layers = {
        l: PrototypeLayer(n_classes=8, n_protos=m, feature_dim=[32, 64, 128, 256][l - 1])
        for l, m in PROTOS_PER_LEVEL.items()
    }

    x = torch.randn(2, 1, 256, 256)
    feats = encoder(x)
    A_dict = {l: proto_layers[l](feats[l]) for l in proto_layers}

    # Fake logits and labels
    logits = torch.randn(2, 8, 256, 256)
    labels = torch.randint(0, 8, (2, 256, 256))

    # Need class weights for SegmentationLoss
    class_weights = torch.ones(8)
    seg_loss = SegmentationLoss(class_weights=class_weights)

    lambda_div = 0.01
    criterion = ProtoSegLoss(seg_loss=seg_loss, lambda_div=lambda_div)
    out = criterion(logits, labels, A_dict)

    assert "loss" in out and "div_loss" in out
    assert out["loss"].requires_grad

    # Verify decomposition
    expected = out["dice_loss"] * 0.5 + out["ce_loss"] * 0.5 + lambda_div * out["div_loss"]
    assert abs(out["loss"].item() - expected.item()) < 1e-5, (
        f"Loss decomposition mismatch: {out['loss'].item():.6f} vs {expected.item():.6f}"
    )
    print(
        f"[PASS] ProtoSegLoss: total={out['loss'].item():.4f} "
        f"(dice={out['dice_loss'].item():.4f}, ce={out['ce_loss'].item():.4f}, "
        f"div={out['div_loss'].item():.4f})"
    )


# ── run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_non_negative,
        test_identical_heatmaps_high_loss,
        test_orthogonal_heatmaps_low_loss,
        test_orthogonal_less_than_identical,
        test_background_exclusion,
        test_gradient_flows,
        test_proto_seg_loss_combines,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
