#!/usr/bin/env python
"""
scripts/test_hard_mask.py
Stage 9 — Unit tests for HardMaskModule and ProtoSegNet hard-mask integration.

Tests
-----
1. Hard-masked output has exact zeros at bottom-quantile spatial locations
2. Mask is binary (values are 0 or 1) in the forward pass
3. Gradient flows to prototype parameters through STE
4. End-to-end shapes unchanged: logits (B,8,256,256), heatmaps dict correct
5. Hard-mask checkpoint round-trips through save/load correctly
6. HardMaskModule with quantile=0.0 behaves like SoftMask (no zeroing)
7. HardMaskModule with quantile=0.99 zeroes almost all locations
"""

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.prototype_layer import HardMaskModule, SoftMaskModule, PrototypeLayer
from src.models.proto_seg_net import ProtoSegNet

PASS = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"


def run(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"       {type(e).__name__}: {e}")
        return False


# ── Fixtures ──────────────────────────────────────────────────────────────────

B, K, M, H, W, C = 2, 8, 4, 32, 32, 128


def make_heatmap():
    """Random heatmap in (0,1] matching PrototypeLayer L2 output range."""
    return torch.rand(B, K, M, H, W).clamp(min=1e-3)


def make_feature():
    return torch.randn(B, C, H, W)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_zeros_at_bottom_quantile():
    """Spatial locations below threshold must be exactly zero in the output."""
    A = make_heatmap()
    Z = make_feature()
    q = 0.5
    mod = HardMaskModule(quantile=q)
    out = mod(A, Z)

    # Reconstruct the mask that was applied
    A_max = A.max(dim=2).values          # (B, K, H, W)
    W = A_max.max(dim=1, keepdim=True).values  # (B, 1, H, W)
    tau = W.flatten(1).quantile(q, dim=1).view(-1, 1, 1, 1)
    below_threshold = (W < tau).expand_as(Z)

    # All output values at below-threshold locations must be zero
    assert out[below_threshold].abs().max().item() == 0.0, (
        "Non-zero values found at locations that should be masked out"
    )


def test_mask_is_binary():
    """The effective mask applied to Z must be binary (0.0 or 1.0)."""
    A = make_heatmap()
    Z = torch.ones(B, C, H, W)  # unit features so output == mask
    mod = HardMaskModule(quantile=0.5)
    out = mod(A, Z)
    unique_vals = out.unique()
    non_binary = [v.item() for v in unique_vals if abs(v.item()) > 1e-6 and abs(v.item() - 1.0) > 1e-6]
    assert len(non_binary) == 0, f"Non-binary values in mask: {non_binary[:5]}"


def test_gradient_flows_to_prototypes():
    """STE must allow gradients to reach prototype parameters."""
    proto = PrototypeLayer(n_classes=K, n_protos=M, feature_dim=C)
    mod = HardMaskModule(quantile=0.5)
    Z = torch.randn(B, C, H, W)

    A = proto(Z)
    out = mod(A, Z)
    loss = out.sum()
    loss.backward()

    grad = proto.prototypes.grad
    assert grad is not None, "No gradient reached prototype parameters"
    assert grad.abs().sum().item() > 0.0, "Prototype gradient is all zeros"


def test_end_to_end_shapes():
    """ProtoSegNet with hard_mask=True must return correct output shapes."""
    model = ProtoSegNet(hard_mask=True, mask_quantile=0.5)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        logits, heatmaps = model(x)

    assert logits.shape == (2, 8, 256, 256), f"Wrong logits shape: {logits.shape}"
    assert set(heatmaps.keys()) == {1, 2, 3, 4}, f"Wrong heatmap levels: {heatmaps.keys()}"
    expected_spatial = {1: (128, 128), 2: (64, 64), 3: (32, 32), 4: (16, 16)}
    expected_protos  = {1: 4, 2: 3, 3: 2, 4: 2}
    for l, A in heatmaps.items():
        assert A.shape == (2, 8, expected_protos[l], *expected_spatial[l]), (
            f"Level {l} heatmap shape {A.shape} != expected"
        )


def test_checkpoint_roundtrip():
    """hard_mask and mask_quantile must survive a save/load cycle."""
    model = ProtoSegNet(hard_mask=True, mask_quantile=0.3)
    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        torch.save({
            "model_state_dict": model.state_dict(),
            "hard_mask": model.hard_mask,
            "mask_quantile": model.mask_quantile,
            "single_scale": model.single_scale,
            "no_soft_mask": model.no_soft_mask,
        }, f.name)
        ckpt = torch.load(f.name, weights_only=False)

    model2 = ProtoSegNet(
        hard_mask=ckpt.get("hard_mask", False),
        mask_quantile=ckpt.get("mask_quantile", 0.5),
    )
    model2.load_state_dict(ckpt["model_state_dict"])

    assert model2.hard_mask is True
    assert abs(model2.mask_quantile - 0.3) < 1e-6
    assert isinstance(model2.mask_module, HardMaskModule)


def test_quantile_zero_no_zeroing():
    """quantile=0.0 means threshold=min(W), so no locations are zeroed."""
    A = make_heatmap()
    Z = make_feature()
    mod = HardMaskModule(quantile=0.0)
    out = mod(A, Z)
    # With q=0, tau = min(W) per sample; all W >= tau, so mask is all ones
    assert (out == 0).sum().item() == 0, "quantile=0.0 should not zero any locations"


def test_quantile_high_zeroes_most():
    """quantile=0.99 should zero ~99% of spatial locations."""
    A = make_heatmap()
    Z = torch.ones(B, C, H, W)
    mod = HardMaskModule(quantile=0.99)
    out = mod(A, Z)
    zero_frac = (out == 0).float().mean().item()
    assert zero_frac > 0.95, (
        f"Expected >95% zeros with quantile=0.99, got {zero_frac:.2%}"
    )


def test_hard_vs_soft_nonzero_regions():
    """Hard mask output must be a strict subset of soft mask output (no new activations)."""
    A = make_heatmap()
    Z = torch.ones(B, C, H, W)

    soft = SoftMaskModule()(A, Z)
    hard = HardMaskModule(quantile=0.5)(A, Z)

    # Wherever hard is non-zero, soft must also be non-zero
    hard_nonzero = hard != 0
    soft_at_hard = soft[hard_nonzero]
    assert (soft_at_hard != 0).all(), (
        "Hard mask activated locations that soft mask had zeroed — unexpected"
    )


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Zeros at bottom-quantile locations",  test_zeros_at_bottom_quantile),
        ("Mask values are binary (0 or 1)",      test_mask_is_binary),
        ("Gradient flows to prototypes (STE)",   test_gradient_flows_to_prototypes),
        ("End-to-end output shapes correct",     test_end_to_end_shapes),
        ("Checkpoint round-trip (hard_mask key)",test_checkpoint_roundtrip),
        ("quantile=0.0 → no zeroing",            test_quantile_zero_no_zeroing),
        ("quantile=0.99 → >95% zeroed",          test_quantile_high_zeroes_most),
        ("Hard mask ⊆ soft mask activations",    test_hard_vs_soft_nonzero_regions),
    ]

    print(f"\nStage 9 — HardMaskModule unit tests ({len(tests)} tests)\n")
    results = [run(name, fn) for name, fn in tests]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(tests)} tests passed")
    if n_pass < len(tests):
        sys.exit(1)
