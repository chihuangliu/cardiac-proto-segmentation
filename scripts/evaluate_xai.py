#!/usr/bin/env python
"""
scripts/evaluate_xai.py
Stage 6 — XAI Evaluation

Evaluate all 4 XAI metrics (AP, IDS, Faithfulness, Stability) on test patients.

Usage:
    python scripts/evaluate_xai.py --modality ct --checkpoint checkpoints/proto_seg_ct.pth
    python scripts/evaluate_xai.py --modality mr --checkpoint checkpoints/proto_seg_mr.pth
    python scripts/evaluate_xai.py --modality ct --checkpoint ... --max-slices 30
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.mmwhs_dataset import MMWHSPatientDataset, LABEL_NAMES
from src.models.proto_seg_net import ProtoSegNet
from src.metrics.activation_precision import activation_precision_patient
from src.metrics.incremental_deletion import incremental_deletion_patient
from src.metrics.faithfulness import faithfulness_patient
from src.metrics.stability import stability_patient

DATA_DIR = ROOT / "data" / "pack" / "processed_data"
N_CLASSES = 8
FG_CLASSES = [LABEL_NAMES[k] for k in range(1, N_CLASSES)]   # exclude Background


# ── helpers ──────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fmt(val: float, fmt: str = ".4f") -> str:
    return "nan" if val != val else f"{val:{fmt}}"


def print_ap_table(patient: str, ap: dict[str, float]) -> None:
    fg = {k: v for k, v in ap.items() if k != "Background"}
    valid = [v for v in fg.values() if v == v]
    mean_ap = sum(valid) / len(valid) if valid else float("nan")
    print(f"\n  Activation Precision — {patient}")
    print(f"  {'Class':<14} {'AP':>7}")
    print(f"  {'-'*22}")
    for cls, val in fg.items():
        print(f"  {cls:<14} {_fmt(val):>7}")
    print(f"  {'-'*22}")
    print(f"  {'Mean (fg)':<14} {_fmt(mean_ap):>7}")


def print_summary(
    patient: str,
    ap: dict[str, float],
    ids_res: dict[str, float],
    faith_res: dict[str, float],
    stab_res: dict[str, float],
) -> None:
    fg_vals = [v for k, v in ap.items() if k != "Background" and v == v]
    mean_ap = sum(fg_vals) / len(fg_vals) if fg_vals else float("nan")

    print(f"\n{'='*60}")
    print(f" Patient: {patient}")
    print(f"{'='*60}")
    print_ap_table(patient, ap)
    print(f"\n  IDS            : {_fmt(ids_res['ids'])}  ± {_fmt(ids_res['ids_std'])}")
    print(f"  Faithfulness   : {_fmt(faith_res['faithfulness'])}  ± {_fmt(faith_res['faithfulness_std'])}")
    print(f"  Stability      : {_fmt(stab_res['stability'])}  ± {_fmt(stab_res['stability_std'])}")
    print()


def check_against_targets(modality: str, results: list[dict]) -> None:
    """Print pass/fail against Stage 6 success criteria."""
    targets = {
        "ct": {"ap": 0.70, "ids": 0.45, "faithfulness": 0.55, "stability": 0.20},
        "mr": {"ap": 0.65, "ids": 0.50, "faithfulness": 0.50, "stability": 0.25},
    }[modality]

    print(f"\n{'='*60}")
    print(" Stage 6 Success Criteria")
    print(f"{'='*60}")
    for r in results:
        p = r["patient"]
        ap_mean = r["ap_mean"]
        ids     = r["ids"]
        faith   = r["faithfulness"]
        stab    = r["stability"]

        def ok(val, target, lower_is_better=False):
            if val != val:
                return "nan"
            return "✅" if (val <= target if lower_is_better else val >= target) else "❌"

        print(f"\n  {p}")
        print(f"    AP    : {_fmt(ap_mean)}  (target ≥ {targets['ap']})  {ok(ap_mean, targets['ap'])}")
        print(f"    IDS   : {_fmt(ids)}  (target ≤ {targets['ids']})  {ok(ids, targets['ids'], lower_is_better=True)}")
        print(f"    Faith : {_fmt(faith)}  (target ≥ {targets['faithfulness']})  {ok(faith, targets['faithfulness'])}")
        print(f"    Stab  : {_fmt(stab)}  (target ≤ {targets['stability']})  {ok(stab, targets['stability'], lower_is_better=True)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XAI metrics for ProtoSegNet")
    parser.add_argument("--modality", required=True, choices=["ct", "mr"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Max slices per patient for IDS and Faithfulness (default: all).",
    )
    parser.add_argument(
        "--skip-ids",
        action="store_true",
        help="Skip IDS computation (slowest metric).",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Skip Faithfulness computation.",
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"Device  : {device}")
    print(f"Modality: {args.modality.upper()}")
    print(f"Checkpoint: {args.checkpoint}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ProtoSegNet(
        n_classes=N_CLASSES,
        single_scale=ckpt.get("single_scale", False),
        no_soft_mask=ckpt.get("no_soft_mask", False),
        hard_mask=ckpt.get("hard_mask", False),
        mask_quantile=ckpt.get("mask_quantile", 0.5),
    ).to(device)
    model.hard_mask_active = ckpt.get("hard_mask_active", ckpt.get("hard_mask", False))
    state = (ckpt.get("model_state_dict")
             or ckpt.get("model_state")
             or ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Load test patients ────────────────────────────────────────────────────
    test_ds = MMWHSPatientDataset(DATA_DIR, modality=args.modality, split="test")
    print(f"Test patients: {len(test_ds)}")

    all_results: list[dict] = []
    t0_total = time.time()

    for i in range(len(test_ds)):
        sample = test_ds[i]
        patient = sample["patient"]
        images = sample["image"]    # (S, 1, H, W)
        labels = sample["label"]    # (S, H, W)
        S = images.shape[0]
        print(f"\n[{i+1}/{len(test_ds)}] {patient}  ({S} slices)")

        # ── AP (all slices — fast) ────────────────────────────────────────────
        t0 = time.time()
        ap = activation_precision_patient(model, images, labels, device, batch_size=16)
        fg_valid = [v for k, v in ap.items() if k != "Background" and v == v]
        ap_mean = sum(fg_valid) / len(fg_valid) if fg_valid else float("nan")
        print(f"  AP done  ({time.time()-t0:.1f}s)  mean_fg={_fmt(ap_mean)}")

        # ── IDS ───────────────────────────────────────────────────────────────
        if args.skip_ids:
            ids_res = {"ids": float("nan"), "ids_std": float("nan")}
        else:
            t0 = time.time()
            ids_res = incremental_deletion_patient(
                model, images, labels, device, max_slices=args.max_slices
            )
            print(f"  IDS done ({time.time()-t0:.1f}s)  ids={_fmt(ids_res['ids'])}")

        # ── Faithfulness ──────────────────────────────────────────────────────
        if args.skip_faithfulness:
            faith_res = {"faithfulness": float("nan"), "faithfulness_std": float("nan")}
        else:
            t0 = time.time()
            faith_res = faithfulness_patient(
                model, images, device, max_slices=args.max_slices
            )
            print(f"  Faith done ({time.time()-t0:.1f}s)  r={_fmt(faith_res['faithfulness'])}")

        # ── Stability (all slices — fast, batched) ────────────────────────────
        t0 = time.time()
        stab_res = stability_patient(model, images, device)
        print(f"  Stab done ({time.time()-t0:.1f}s)  stab={_fmt(stab_res['stability'])}")

        # ── Print detailed summary ─────────────────────────────────────────────
        print_summary(patient, ap, ids_res, faith_res, stab_res)

        all_results.append({
            "patient": patient,
            "ap": ap,
            "ap_mean": ap_mean,
            "ids": ids_res["ids"],
            "ids_std": ids_res["ids_std"],
            "faithfulness": faith_res["faithfulness"],
            "faithfulness_std": faith_res["faithfulness_std"],
            "stability": stab_res["stability"],
            "stability_std": stab_res["stability_std"],
        })

    total_time = time.time() - t0_total
    print(f"\nTotal evaluation time: {total_time/60:.1f} min")

    # ── Success criteria check ────────────────────────────────────────────────
    check_against_targets(args.modality, all_results)

    # ── Aggregate across patients ─────────────────────────────────────────────
    if len(all_results) > 1:
        import statistics

        def nanmean(vals):
            v = [x for x in vals if x == x]
            return sum(v) / len(v) if v else float("nan")

        print(f"\n{'='*60}")
        print(f" Aggregate over {len(all_results)} patients")
        print(f"{'='*60}")
        print(f"  Mean AP        : {_fmt(nanmean([r['ap_mean'] for r in all_results]))}")
        print(f"  Mean IDS       : {_fmt(nanmean([r['ids']     for r in all_results]))}")
        print(f"  Mean Faith     : {_fmt(nanmean([r['faithfulness'] for r in all_results]))}")
        print(f"  Mean Stab      : {_fmt(nanmean([r['stability']    for r in all_results]))}")


if __name__ == "__main__":
    main()
