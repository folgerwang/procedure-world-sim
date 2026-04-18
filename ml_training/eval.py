"""
Evaluation script: load a trained checkpoint, run inference on the test
split, and report PCK / MPJPE / per-joint metrics.

Usage:
    python eval.py --checkpoint checkpoints/best_model.pth \
                   --config configs/config_default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet_heatmap import build_model
from data.data_loader import create_dataloaders
from utils.heatmap_utils import extract_peaks
from utils.metrics import pck, pck_per_joint
from utils.render_utils import STANDARD_JOINT_NAMES


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(cfg: dict, checkpoint_path: str):
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ── Load model ──
    model = build_model(
        num_joints=cfg["model"]["num_joints"],
        in_channels=cfg["model"]["input_channels"],
        base_channels=cfg["model"]["base_channels"],
        device=device,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?'):.5f})")

    # ── Data ──
    data_cfg = cfg["data"]
    _, _, test_loader = create_dataloaders(
        h5_path=data_cfg["dataset_path"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        train_frac=data_cfg["train_split"],
        val_frac=data_cfg["val_split"],
    )

    resolution = cfg["model"]["resolution"]
    num_joints = cfg["model"]["num_joints"]

    # ── Accumulate metrics ──
    all_pred_peaks = []
    all_gt_peaks = []
    total_loss = 0.0
    num_batches = 0
    criterion = torch.nn.MSELoss()

    for batch in test_loader:
        inp = batch["input"].to(device)
        target = batch["heatmaps"].to(device)

        pred = model(inp)
        loss = criterion(pred, target)

        pred_peaks, pred_conf = extract_peaks(pred)
        gt_peaks, _ = extract_peaks(target)

        all_pred_peaks.append(pred_peaks.cpu())
        all_gt_peaks.append(gt_peaks.cpu())
        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        print("No test samples found!")
        return

    avg_loss = total_loss / num_batches

    all_pred = torch.cat(all_pred_peaks, dim=0)   # (N, J, 2)
    all_gt = torch.cat(all_gt_peaks, dim=0)        # (N, J, 2)

    # ── Overall metrics ──
    pck_02 = pck(all_pred, all_gt, threshold_frac=0.2, image_size=resolution)
    pck_01 = pck(all_pred, all_gt, threshold_frac=0.1, image_size=resolution)
    pck_005 = pck(all_pred, all_gt, threshold_frac=0.05, image_size=resolution)

    # Mean pixel error.
    pixel_err = torch.norm(all_pred - all_gt, dim=2).mean().item()

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS ({all_pred.shape[0]} samples)")
    print(f"{'='*60}")
    print(f"  MSE Loss:        {avg_loss:.5f}")
    print(f"  Mean Pixel Error: {pixel_err:.2f} px")
    print(f"  PCK@0.05:        {pck_005:.3f}")
    print(f"  PCK@0.10:        {pck_01:.3f}")
    print(f"  PCK@0.20:        {pck_02:.3f}")

    # ── Per-joint metrics ──
    per_joint = pck_per_joint(all_pred, all_gt,
                              threshold_frac=0.2, image_size=resolution)
    per_joint_err = torch.norm(all_pred - all_gt, dim=2).mean(dim=0)

    print(f"\n{'='*60}")
    print(f"PER-JOINT BREAKDOWN")
    print(f"{'='*60}")
    print(f"  {'Joint':<20s}  {'PCK@0.2':>8s}  {'Pixel Err':>9s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*9}")
    for j in range(min(num_joints, len(STANDARD_JOINT_NAMES))):
        name = STANDARD_JOINT_NAMES[j]
        print(f"  {name:<20s}  {per_joint[j]:.3f}     {per_joint_err[j]:.2f} px")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate joint heatmap model")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", default="configs/config_default.yaml",
                        help="Config YAML")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["training"]["device"] = args.device

    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
