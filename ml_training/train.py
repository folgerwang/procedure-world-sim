"""
Training loop for the multi-view joint heatmap U-Net.

Usage:
    python train.py --config configs/config_default.yaml
    python train.py --config configs/config_quick_test.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet_heatmap import build_model
from data.data_loader import create_dataloaders
from utils.heatmap_utils import extract_peaks
from utils.metrics import pck, pck_per_joint


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model: nn.Module, loader, optimizer, criterion,
                    device: str, epoch: int, writer: SummaryWriter
                    ) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        inp = batch["input"].to(device)           # (B, 7, H, W)
        target = batch["heatmaps"].to(device)     # (B, J, H, W)

        optimizer.zero_grad()
        pred = model(inp)                          # (B, J, H, W)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    writer.add_scalar("train/loss", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def validate(model: nn.Module, loader, criterion, device: str,
             epoch: int, writer: SummaryWriter,
             resolution: int = 256) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_pck = 0.0
    num_batches = 0

    for batch in loader:
        inp = batch["input"].to(device)
        target = batch["heatmaps"].to(device)

        pred = model(inp)
        loss = criterion(pred, target)

        # Compute PCK from predicted vs GT peaks.
        pred_peaks, _ = extract_peaks(pred)
        gt_peaks, _ = extract_peaks(target)
        batch_pck = pck(pred_peaks, gt_peaks,
                        threshold_frac=0.2, image_size=resolution)

        total_loss += loss.item()
        total_pck += batch_pck
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_pck = total_pck / max(num_batches, 1)

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/pck_0.2", avg_pck, epoch)
    return avg_loss, avg_pck


def train(cfg: dict):
    """Main training function."""
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")
    print(f"Config: {cfg['model']}")

    # ── Model ──
    model = build_model(
        num_joints=cfg["model"]["num_joints"],
        in_channels=cfg["model"]["input_channels"],
        base_channels=cfg["model"]["base_channels"],
        device=device,
    )

    # ── Data ──
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_loader, val_loader, _ = create_dataloaders(
        h5_path=data_cfg["dataset_path"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        train_frac=data_cfg["train_split"],
        val_frac=data_cfg["val_split"],
        aug_config=aug_cfg,
    )

    # ── Optimizer & scheduler ──
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["lr_step_size"],
        gamma=cfg["training"]["lr_gamma"],
    )
    criterion = nn.MSELoss()

    # ── Logging ──
    log_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # ── Checkpointing ──
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    # ── Training loop ──
    num_epochs = cfg["training"]["epochs"]
    resolution = cfg["model"]["resolution"]

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer)
        val_loss, val_pck = validate(
            model, val_loader, criterion, device, epoch, writer, resolution)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{num_epochs}  "
              f"train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  "
              f"val_pck={val_pck:.3f}  "
              f"lr={lr:.1e}  "
              f"({elapsed:.1f}s)")

        # Save best model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_pck": val_pck,
                "config": cfg,
            }, "checkpoints/best_model.pth")
            print(f"  → Saved best model (val_loss={val_loss:.5f})")

        # Save periodic checkpoint.
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, f"checkpoints/epoch_{epoch:03d}.pth")

    writer.close()
    print(f"\nTraining complete. Best model at epoch {best_epoch} "
          f"(val_loss={best_val_loss:.5f})")
    print(f"Checkpoint: checkpoints/best_model.pth")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train joint heatmap U-Net")
    parser.add_argument("--config", default="configs/config_default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dataset", default=None,
                        help="Override dataset path from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--device", default=None,
                        help="Override device (cuda/cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides.
    if args.dataset:
        cfg["data"]["dataset_path"] = args.dataset
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.device:
        cfg["training"]["device"] = args.device

    train(cfg)


if __name__ == "__main__":
    main()
