"""
End-to-end training from captured rig editor data.

Called by the C++ AutoRigPlugin "Train Model" button.
Converts exported PNG + JSON captures into an HDF5 dataset,
trains the U-Net heatmap model, and exports to TorchScript.

Usage:
    python train_from_captures.py \
        --data_dir /path/to/training_data/model_name \
        --output_model /path/to/rig_diffusion.pt \
        --epochs 50
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_missing = []
try:
    import h5py
except ImportError:
    _missing.append("h5py")

try:
    from PIL import Image
except ImportError:
    _missing.append("Pillow")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    _missing.append("torch")

if _missing:
    print("=" * 60)
    print("  ERROR: Missing required Python packages: " + ", ".join(_missing))
    print()
    print("  Please run:")
    print("    cd ml_training")
    print("    pip install -r requirements.txt")
    print("=" * 60)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet_heatmap import build_model, UNetHeatmap
from utils.heatmap_utils import generate_heatmaps


# ── Constants ────────────────────────────────────────────────────────────────

NUM_JOINTS = 19
INPUT_CHANNELS = 7   # RGB(3) + Normal(3) + Silhouette(1)
RESOLUTION = 256
BASE_CHANNELS = 64
HEATMAP_SIGMA = 2.5


# ── Step 1: Convert captured PNGs + JSON to HDF5 ────────────────────────────

def convert_captures_to_h5(data_dir: str, h5_path: str, resolution: int = 256):
    """Read exported PNG images and JSON metadata, write an HDF5 dataset.

    The C++ exporter writes per-view files:
        <base>_view<N>_color.png       — RGB (H, W, 3)
        <base>_view<N>_silhouette.png  — grayscale (H, W)
        <base>_view<N>_heatmap.png     — combined heatmap (unused, we regenerate)
        <base>_view<N>_meta.json       — joint UVs, camera angles

    We convert these to the HDF5 format expected by JointHeatmapDataset:
        /model_0000/view_YY/input     — (7, H, W) float32
        /model_0000/view_YY/heatmaps  — (19, H, W) float32
        /model_0000/view_YY/joint_uvs — (19, 2) float32
    """
    # Find all meta JSON files
    meta_files = sorted(glob.glob(os.path.join(data_dir, "*_meta.json")))
    if not meta_files:
        print(f"ERROR: No *_meta.json files found in {data_dir}")
        return False

    print(f"Found {len(meta_files)} view(s) in {data_dir}")

    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)

    with h5py.File(h5_path, "w") as hf:
        model_grp = hf.create_group("model_0000")

        for vi, meta_path in enumerate(meta_files):
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Derive image paths from meta path
            base = meta_path.replace("_meta.json", "")
            color_path = base + "_color.png"
            sil_path = base + "_silhouette.png"

            if not os.path.exists(color_path):
                print(f"  WARNING: missing {color_path}, skipping view {vi}")
                continue

            # Load and resize images
            color_img = Image.open(color_path).convert("RGB")
            color_img = color_img.resize((resolution, resolution), Image.BILINEAR)
            color_np = np.array(color_img, dtype=np.float32) / 255.0  # (H, W, 3)

            if os.path.exists(sil_path):
                sil_img = Image.open(sil_path).convert("L")
                sil_img = sil_img.resize((resolution, resolution), Image.BILINEAR)
                sil_np = np.array(sil_img, dtype=np.float32) / 255.0  # (H, W)
            else:
                # Generate silhouette from color (non-black pixels)
                sil_np = (color_np.sum(axis=2) > 0.01).astype(np.float32)

            # Build input tensor: RGB(3) + Normal(3) + Silhouette(1) = 7 channels
            # We don't have separate normal maps from the capture, so we use the
            # color image as a proxy for normals (the rasterizer already encodes
            # lighting from normals into the color).
            # For a proper pipeline we'd export normals separately, but for
            # single-model fine-tuning this works well enough.
            rgb = color_np.transpose(2, 0, 1)         # (3, H, W)
            normals = color_np.transpose(2, 0, 1)     # (3, H, W) — proxy
            sil = sil_np[np.newaxis, :, :]             # (1, H, W)
            input_tensor = np.concatenate([rgb, normals, sil], axis=0)  # (7, H, W)

            # Build ground-truth heatmaps from joint UVs
            joints = meta["joints"]
            joint_uvs_px = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
            for j, jdata in enumerate(joints):
                if j >= NUM_JOINTS:
                    break
                u, v = jdata["uv"]
                joint_uvs_px[j, 0] = u * resolution   # column (x)
                joint_uvs_px[j, 1] = v * resolution   # row (y)

            heatmaps = generate_heatmaps(
                resolution, resolution, joint_uvs_px, sigma=HEATMAP_SIGMA)

            # Write to HDF5
            view_grp = model_grp.create_group(f"view_{vi:02d}")
            view_grp.create_dataset("input", data=input_tensor, dtype="float32")
            view_grp.create_dataset("heatmaps", data=heatmaps, dtype="float32")
            view_grp.create_dataset("joint_uvs", data=joint_uvs_px, dtype="float32")
            view_grp.create_dataset(
                "joint_visible",
                data=np.ones(NUM_JOINTS, dtype=np.float32))

            print(f"  View {vi}: input {input_tensor.shape}, "
                  f"heatmaps {heatmaps.shape}, "
                  f"az={meta.get('azimuth_deg', '?')}")

    print(f"Wrote HDF5 dataset: {h5_path}")
    return True


# ── Step 2: Simple training loop (self-contained) ───────────────────────────

class SimpleViewDataset(torch.utils.data.Dataset):
    """Lightweight HDF5 dataset for captured views (no augmentation)."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.samples = []

        with h5py.File(h5_path, "r") as hf:
            for model_key in sorted(hf.keys()):
                for view_key in sorted(hf[model_key].keys()):
                    self.samples.append((model_key, view_key))

        print(f"Dataset: {len(self.samples)} samples from {h5_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        model_key, view_key = self.samples[idx]
        with h5py.File(self.h5_path, "r") as hf:
            grp = hf[model_key][view_key]
            inp = torch.from_numpy(grp["input"][:])
            hm = torch.from_numpy(grp["heatmaps"][:])
        return {"input": inp, "heatmaps": hm}


def train_model(h5_path: str, output_model: str, epochs: int = 50,
                lr: float = 1e-3, device: str = "auto"):
    """Train the U-Net from scratch on the captured data."""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    # Build model
    model = build_model(
        num_joints=NUM_JOINTS,
        in_channels=INPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        device=device,
    )

    # Dataset — with few samples, we just use all of them for training
    dataset = SimpleViewDataset(h5_path)
    if len(dataset) == 0:
        print("ERROR: Dataset is empty, nothing to train on.")
        return False

    # With very few samples (< 20), we duplicate the dataset to fill batches
    effective_dataset = dataset
    if len(dataset) < 16:
        # Repeat the dataset to have at least 16 samples per "epoch"
        repeats = max(16 // len(dataset), 1)
        effective_dataset = torch.utils.data.ConcatDataset([dataset] * repeats)
        print(f"Small dataset ({len(dataset)} samples) — "
              f"repeating {repeats}x = {len(effective_dataset)} effective samples")

    loader = torch.utils.data.DataLoader(
        effective_dataset,
        batch_size=min(8, len(effective_dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 5),
                                          gamma=0.5)
    criterion = nn.MSELoss()

    print(f"\nStarting training: {epochs} epochs, {len(effective_dataset)} samples/epoch")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            inp = batch["input"].to(device)
            target = batch["heatmaps"].to(device)

            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}  "
                  f"best={best_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.6f}")

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")

    # Export to TorchScript
    print(f"\nExporting TorchScript model to: {output_model}")
    model.eval()
    dummy = torch.randn(1, INPUT_CHANNELS, RESOLUTION, RESOLUTION)
    traced = torch.jit.trace(model, dummy)

    with torch.no_grad():
        out_orig = model(dummy)
        out_traced = traced(dummy)
        diff = (out_orig - out_traced).abs().max().item()
        print(f"Trace verification: max diff = {diff:.2e}")

    os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)
    traced.save(output_model)

    size_mb = os.path.getsize(output_model) / (1024 * 1024)
    print(f"Model saved: {output_model} ({size_mb:.1f} MB)")

    # Also save a checkpoint with config for later fine-tuning
    ckpt_path = output_model.replace(".pt", "_checkpoint.pth")
    torch.save({
        "model_state_dict": best_state,
        "config": {
            "model": {
                "num_joints": NUM_JOINTS,
                "input_channels": INPUT_CHANNELS,
                "base_channels": BASE_CHANNELS,
                "resolution": RESOLUTION,
            }
        },
        "best_loss": best_loss,
        "epochs": epochs,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train auto-rig model from captured rig editor data")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with *_color.png, *_meta.json files")
    parser.add_argument("--output_model", required=True,
                        help="Output TorchScript model path (.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training resolution")
    args = parser.parse_args()

    global RESOLUTION
    RESOLUTION = args.resolution

    print("=" * 60)
    print("  Auto-Rig Training Pipeline")
    print("=" * 60)
    print(f"  Data:       {args.data_dir}")
    print(f"  Output:     {args.output_model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Resolution: {RESOLUTION}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Convert captures to HDF5
    h5_path = os.path.join(os.path.dirname(args.data_dir), "captures_dataset.h5")
    print(f"\n[Step 1/2] Converting captures to HDF5: {h5_path}")
    if not convert_captures_to_h5(args.data_dir, h5_path, RESOLUTION):
        print("FAILED: Could not convert captures to HDF5")
        sys.exit(1)

    # Step 2: Train and export
    print(f"\n[Step 2/2] Training model...")
    if not train_model(h5_path, args.output_model, args.epochs,
                       args.lr, args.device):
        print("FAILED: Training failed")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
