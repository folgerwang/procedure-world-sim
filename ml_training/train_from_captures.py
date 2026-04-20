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
HEATMAP_SIGMA = 8.0   # was 2.5 — too small, model learns to output near-zero


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

# Left/right joint swap pairs for horizontal flip augmentation.
# When the image is flipped, left_shoulder becomes right_shoulder, etc.
LR_SWAP_PAIRS = [
    (5, 9),    # left_shoulder <-> right_shoulder
    (6, 10),   # left_upper_arm <-> right_upper_arm
    (7, 11),   # left_lower_arm <-> right_lower_arm
    (8, 12),   # left_hand <-> right_hand
    (13, 16),  # left_upper_leg <-> right_upper_leg
    (14, 17),  # left_lower_leg <-> right_lower_leg
    (15, 18),  # left_foot <-> right_foot
]


class AugmentedViewDataset(torch.utils.data.Dataset):
    """HDF5 dataset with online augmentation for better generalization.

    Augmentations applied randomly each time a sample is fetched:
      - Horizontal flip (with left/right joint channel swap)
      - Random brightness / contrast / hue jitter on RGB channels
      - Random small affine: translate ±10%, scale 0.85–1.15
      - Gaussian noise on input
    """

    def __init__(self, h5_path: str, augment: bool = True):
        self.h5_path = h5_path
        self.augment = augment
        self.samples = []

        with h5py.File(h5_path, "r") as hf:
            for model_key in sorted(hf.keys()):
                for view_key in sorted(hf[model_key].keys()):
                    self.samples.append((model_key, view_key))

        print(f"Dataset: {len(self.samples)} samples from {h5_path}"
              f" (augmentation={'ON' if augment else 'OFF'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        model_key, view_key = self.samples[idx]
        with h5py.File(self.h5_path, "r") as hf:
            grp = hf[model_key][view_key]
            inp = torch.from_numpy(grp["input"][:])      # (7, H, W)
            hm = torch.from_numpy(grp["heatmaps"][:])    # (J, H, W)

        # ── Color → luminance filter (applied to BOTH train and eval) ──
        # The captured input has 7 channels: RGB(3) + normals-proxy(3) + sil(1).
        # The "normals" slot is actually a copy of RGB, so 6/7 channels are
        # pure color — the model gets biased toward pigment instead of shape.
        # We collapse RGB and normals-proxy to BT.709 luminance and broadcast
        # that across the 3 channels each. The C++ auto-rig inference path
        # (rig_diffusion_model.cpp::prepareInput) applies the identical
        # transform so train and infer see the same inputs.
        #
        # Silhouette (channel 6) is already shape info and is left alone.
        LUMA_W = torch.tensor([0.2126, 0.7152, 0.0722],
                              dtype=inp.dtype, device=inp.device).view(3, 1, 1)
        rgb_luma  = (inp[0:3] * LUMA_W).sum(dim=0, keepdim=True)   # (1,H,W)
        norm_luma = (inp[3:6] * LUMA_W).sum(dim=0, keepdim=True)
        inp[0:3] = rgb_luma.expand(3, -1, -1)
        inp[3:6] = norm_luma.expand(3, -1, -1)

        if not self.augment:
            return {"input": inp, "heatmaps": hm}

        # ── Horizontal flip (15% chance) ──
        # Keep low — too much flipping confuses left/right with few training views.
        if np.random.random() < 0.15:
            inp = inp.flip(2)        # flip W dimension
            hm = hm.flip(2)
            # Swap left/right joint heatmap channels
            for li, ri in LR_SWAP_PAIRS:
                hm[li], hm[ri] = hm[ri].clone(), hm[li].clone()

        # ── Luma jitter (applied after the grayscale filter above) ──
        # With color stripped out, the model only has brightness/contrast to
        # key on. Widen the jitter range so it learns to cope with varied
        # lighting — ±40% brightness + ±40% contrast.
        if np.random.random() < 0.9:
            gray = inp[:6]  # (6, H, W) — all grayscale after the luma filter
            brightness = 1.0 + (np.random.random() - 0.5) * 0.8   # ±40%
            gray = gray * brightness
            contrast = 1.0 + (np.random.random() - 0.5) * 0.8     # ±40%
            mean = gray.mean()
            gray = (gray - mean) * contrast + mean
            # Slight per-channel desync so the 2 groups of 3 identical channels
            # aren't pixel-perfect duplicates — acts as mild noise regularisation.
            for c in range(6):
                gray[c] = gray[c] * (0.92 + np.random.random() * 0.16)
            inp[:6] = gray.clamp(0.0, 1.0)

        # ── Random translate + scale (affine) ──
        if np.random.random() < 0.5:
            _, H, W = inp.shape
            # Random scale 0.85 – 1.15
            scale = 0.85 + np.random.random() * 0.3
            # Random translate ±10% of image size
            tx = (np.random.random() - 0.5) * 0.2 * W
            ty = (np.random.random() - 0.5) * 0.2 * H

            # Build affine grid for spatial transform
            theta = torch.tensor([
                [scale, 0, tx / (W / 2)],
                [0, scale, ty / (H / 2)]
            ], dtype=torch.float32).unsqueeze(0)

            grid = torch.nn.functional.affine_grid(
                theta, [1, 1, H, W], align_corners=False)

            # Apply to input (all 7 channels)
            inp_4d = inp.unsqueeze(0)
            inp = torch.nn.functional.grid_sample(
                inp_4d, grid, mode='bilinear',
                padding_mode='zeros', align_corners=False)[0]

            # Apply same transform to heatmaps
            hm_4d = hm.unsqueeze(0)
            hm = torch.nn.functional.grid_sample(
                hm_4d, grid, mode='bilinear',
                padding_mode='zeros', align_corners=False)[0]

        # ── Gaussian blur (key for generalization!) ──
        # Makes the model focus on body shape/silhouette, not pixel details.
        if np.random.random() < 0.7:
            _, H, W = inp.shape
            # Random kernel size (must be odd)
            k = np.random.choice([3, 5, 7, 9, 11])
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8 + np.random.random() * 2.0
            # Apply blur to RGB + normals (channels 0-5), keep silhouette sharp
            inp_blur = inp[:6].unsqueeze(0)  # (1, 6, H, W)
            padding = k // 2
            # Per-channel 2D gaussian blur via conv2d
            x = torch.arange(k, dtype=torch.float32) - k // 2
            kernel_1d = torch.exp(-x * x / (2 * sigma * sigma))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)  # (k, k)
            kernel_2d = kernel_2d.view(1, 1, k, k).repeat(6, 1, 1, 1)  # (6, 1, k, k)
            inp_blur = torch.nn.functional.conv2d(
                inp_blur, kernel_2d, padding=padding, groups=6)
            inp[:6] = inp_blur[0].clamp(0.0, 1.0)

        # ── Random erasing / cutout (forces learning from partial views) ──
        if np.random.random() < 0.3:
            _, H, W = inp.shape
            eh = int(H * (0.05 + np.random.random() * 0.15))
            ew = int(W * (0.05 + np.random.random() * 0.15))
            ey = np.random.randint(0, H - eh)
            ex = np.random.randint(0, W - ew)
            inp[:, ey:ey+eh, ex:ex+ew] = 0.0

        # ── Gaussian noise ──
        if np.random.random() < 0.3:
            noise = torch.randn_like(inp) * 0.03
            inp = (inp + noise).clamp(0.0, 1.0)

        return {"input": inp, "heatmaps": hm}


def train_model(h5_path: str, output_model: str, epochs: int = 50,
                lr: float = 1e-3, device: str = "auto"):
    """Train the U-Net from scratch on the captured data."""

    # ── Device selection with verbose diagnostics ──
    # The C++ plugin parses lines starting with "[DEVICE]" to display which
    # device training is actually running on.
    requested = device
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    cuda_built = torch.backends.cuda.is_built()  # was torch compiled with CUDA?

    if device == "auto":
        device = "cuda" if cuda_available else "cpu"

    print("-" * 60, flush=True)
    print(f"[DEVICE] requested={requested}  resolved={device}", flush=True)
    print(f"[DEVICE] torch={torch.__version__}  "
          f"cuda_built={cuda_built}  "
          f"cuda_available={cuda_available}  "
          f"device_count={cuda_count}", flush=True)
    if cuda_available:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        print(f"[DEVICE] gpu[{idx}]=\"{name}\"  compute_capability={cap[0]}.{cap[1]}  "
              f"cuda_runtime={torch.version.cuda}", flush=True)
    else:
        # Explain *why* CUDA isn't being used — helps diagnose "why is this on CPU?"
        if not cuda_built:
            reason = ("torch was built without CUDA (this is the CPU-only wheel). "
                      "Install the CUDA wheel from https://pytorch.org/get-started/locally/ "
                      "e.g. pip install torch --index-url "
                      "https://download.pytorch.org/whl/cu121")
        elif cuda_count == 0:
            reason = ("torch has CUDA support but no visible CUDA device. "
                      "Check nvidia-smi / driver install / CUDA_VISIBLE_DEVICES.")
        else:
            reason = "user explicitly requested CPU"
        print(f"[DEVICE] reason_no_cuda={reason}", flush=True)
    if device == "cuda" and requested != "cuda" and requested != "auto":
        print(f"[DEVICE] WARNING: requested {requested} but using {device}", flush=True)
    print("-" * 60, flush=True)

    # Build model
    model = build_model(
        num_joints=NUM_JOINTS,
        in_channels=INPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        device=device,
    )

    # Dataset with online augmentation — each fetch returns a randomly
    # augmented version, so the model sees different variations every epoch.
    dataset = AugmentedViewDataset(h5_path, augment=True)
    if len(dataset) == 0:
        print("ERROR: Dataset is empty, nothing to train on.")
        return False

    # With few samples, repeat so each epoch has enough gradient steps.
    # With augmentation, every repeat is a different random variation.
    samples_per_epoch = max(64, len(dataset))  # at least 64 samples/epoch
    repeats = max(samples_per_epoch // len(dataset), 1)
    effective_dataset = torch.utils.data.ConcatDataset([dataset] * repeats)
    print(f"Base samples: {len(dataset)}, repeats: {repeats}x, "
          f"effective: {len(effective_dataset)}/epoch (each augmented differently)")

    loader = torch.utils.data.DataLoader(
        effective_dataset,
        batch_size=min(8, len(effective_dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Weighted MSE: upweight pixels near joint peaks so the model can't
    # cheat by predicting all-zeros (which gives tiny MSE on sparse targets).
    PEAK_WEIGHT = 20.0   # pixels with target > 0.5 get this extra weight

    def weighted_mse_loss(pred, target):
        """MSE with extra weight on peak regions."""
        weight = torch.ones_like(target)
        weight[target > 0.1] = PEAK_WEIGHT * 0.5
        weight[target > 0.5] = PEAK_WEIGHT
        return (weight * (pred - target) ** 2).mean()

    print(f"\nStarting training: {epochs} epochs, {len(effective_dataset)} samples/epoch")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Heatmap sigma: {HEATMAP_SIGMA}, peak weight: {PEAK_WEIGHT}")
    print(f"[INPUT] runtime filter: RGB + normals-proxy -> BT.709 luminance "
          f"(forces shape/shading learning, not color)")
    print("-" * 60)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_t0 = time.time()

        for batch in loader:
            inp = batch["input"].to(device)
            target = batch["heatmaps"].to(device)

            optimizer.zero_grad()
            pred = model(inp)
            loss = weighted_mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Emit a per-epoch timing line on epoch 1 (and every 25 after) so the
        # C++ UI / log makes CPU-vs-GPU speed painfully obvious.
        if epoch == 1 or epoch % 25 == 0:
            epoch_secs = time.time() - epoch_t0
            print(f"[TIMING] epoch={epoch}  batches={n_batches}  "
                  f"seconds={epoch_secs:.2f}  "
                  f"sec_per_batch={epoch_secs / max(n_batches, 1):.3f}  "
                  f"device={device}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print every epoch so C++ progress bar updates smoothly
        print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}  "
              f"best={best_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}",
              flush=True)

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
