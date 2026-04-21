"""
Debug script: visualize what the training pipeline actually feeds to the model.

Loads the HDF5 dataset, applies augmentation N times, and saves side-by-side
images of (input_rgb | silhouette | heatmap_overlay) so you can verify that
the augmented heatmaps still align with the augmented input.

Usage:
    cd ml_training
    python debug_augmentation.py --h5 ../realworld/assets/rigs_training/captures_dataset.h5 --out debug_aug_output
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: pip install Pillow"); sys.exit(1)
try:
    import h5py
except ImportError:
    print("ERROR: pip install h5py"); sys.exit(1)
try:
    import torch
except ImportError:
    print("ERROR: pip install torch"); sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_from_captures import AugmentedViewDataset


def heatmap_peaks(hm_np):
    """Extract (x, y) peak positions from (J, H, W) heatmaps."""
    J, H, W = hm_np.shape
    peaks = []
    for j in range(J):
        ch = hm_np[j]
        idx = ch.argmax()
        y, x = divmod(int(idx), W)
        conf = float(ch[y, x])
        peaks.append((x, y, conf))
    return peaks


def render_sample(inp_np, hm_np, title=""):
    """Render a debug visualization of one augmented sample.

    inp_np: (7, H, W) float32 [0,1]
    hm_np:  (J, H, W) float32 [0,1]

    Returns: PIL Image with 3 panels side by side:
      [RGB input] [Silhouette] [RGB + heatmap overlay + joint peaks]
    """
    H, W = inp_np.shape[1], inp_np.shape[2]

    # Panel 1: RGB (channels 0-2)
    rgb = (inp_np[:3].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    panel_rgb = Image.fromarray(rgb)

    # Panel 2: Silhouette (channel 6)
    sil = (inp_np[6] * 255).clip(0, 255).astype(np.uint8)
    panel_sil = Image.fromarray(sil, mode='L').convert('RGB')

    # Panel 3: RGB + heatmap overlay + joint markers
    # Sum all joint heatmaps into one combined heat image
    combined_hm = hm_np.max(axis=0)  # (H, W)
    # Color the heatmap: hot colormap
    heat_r = (combined_hm * 255).clip(0, 255).astype(np.uint8)
    heat_g = ((combined_hm ** 0.5) * 128).clip(0, 255).astype(np.uint8)
    heat_b = np.zeros_like(heat_r)
    heat_rgb = np.stack([heat_r, heat_g, heat_b], axis=2)

    # Blend with RGB
    alpha = 0.5
    overlay = (rgb.astype(float) * (1 - alpha) + heat_rgb.astype(float) * alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)
    panel_overlay = Image.fromarray(overlay)

    # Draw joint peak markers
    draw = ImageDraw.Draw(panel_overlay)
    peaks = heatmap_peaks(hm_np)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255),
        (255, 64, 64), (64, 255, 64), (64, 64, 255), (128, 255, 0),
        (255, 0, 128), (0, 255, 128), (128, 128, 255), (255, 128, 128),
        (128, 255, 128), (200, 200, 0)
    ]
    for j, (px, py, conf) in enumerate(peaks):
        if conf < 0.01:
            continue  # skip invisible joints
        c = colors[j % len(colors)]
        r = 3
        draw.ellipse([px-r, py-r, px+r, py+r], fill=c, outline=(255, 255, 255))
        draw.text((px+4, py-4), f"{j}", fill=c)

    # Combine panels
    gap = 4
    total_w = W * 3 + gap * 2
    total_h = H + 20  # extra space for title
    canvas = Image.new('RGB', (total_w, total_h), (30, 30, 30))
    canvas.paste(panel_rgb, (0, 20))
    canvas.paste(panel_sil, (W + gap, 20))
    canvas.paste(panel_overlay, (W * 2 + gap * 2, 20))

    # Title
    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((4, 2), title, fill=(255, 255, 255))
    draw_canvas.text((W + gap + 4, 2), "Silhouette", fill=(200, 200, 200))
    draw_canvas.text((W * 2 + gap * 2 + 4, 2), "Heatmap overlay", fill=(200, 200, 200))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize augmented training samples")
    parser.add_argument("--h5", required=True, help="HDF5 dataset path")
    parser.add_argument("--out", default="debug_aug_output", help="Output directory")
    parser.add_argument("--n_raw", type=int, default=5, help="Number of raw samples to visualize")
    parser.add_argument("--n_aug", type=int, default=4, help="Augmented versions per raw sample")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load un-augmented dataset to show ground truth
    ds_clean = AugmentedViewDataset(args.h5, augment=False)
    ds_aug = AugmentedViewDataset(args.h5, augment=True)

    n_samples = min(args.n_raw, len(ds_clean))
    print(f"\nGenerating debug images for {n_samples} raw samples "
          f"x {args.n_aug} augmented versions each...")
    print(f"Output: {args.out}/\n")

    for i in range(n_samples):
        # Raw (un-augmented)
        sample_clean = ds_clean[i]
        inp_clean = sample_clean["input"].numpy()
        hm_clean = sample_clean["heatmaps"].numpy()

        img = render_sample(inp_clean, hm_clean, f"Raw sample {i} (no augmentation)")
        path = os.path.join(args.out, f"sample_{i:02d}_raw.png")
        img.save(path)
        print(f"  {path}")

        # Check heatmap stats
        peaks = heatmap_peaks(hm_clean)
        n_visible = sum(1 for _, _, c in peaks if c > 0.01)
        hm_max = hm_clean.max()
        hm_sum = hm_clean.sum()
        print(f"    Heatmap max={hm_max:.4f}, sum={hm_sum:.1f}, "
              f"visible joints={n_visible}/{len(peaks)}")

        # Augmented versions
        for a in range(args.n_aug):
            sample_aug = ds_aug[i]
            inp_aug = sample_aug["input"].numpy()
            hm_aug = sample_aug["heatmaps"].numpy()

            img = render_sample(inp_aug, hm_aug, f"Sample {i}, aug #{a}")
            path = os.path.join(args.out, f"sample_{i:02d}_aug{a}.png")
            img.save(path)

            peaks_aug = heatmap_peaks(hm_aug)
            n_vis_aug = sum(1 for _, _, c in peaks_aug if c > 0.01)
            hm_max_aug = hm_aug.max()
            print(f"  {path}  (hm_max={hm_max_aug:.4f}, visible={n_vis_aug})")

    print(f"\nDone! Check {args.out}/ to verify augmented inputs match heatmaps.")


if __name__ == "__main__":
    main()
