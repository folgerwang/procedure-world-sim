"""
Compare auto-rig inference captures with training images.

Loads dumped auto-rig captures, runs the model on each, draws predicted
skeleton overlays, and places them side-by-side with the nearest training
image (by azimuth) for visual comparison.

Usage:
    python compare_autorig.py \
        --model_path ../realworld/assets/models/rig_diffusion_v008.pt \
        --autorig_dir ../realworld/assets/debug_autorig_captures \
        --train_dir ../realworld/assets/rigs_training/scene-skinned \
        --output_dir compare_output
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: Pillow is required. pip install Pillow")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. pip install torch")
    sys.exit(1)


NUM_JOINTS = 19
JOINT_NAMES = [
    "hips", "spine", "chest", "neck", "head",
    "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
]

JOINT_PARENTS = [
    -1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 0, 16, 17,
]

JOINT_COLORS = {
    "hips": (0, 255, 0), "spine": (0, 255, 0), "chest": (0, 255, 0),
    "neck": (0, 255, 0), "head": (0, 255, 0),
    "left_shoulder": (100, 150, 255), "left_upper_arm": (100, 150, 255),
    "left_lower_arm": (80, 120, 255), "left_hand": (60, 100, 255),
    "right_shoulder": (255, 100, 100), "right_upper_arm": (255, 100, 100),
    "right_lower_arm": (255, 80, 80), "right_hand": (255, 60, 60),
    "left_upper_leg": (100, 150, 255), "left_lower_leg": (80, 120, 255),
    "left_foot": (60, 100, 255),
    "right_upper_leg": (255, 100, 100), "right_lower_leg": (255, 80, 80),
    "right_foot": (255, 60, 60),
}


def build_input_tensor(color_path, resolution=256):
    """Build the 7-channel input from a color PNG (same as training pipeline)."""
    color_img = Image.open(color_path).convert("RGB")
    color_img = color_img.resize((resolution, resolution), Image.BILINEAR)
    color_np = np.array(color_img, dtype=np.float32) / 255.0  # (H, W, 3)

    # Silhouette from non-black pixels
    sil_np = (color_np.sum(axis=2) > 0.01).astype(np.float32)

    # Try loading silhouette if it exists alongside color
    sil_path = color_path.replace("_color.png", "_silhouette.png")
    if os.path.exists(sil_path):
        sil_img = Image.open(sil_path).convert("L")
        sil_img = sil_img.resize((resolution, resolution), Image.BILINEAR)
        sil_np = np.array(sil_img, dtype=np.float32) / 255.0

    rgb = color_np.transpose(2, 0, 1)       # (3, H, W)
    normals = color_np.transpose(2, 0, 1)   # proxy
    sil = sil_np[np.newaxis, :, :]           # (1, H, W)
    return np.concatenate([rgb, normals, sil], axis=0)  # (7, H, W)


def predict_joints(model, input_tensor, device="cpu"):
    """Run model inference, return predicted joint UVs and confidences."""
    inp = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(inp)
    if isinstance(output, tuple):
        heatmaps = output[0].cpu().numpy()[0]
    else:
        heatmaps = output.cpu().numpy()[0]

    J, H, W = heatmaps.shape
    pred_uvs = np.zeros((J, 2), dtype=np.float32)
    confidences = np.zeros(J, dtype=np.float32)
    for j in range(J):
        hm = heatmaps[j]
        idx = np.argmax(hm)
        py, px = divmod(idx, W)
        pred_uvs[j, 0] = (px + 0.5) / W
        pred_uvs[j, 1] = (py + 0.5) / H
        confidences[j] = hm[py, px]
    return pred_uvs, confidences


def draw_skeleton(img, uvs, resolution, title, label=True):
    """Draw skeleton on image, return annotated copy."""
    overlay = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.2)
    draw = ImageDraw.Draw(overlay)

    def uv2px(uv):
        return (int(uv[0] * resolution), int(uv[1] * resolution))

    for j in range(min(len(uvs), NUM_JOINTS)):
        p = JOINT_PARENTS[j]
        if p >= 0 and p < len(uvs):
            name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else "joint"
            color = JOINT_COLORS.get(name, (255, 255, 0))
            draw.line([uv2px(uvs[p]), uv2px(uvs[j])], fill=color, width=2)

    for j in range(min(len(uvs), NUM_JOINTS)):
        px = uv2px(uvs[j])
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else "joint"
        color = JOINT_COLORS.get(name, (255, 255, 0))
        draw.ellipse([px[0]-4, px[1]-4, px[0]+4, px[1]+4],
                     fill=color, outline=(255, 255, 255), width=1)
        if label:
            short = name.replace("left_", "L ").replace("right_", "R ")
            draw.text((px[0]+6, px[1]-6), short, fill=(255, 255, 255))

    draw.rectangle([0, 0, resolution, 16], fill=(0, 0, 0))
    draw.text((4, 1), title, fill=(255, 255, 255))
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Compare auto-rig captures with training data")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--autorig_dir", required=True,
                        help="Directory with autorig_view*_color.png")
    parser.add_argument("--train_dir", required=True,
                        help="Directory with *_color.png + *_meta.json training data")
    parser.add_argument("--output_dir", default="compare_output")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    res = args.resolution

    # Load model
    print(f"Loading model: {args.model_path}")
    model = torch.jit.load(args.model_path, map_location=args.device)
    model.eval()

    # Find auto-rig captures
    ar_files = sorted(glob.glob(os.path.join(args.autorig_dir, "*_color.png")))
    print(f"Found {len(ar_files)} auto-rig captures in {args.autorig_dir}")

    # Find training images
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*_color.png")))
    print(f"Found {len(train_files)} training images in {args.train_dir}")

    # Load training GT joint UVs (for drawing GT skeleton on training images)
    train_meta = {}
    for tf in train_files:
        meta_path = tf.replace("_color.png", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            gt_uvs = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
            for j, jdata in enumerate(meta["joints"]):
                if j >= NUM_JOINTS:
                    break
                gt_uvs[j, 0] = jdata["uv"][0]
                gt_uvs[j, 1] = jdata["uv"][1]
            raw_az = meta.get("azimuth_deg", 0)
            norm_az = raw_az % 360  # normalize negative values like -430 → 290
            train_meta[tf] = {"gt_uvs": gt_uvs, "az": norm_az}

    all_panels = []

    for ai, ar_path in enumerate(ar_files):
        print(f"\n--- Auto-rig view {ai}: {os.path.basename(ar_path)} ---")

        # Run model on auto-rig image
        ar_tensor = build_input_tensor(ar_path, res)
        ar_pred_uvs, ar_confs = predict_joints(model, ar_tensor, args.device)
        avg_conf = ar_confs.mean()
        print(f"  Avg confidence: {avg_conf:.4f}")

        # Draw skeleton on auto-rig image
        ar_img = Image.open(ar_path).convert("RGB").resize((res, res), Image.BILINEAR)
        ar_overlay = draw_skeleton(ar_img, ar_pred_uvs, res,
                                   f"Auto-rig v{ai} (conf={avg_conf:.3f})")

        # Find closest training image by azimuth.
        # Auto-rig views are evenly spaced: view0=0°, view1=45°, etc.
        ar_azimuth = 360.0 * ai / len(ar_files)
        print(f"  Auto-rig azimuth: {ar_azimuth:.1f}°")

        best_train = None
        best_dist = 999
        for tf, tm in train_meta.items():
            d = abs(tm["az"] - ar_azimuth)
            d = min(d, 360 - d)  # wrap around
            if d < best_dist:
                best_dist = d
                best_train = tf

        if best_train is not None:
            # Run model on training image too
            tr_tensor = build_input_tensor(best_train, res)
            tr_pred_uvs, tr_confs = predict_joints(model, tr_tensor, args.device)

            tr_img = Image.open(best_train).convert("RGB").resize((res, res), Image.BILINEAR)
            gt_uvs = train_meta[best_train]["gt_uvs"]
            tr_az = train_meta[best_train]["az"]

            # Draw: GT skeleton + model prediction on training image
            tr_gt_overlay = draw_skeleton(tr_img.copy(), gt_uvs, res,
                                          f"Train GT az={tr_az:.0f}")
            tr_pred_overlay = draw_skeleton(tr_img.copy(), tr_pred_uvs, res,
                                            f"Train ML az={tr_az:.0f} (conf={tr_confs.mean():.3f})")

            # 3-panel: [training GT] [training ML pred] [auto-rig ML pred]
            panel = Image.new("RGB", (res * 3, res), (30, 30, 30))
            panel.paste(tr_gt_overlay, (0, 0))
            panel.paste(tr_pred_overlay, (res, 0))
            panel.paste(ar_overlay, (res * 2, 0))

            print(f"  Nearest training: az={tr_az:.0f} (dist={best_dist:.0f}°)")
            print(f"  Train ML conf: {tr_confs.mean():.4f}  Auto-rig conf: {avg_conf:.4f}")
        else:
            # No training match, just show auto-rig
            panel = Image.new("RGB", (res * 3, res), (30, 30, 30))
            panel.paste(ar_overlay, (res * 2, 0))

        panel_path = os.path.join(args.output_dir, f"compare_view{ai:02d}.png")
        panel.save(panel_path)
        all_panels.append(panel_path)

    # Build grid
    if all_panels:
        images = [Image.open(p) for p in all_panels]
        w, h = images[0].size
        cols = 2
        rows = (len(images) + cols - 1) // cols
        grid = Image.new("RGB", (w * cols, h * rows), (30, 30, 30))
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            grid.paste(img, (c * w, r * h))
        grid_path = os.path.join(args.output_dir, "comparison_grid.png")
        grid.save(grid_path)
        print(f"\nSaved grid: {grid_path}")
        print(f"Saved {len(all_panels)} comparison panels to {args.output_dir}/")


if __name__ == "__main__":
    main()
