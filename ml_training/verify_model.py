"""
Verify trained model by feeding back the training data and comparing
predicted joint positions with ground truth.

Usage:
    python verify_model.py \
        --model_path ../realworld/assets/models/rig_diffusion_v006.pt \
        --data_dir  ../realworld/assets/rigs_training/scene-skinned
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. pip install Pillow")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. pip install torch")
    sys.exit(1)


NUM_JOINTS = 19
RESOLUTION = 256
JOINT_NAMES = [
    "hips", "spine", "chest", "neck", "head",
    "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
]

# Parent index for each joint (-1 = root). Must match rig_types.h
JOINT_PARENTS = [
    -1,  # hips
     0,  # spine -> hips
     1,  # chest -> spine
     2,  # neck -> chest
     3,  # head -> neck
     2,  # left_shoulder -> chest
     5,  # left_upper_arm -> left_shoulder
     6,  # left_lower_arm -> left_upper_arm
     7,  # left_hand -> left_lower_arm
     2,  # right_shoulder -> chest
     9,  # right_upper_arm -> right_shoulder
    10,  # right_lower_arm -> right_upper_arm
    11,  # right_hand -> right_lower_arm
     0,  # left_upper_leg -> hips
    13,  # left_lower_leg -> left_upper_leg
    14,  # left_foot -> left_lower_leg
     0,  # right_upper_leg -> hips
    16,  # right_lower_leg -> right_upper_leg
    17,  # right_foot -> right_lower_leg
]

# Colors: left side = blue, right side = red, center = green
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


try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    pass  # ImageDraw is part of Pillow


def _draw_skeleton_on_image(img: Image.Image, uvs: np.ndarray,
                            resolution: int, title: str,
                            label_all: bool = True):
    """Draw a single skeleton on an image. Returns the annotated image."""
    overlay = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.25)
    draw = ImageDraw.Draw(overlay)

    def uv_to_px(uv):
        return (int(uv[0] * resolution), int(uv[1] * resolution))

    # Draw bones (solid, colored)
    for j in range(min(len(uvs), NUM_JOINTS)):
        p = JOINT_PARENTS[j]
        if p >= 0 and p < len(uvs):
            pa = uv_to_px(uvs[p])
            pb = uv_to_px(uvs[j])
            name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else "joint"
            color = JOINT_COLORS.get(name, (255, 255, 0))
            draw.line([pa, pb], fill=color, width=2)

    # Draw joint dots (filled, colored) + labels
    for j in range(min(len(uvs), NUM_JOINTS)):
        px = uv_to_px(uvs[j])
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"j{j}"
        color = JOINT_COLORS.get(name, (255, 255, 0))
        draw.ellipse([px[0]-4, px[1]-4, px[0]+4, px[1]+4],
                      fill=color, outline=(255, 255, 255), width=1)
        if label_all:
            short = name.replace("left_", "L ").replace("right_", "R ")
            draw.text((px[0] + 6, px[1] - 6), short, fill=(255, 255, 255))

    # Title bar
    draw.rectangle([0, 0, resolution, 16], fill=(0, 0, 0))
    draw.text((4, 1), title, fill=(255, 255, 255))

    return overlay


def draw_skeleton_overlay(color_path: str, pred_uvs: np.ndarray,
                          gt_uvs: np.ndarray, resolution: int,
                          output_path: str):
    """Draw side-by-side: left = GT input, right = ML prediction."""
    img = Image.open(color_path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BILINEAR)

    gt_img = _draw_skeleton_on_image(img.copy(), gt_uvs, resolution,
                                     "GT (input labels)")
    pred_img = _draw_skeleton_on_image(img.copy(), pred_uvs, resolution,
                                       "ML (model output)")

    # Side by side
    combined = Image.new("RGB", (resolution * 2, resolution), (0, 0, 0))
    combined.paste(gt_img, (0, 0))
    combined.paste(pred_img, (resolution, 0))
    combined.save(output_path)


def make_grid_image(image_paths: list, cols: int = 5):
    """Combine multiple images into a grid."""
    if not image_paths:
        return
    images = [Image.open(p) for p in image_paths]
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows), (30, 30, 30))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))
    return grid


def load_view(meta_path: str, resolution: int = 256):
    """Load a single training view: build the 7-channel input + GT joint UVs."""
    with open(meta_path, "r") as f:
        meta = json.load(f)

    base = meta_path.replace("_meta.json", "")
    color_path = base + "_color.png"
    sil_path = base + "_silhouette.png"

    if not os.path.exists(color_path):
        return None, None

    # Load color
    color_img = Image.open(color_path).convert("RGB")
    color_img = color_img.resize((resolution, resolution), Image.BILINEAR)
    color_np = np.array(color_img, dtype=np.float32) / 255.0  # (H, W, 3)

    # Load silhouette
    if os.path.exists(sil_path):
        sil_img = Image.open(sil_path).convert("L")
        sil_img = sil_img.resize((resolution, resolution), Image.BILINEAR)
        sil_np = np.array(sil_img, dtype=np.float32) / 255.0
    else:
        sil_np = (color_np.sum(axis=2) > 0.01).astype(np.float32)

    # Build 7-channel input: RGB(3) + normals-as-color(3) + silhouette(1)
    rgb = color_np.transpose(2, 0, 1)
    normals = color_np.transpose(2, 0, 1)  # proxy
    sil = sil_np[np.newaxis, :, :]
    input_tensor = np.concatenate([rgb, normals, sil], axis=0)  # (7, H, W)

    # Ground truth joint UVs
    gt_uvs = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    joints = meta["joints"]
    for j, jdata in enumerate(joints):
        if j >= NUM_JOINTS:
            break
        u, v = jdata["uv"]
        gt_uvs[j, 0] = u  # already in [0,1]
        gt_uvs[j, 1] = v

    return input_tensor, gt_uvs


def predict_joints(model, input_tensor, device="cpu"):
    """Run model inference, return predicted joint UVs and confidences."""
    inp = torch.from_numpy(input_tensor).unsqueeze(0).to(device)  # (1, 7, H, W)

    with torch.no_grad():
        output = model(inp)

    if isinstance(output, tuple):
        heatmaps = output[0].cpu().numpy()[0]  # (J, H, W)
    else:
        heatmaps = output.cpu().numpy()[0]  # (J, H, W)

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


def main():
    parser = argparse.ArgumentParser(
        description="Verify trained model against training data")
    parser.add_argument("--model_path", required=True,
                        help="Path to TorchScript model (.pt)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with *_color.png, *_meta.json files")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="verify_output",
                        help="Directory to save overlay images")
    args = parser.parse_args()

    print("=" * 70)
    print("  Model Verification: Training Data Reconstruction")
    print("=" * 70)
    print(f"  Model:  {args.model_path}")
    print(f"  Data:   {args.data_dir}")
    print(f"  Res:    {args.resolution}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = torch.jit.load(args.model_path, map_location=args.device)
    model.eval()

    # Quick check: what does the model output?
    dummy = torch.randn(1, 7, args.resolution, args.resolution).to(args.device)
    with torch.no_grad():
        test_out = model(dummy)
    if isinstance(test_out, tuple):
        print(f"  Output format: tuple, heatmaps shape = {test_out[0].shape}")
    else:
        print(f"  Output format: tensor, shape = {test_out.shape}")

    # Find all training views
    meta_files = sorted(glob.glob(os.path.join(args.data_dir, "*_meta.json")))
    print(f"\nFound {len(meta_files)} training views\n")

    if not meta_files:
        print("ERROR: No meta files found!")
        sys.exit(1)

    all_errors = []
    all_confidences = []
    overlay_paths = []

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving overlay images to: {args.output_dir}/\n")

    for vi, meta_path in enumerate(meta_files):
        input_tensor, gt_uvs = load_view(meta_path, args.resolution)
        if input_tensor is None:
            continue

        pred_uvs, confs = predict_joints(model, input_tensor, args.device)

        # Compute per-joint Euclidean distance in pixels
        dx = (pred_uvs[:, 0] - gt_uvs[:, 0]) * args.resolution
        dy = (pred_uvs[:, 1] - gt_uvs[:, 1]) * args.resolution
        errors_px = np.sqrt(dx * dx + dy * dy)

        avg_err = errors_px.mean()
        max_err = errors_px.max()
        avg_conf = confs.mean()

        view_name = os.path.basename(meta_path).replace("_meta.json", "")
        print(f"View {vi:2d} ({view_name}):")
        print(f"  Avg error: {avg_err:6.2f} px   Max error: {max_err:6.2f} px   "
              f"Avg conf: {avg_conf:.4f}")

        # Per-joint details
        worst_joints = np.argsort(errors_px)[::-1][:5]
        for j in worst_joints:
            name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"joint_{j}"
            print(f"    {name:20s}: err={errors_px[j]:6.2f}px  "
                  f"pred=({pred_uvs[j,0]:.3f},{pred_uvs[j,1]:.3f})  "
                  f"gt=({gt_uvs[j,0]:.3f},{gt_uvs[j,1]:.3f})  "
                  f"conf={confs[j]:.4f}")
        print()

        # Save overlay image
        base = meta_path.replace("_meta.json", "")
        color_path = base + "_color.png"
        if os.path.exists(color_path):
            overlay_path = os.path.join(
                args.output_dir, f"{view_name}_overlay.png")
            draw_skeleton_overlay(
                color_path, pred_uvs, gt_uvs, args.resolution, overlay_path)
            overlay_paths.append(overlay_path)

        all_errors.append(errors_px)
        all_confidences.append(confs)

    # Build grid image of all views (each image is 2x wide, so use fewer cols)
    if overlay_paths:
        grid = make_grid_image(overlay_paths, cols=4)
        grid_path = os.path.join(args.output_dir, "grid_all_views.png")
        grid.save(grid_path)
        print(f"\nSaved grid image: {grid_path}")
        print(f"Saved {len(overlay_paths)} individual overlays to {args.output_dir}/")

    # Summary
    all_errors = np.concatenate(all_errors)
    all_confidences = np.concatenate(all_confidences)

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total joints evaluated: {len(all_errors)}")
    print(f"  Average error:     {all_errors.mean():6.2f} px  (at {args.resolution}x{args.resolution})")
    print(f"  Median error:      {np.median(all_errors):6.2f} px")
    print(f"  95th percentile:   {np.percentile(all_errors, 95):6.2f} px")
    print(f"  Max error:         {all_errors.max():6.2f} px")
    print(f"  Avg confidence:    {all_confidences.mean():.4f}")
    print(f"  Max confidence:    {all_confidences.max():.4f}")
    print()

    # Verdict
    if all_errors.mean() < 5.0:
        print("  VERDICT: Model reconstructs training data WELL (avg < 5px)")
        print("  -> The model has learned the training data. Issue is generalization.")
    elif all_errors.mean() < 15.0:
        print("  VERDICT: Model reconstructs training data PARTIALLY (avg 5-15px)")
        print("  -> Needs more epochs or training data refinement.")
    else:
        print("  VERDICT: Model FAILS to reconstruct training data (avg > 15px)")
        print("  -> Something is wrong with the training pipeline or data format.")
    print("=" * 70)


if __name__ == "__main__":
    main()
