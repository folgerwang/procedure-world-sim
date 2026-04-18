"""
Export a trained checkpoint to TorchScript (.pt) or ONNX (.onnx) for
C++ inference in the RealWorld engine.

Usage:
    python export.py --checkpoint checkpoints/best_model.pth \
                     --output ../realworld/assets/models/rig_diffusion.pt \
                     --format torchscript

    python export.py --checkpoint checkpoints/best_model.pth \
                     --output ../realworld/assets/models/rig_diffusion.onnx \
                     --format onnx
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet_heatmap import UNetHeatmap, count_parameters


def export_torchscript(model: UNetHeatmap, output_path: str,
                       resolution: int = 256):
    """Export model via torch.jit.trace to a .pt file."""
    model.eval()
    dummy = torch.randn(1, model.head.in_channels + 6,  # actually 7 from enc1
                         resolution, resolution)
    # Build proper dummy input matching the actual input channels.
    in_ch = 7   # RGB + Normal + Silhouette
    dummy = torch.randn(1, in_ch, resolution, resolution)

    # Trace the model.
    traced = torch.jit.trace(model, dummy)

    # Verify.
    with torch.no_grad():
        out_orig = model(dummy)
        out_traced = traced(dummy)
        diff = (out_orig - out_traced).abs().max().item()
        print(f"Trace verification: max diff = {diff:.2e}")
        assert diff < 1e-5, f"Trace mismatch: {diff}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    traced.save(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported TorchScript model: {output_path} ({size_mb:.1f} MB)")


def export_onnx(model: UNetHeatmap, output_path: str,
                resolution: int = 256):
    """Export model to ONNX format."""
    model.eval()
    in_ch = 7
    dummy = torch.randn(1, in_ch, resolution, resolution)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "heatmaps": {0: "batch_size"},
        },
        opset_version=14,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported ONNX model: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--output", default=None,
                        help="Output path (.pt or .onnx)")
    parser.add_argument("--format", choices=["torchscript", "onnx"],
                        default="torchscript")
    parser.add_argument("--config", default=None,
                        help="Config YAML (optional, uses checkpoint's config)")
    args = parser.parse_args()

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", None)

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    if cfg is None:
        print("ERROR: No config found. Provide --config or use a checkpoint "
              "that includes config.")
        sys.exit(1)

    # Build model and load weights.
    model = UNetHeatmap(
        num_joints=cfg["model"]["num_joints"],
        in_channels=cfg["model"]["input_channels"],
        base_channels=cfg["model"]["base_channels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: {count_parameters(model):,} parameters")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")

    # Determine output path.
    output_path = args.output
    if output_path is None:
        ext = ".pt" if args.format == "torchscript" else ".onnx"
        output_path = cfg.get("export", {}).get(
            "output_path", f"rig_diffusion{ext}")

    resolution = cfg["model"]["resolution"]

    if args.format == "torchscript":
        export_torchscript(model, output_path, resolution)
    else:
        export_onnx(model, output_path, resolution)

    print("Done!")


if __name__ == "__main__":
    main()
