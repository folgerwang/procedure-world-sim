"""
Dataset preparation: load rigged 3D models, render multi-view captures,
generate ground-truth joint heatmaps, and write everything to HDF5.

Supports two data sources:
  1. RigNet-format models: rigged OBJ/glTF files with skeleton annotations
  2. Local glTF/GLB files with existing skins (extract joint positions)

Usage:
    python -m data.dataset_loader \
        --input_dir /path/to/rigged/models \
        --output_file datasets/dataset.h5 \
        --num_views 8 \
        --resolution 256
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import trimesh
except ImportError:
    trimesh = None

# Allow running as module from ml_training/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.heatmap_utils import (
    generate_heatmaps,
    project_joints_to_2d,
)
from utils.render_utils import (
    render_multiview,
    pack_input_tensor,
    STANDARD_JOINT_NAMES,
    STANDARD_JOINT_PARENTS,
)


# ── glTF / GLB skeleton extraction ──────────────────────────────────────────

def _load_gltf_skeleton(path: str) -> dict | None:
    """Extract skeleton joint positions from a rigged glTF/GLB.

    Returns dict with keys: joint_names (list[str]), joint_positions (J,3),
    joint_parents (J,), or None if no skin found.
    """
    try:
        import json as _json

        # Use trimesh to load the scene so we can inspect the gltf data.
        scene = trimesh.load(path, process=False)
        # trimesh doesn't directly expose glTF skins, so we parse the raw
        # glTF JSON ourselves.
        raw = _read_gltf_json(path)
        if raw is None or "skins" not in raw or len(raw["skins"]) == 0:
            return None

        skin = raw["skins"][0]
        joint_node_indices = skin.get("joints", [])
        if not joint_node_indices:
            return None

        nodes = raw.get("nodes", [])

        # Build node world transforms by walking the tree.
        node_world = [np.eye(4, dtype=np.float32) for _ in nodes]
        is_child = [False] * len(nodes)
        for n in nodes:
            for c in n.get("children", []):
                is_child[c] = True

        def walk(idx, parent_world):
            local = _node_local_matrix(nodes[idx])
            node_world[idx] = parent_world @ local
            for c in nodes[idx].get("children", []):
                walk(c, node_world[idx])

        for i in range(len(nodes)):
            if not is_child[i]:
                walk(i, np.eye(4, dtype=np.float32))

        # Extract joint world positions.
        joint_names = []
        joint_positions = []
        for ji in joint_node_indices:
            n = nodes[ji]
            joint_names.append(n.get("name", f"joint_{ji}"))
            pos = node_world[ji][:3, 3]
            joint_positions.append(pos)

        # Build parent indices within the skin's joint list.
        joint_set = set(joint_node_indices)
        idx_map = {ni: i for i, ni in enumerate(joint_node_indices)}
        joint_parents = []
        for ji in joint_node_indices:
            # Find parent: search which node has ji as child.
            parent_idx = -1
            for pi, n in enumerate(nodes):
                if ji in n.get("children", []) and pi in joint_set:
                    parent_idx = idx_map[pi]
                    break
            joint_parents.append(parent_idx)

        return {
            "joint_names": joint_names,
            "joint_positions": np.array(joint_positions, dtype=np.float32),
            "joint_parents": np.array(joint_parents, dtype=np.int32),
        }

    except Exception as e:
        print(f"  [WARN] Failed to extract skeleton from {path}: {e}")
        return None


def _read_gltf_json(path: str) -> dict | None:
    """Read the JSON portion of a .gltf or .glb file."""
    import struct

    path = str(path)
    if path.endswith(".gltf"):
        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".glb"):
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"glTF":
                return None
            version, length = struct.unpack("<II", f.read(8))
            # First chunk should be JSON.
            chunk_len, chunk_type = struct.unpack("<II", f.read(8))
            if chunk_type != 0x4E4F534A:  # "JSON"
                return None
            json_bytes = f.read(chunk_len)
            return json.loads(json_bytes.decode("utf-8"))
    return None


def _node_local_matrix(node: dict) -> np.ndarray:
    """Compute a node's local 4x4 transform from glTF TRS or matrix."""
    if "matrix" in node:
        m = np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T
        return m

    T = np.eye(4, dtype=np.float32)
    R = np.eye(4, dtype=np.float32)
    S = np.eye(4, dtype=np.float32)

    if "translation" in node:
        t = node["translation"]
        T[0, 3], T[1, 3], T[2, 3] = t[0], t[1], t[2]

    if "rotation" in node:
        q = node["rotation"]  # [x, y, z, w]
        R[:3, :3] = _quat_to_mat3(q[0], q[1], q[2], q[3])

    if "scale" in node:
        s = node["scale"]
        S[0, 0], S[1, 1], S[2, 2] = s[0], s[1], s[2]

    return T @ R @ S


def _quat_to_mat3(x, y, z, w) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    m = np.eye(3, dtype=np.float32)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y - z * w)
    m[0, 2] = 2 * (x * z + y * w)
    m[1, 0] = 2 * (x * y + z * w)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z - x * w)
    m[2, 0] = 2 * (x * z - y * w)
    m[2, 1] = 2 * (y * z + x * w)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


# ── Remap arbitrary skeletons to standard 19-joint set ──────────────────────

# Common name aliases that map to our standard joint names.
_JOINT_ALIASES = {
    "hips": "hips", "pelvis": "hips", "root": "hips",
    "spine": "spine", "spine1": "spine",
    "chest": "chest", "spine2": "chest", "upperspine": "chest",
    "neck": "neck", "neck1": "neck",
    "head": "head",
    "leftshoulder": "left_shoulder", "l_shoulder": "left_shoulder",
    "leftupperarm": "left_upper_arm", "leftarm": "left_upper_arm",
        "l_upperarm": "left_upper_arm",
    "leftlowerarm": "left_lower_arm", "leftforearm": "left_lower_arm",
        "l_forearm": "left_lower_arm",
    "lefthand": "left_hand", "l_hand": "left_hand",
    "rightshoulder": "right_shoulder", "r_shoulder": "right_shoulder",
    "rightupperarm": "right_upper_arm", "rightarm": "right_upper_arm",
        "r_upperarm": "right_upper_arm",
    "rightlowerarm": "right_lower_arm", "rightforearm": "right_lower_arm",
        "r_forearm": "right_lower_arm",
    "righthand": "right_hand", "r_hand": "right_hand",
    "leftupperleg": "left_upper_leg", "leftthigh": "left_upper_leg",
        "l_thigh": "left_upper_leg",
    "leftlowerleg": "left_lower_leg", "leftshin": "left_lower_leg",
        "leftleg": "left_lower_leg", "l_shin": "left_lower_leg",
    "leftfoot": "left_foot", "l_foot": "left_foot",
    "rightupperleg": "right_upper_leg", "rightthigh": "right_upper_leg",
        "r_thigh": "right_upper_leg",
    "rightlowerleg": "right_lower_leg", "rightshin": "right_lower_leg",
        "rightleg": "right_lower_leg", "r_shin": "right_lower_leg",
    "rightfoot": "right_foot", "r_foot": "right_foot",
}


def remap_skeleton(joint_names: list[str],
                   joint_positions: np.ndarray
                   ) -> tuple[np.ndarray, list[bool]]:
    """Map an arbitrary skeleton to the standard 19-joint layout.

    Returns
    -------
    positions : (19, 3) float — mapped positions (zeros for missing joints)
    found : list[bool] — which of the 19 standard joints were matched
    """
    # Normalize names for matching: lowercase, strip spaces/underscores.
    def normalize(name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "").replace("-", "")

    name_to_idx = {}
    for i, name in enumerate(joint_names):
        key = normalize(name)
        name_to_idx[key] = i

    positions = np.zeros((19, 3), dtype=np.float32)
    found = [False] * 19

    for si, std_name in enumerate(STANDARD_JOINT_NAMES):
        std_key = normalize(std_name)
        # Try exact match first, then aliases.
        idx = name_to_idx.get(std_key)
        if idx is None:
            # Search aliases.
            for alias, target in _JOINT_ALIASES.items():
                if target == std_name and alias in name_to_idx:
                    idx = name_to_idx[alias]
                    break
        if idx is not None:
            positions[si] = joint_positions[idx]
            found[si] = True

    return positions, found


# ── Main dataset builder ────────────────────────────────────────────────────

def find_rigged_models(input_dir: str) -> list[str]:
    """Recursively find .glb/.gltf files that might have skins."""
    models = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".glb", ".gltf"):
                models.append(os.path.join(root, f))
    models.sort()
    return models


def build_dataset(input_dir: str, output_file: str,
                  num_views: int = 8, resolution: int = 256,
                  heatmap_sigma: float = 2.5,
                  max_models: int = 0):
    """Build the training dataset.

    Scans input_dir for rigged glTF/GLB models, renders multi-view
    captures, generates ground-truth heatmaps, and writes to HDF5.
    """
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")
    if trimesh is None:
        raise ImportError("trimesh is required: pip install trimesh")

    model_paths = find_rigged_models(input_dir)
    if max_models > 0:
        model_paths = model_paths[:max_models]

    print(f"Found {len(model_paths)} model files in {input_dir}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    hf = h5py.File(output_file, "w")

    successful = 0
    for mi, path in enumerate(model_paths):
        print(f"\n[{mi+1}/{len(model_paths)}] {os.path.basename(path)}")

        # 1. Extract skeleton.
        skel = _load_gltf_skeleton(path)
        if skel is None:
            print("  → No skin/skeleton found, skipping.")
            continue

        # 2. Remap to standard 19 joints.
        std_positions, found = remap_skeleton(
            skel["joint_names"], skel["joint_positions"])
        num_found = sum(found)
        if num_found < 5:
            print(f"  → Only {num_found}/19 joints matched, skipping.")
            continue
        print(f"  → Mapped {num_found}/19 standard joints")

        # 3. Load mesh for rendering.
        try:
            mesh = trimesh.load(path, process=True, force="mesh")
            if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                print("  → Empty mesh, skipping.")
                continue
        except Exception as e:
            print(f"  → Failed to load mesh: {e}")
            continue

        # 4. Render multi-view captures.
        try:
            captures = render_multiview(mesh, num_views=num_views,
                                        resolution=resolution)
        except Exception as e:
            print(f"  → Render failed: {e}")
            continue

        # 5. Generate ground-truth heatmaps for each view.
        model_id = f"model_{successful:04d}"
        grp = hf.create_group(model_id)
        grp.attrs["source_path"] = path
        grp.attrs["num_joints_found"] = num_found

        grp.create_dataset("joint_positions", data=std_positions)
        grp.create_dataset("joint_found", data=np.array(found, dtype=bool))
        grp.create_dataset("joint_parents",
                           data=np.array(STANDARD_JOINT_PARENTS, dtype=np.int32))

        for vi, cap in enumerate(captures):
            # Project joints into this view.
            uvs, visible = project_joints_to_2d(
                std_positions,
                cap["view_matrix"],
                cap["proj_matrix"],
                resolution, resolution,
            )

            # Heatmaps only for found + visible joints.
            vis_mask = np.array(found) & visible
            heatmaps = generate_heatmaps(
                resolution, resolution, uvs,
                sigma=heatmap_sigma, visibility=vis_mask)

            # Pack input tensor (7, H, W).
            input_tensor = pack_input_tensor(cap)

            vgrp = grp.create_group(f"view_{vi:02d}")
            vgrp.create_dataset("input", data=input_tensor,
                                compression="gzip", compression_opts=4)
            vgrp.create_dataset("heatmaps", data=heatmaps,
                                compression="gzip", compression_opts=4)
            vgrp.create_dataset("joint_uvs", data=uvs)
            vgrp.create_dataset("joint_visible", data=vis_mask)
            vgrp.create_dataset("view_matrix", data=cap["view_matrix"])
            vgrp.create_dataset("proj_matrix", data=cap["proj_matrix"])
            vgrp.create_dataset("depth", data=cap["depth"],
                                compression="gzip", compression_opts=4)

        successful += 1
        print(f"  → Saved {num_views} views to {model_id}")

    hf.attrs["num_models"] = successful
    hf.attrs["num_views"] = num_views
    hf.attrs["resolution"] = resolution
    hf.attrs["num_joints"] = 19
    hf.close()

    print(f"\nDataset complete: {successful} models → {output_file}")


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build multi-view joint heatmap training dataset.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing rigged glTF/GLB models")
    parser.add_argument("--output_file", default="datasets/dataset.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--heatmap_sigma", type=float, default=2.5)
    parser.add_argument("--max_models", type=int, default=0,
                        help="Limit number of models (0 = all)")
    args = parser.parse_args()

    build_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        num_views=args.num_views,
        resolution=args.resolution,
        heatmap_sigma=args.heatmap_sigma,
        max_models=args.max_models,
    )


if __name__ == "__main__":
    main()
