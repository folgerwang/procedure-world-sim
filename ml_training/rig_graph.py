"""
rig_graph.py — node-graph rig format for the 3D auto-rig pipeline.

A rig is a fixed-topology graph: 19 humanoid joints (nodes) with a known
parent table (edges).  The same structure carries either 2D positions
(per-view labels) or 3D positions (training targets / model output).

JSON layout:
{
  "skeleton": "humanoid19",
  "dim": 3,                       # 2 for per-view, 3 for the lifted rig
  "norm": {"center": [...], "scale": s},   # how to map normalized<->model space
  "nodes": [
     {"id": 0, "name": "hips",  "parent": -1, "pos": [x, y, z]},
     ...
  ]
}

Normalization: 3D positions are stored model-space, but for ML we train on
*normalized* coordinates — centred on the hips and scaled by body height —
so the network learns translation/scale-invariant structure.  `norm` records
the mapping so predictions can be returned to model space:
    p_model = p_norm * scale + center
"""

from __future__ import annotations
import json
import numpy as np

# Must match rig_types.h getStandardJointNames()/getStandardJointParents().
JOINT_NAMES = [
    "hips", "spine", "chest", "neck", "head",
    "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
]
JOINT_PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 0, 16, 17]
NUM_JOINTS = len(JOINT_NAMES)
HIPS = 0

# Left/right pairs (for symmetry losses / regularization).
LR_PAIRS = [(5, 9), (6, 10), (7, 11), (8, 12), (13, 16), (14, 17), (15, 18)]
# Bone edges (child -> parent) for bone-length losses.
BONES = [(j, JOINT_PARENTS[j]) for j in range(NUM_JOINTS) if JOINT_PARENTS[j] >= 0]


def normalize(pos: np.ndarray):
    """pos: (19, 3) model-space -> (normalized (19,3), center(3), scale)."""
    pos = np.asarray(pos, dtype=np.float64)
    center = pos[HIPS].copy()
    # Scale = vertical extent of the skeleton (robust, non-zero).
    scale = float(pos[:, 1].max() - pos[:, 1].min())
    if scale < 1e-6:
        scale = float(np.linalg.norm(pos.max(0) - pos.min(0))) or 1.0
    return (pos - center) / scale, center, scale


def denormalize(pos_n: np.ndarray, center, scale) -> np.ndarray:
    return np.asarray(pos_n, dtype=np.float64) * scale + np.asarray(center)


def to_dict(pos: np.ndarray, norm=None) -> dict:
    pos = np.asarray(pos, dtype=float)
    dim = pos.shape[1]
    d = {"skeleton": "humanoid19", "dim": dim, "nodes": []}
    if norm is not None:
        center, scale = norm
        d["norm"] = {"center": list(map(float, center)), "scale": float(scale)}
    for j in range(NUM_JOINTS):
        d["nodes"].append({
            "id": j, "name": JOINT_NAMES[j], "parent": JOINT_PARENTS[j],
            "pos": [float(x) for x in pos[j]],
        })
    return d


def from_dict(d: dict) -> np.ndarray:
    dim = d.get("dim", 3)
    pos = np.zeros((NUM_JOINTS, dim), dtype=np.float64)
    for n in d["nodes"]:
        pos[n["id"]] = n["pos"]
    return pos


def save(path: str, pos: np.ndarray, norm=None):
    with open(path, "w") as f:
        json.dump(to_dict(pos, norm), f, indent=2)


def load(path: str) -> np.ndarray:
    with open(path) as f:
        return from_dict(json.load(f))


if __name__ == "__main__":
    # Round-trip self-test.
    rng = np.random.default_rng(0)
    p = rng.normal(size=(NUM_JOINTS, 3)) * np.array([0.3, 1.0, 0.2]) + np.array([0, 1, 0])
    pn, c, s = normalize(p)
    back = denormalize(pn, c, s)
    assert np.allclose(back, p, atol=1e-9), "normalize/denormalize round-trip failed"
    d = to_dict(p, norm=(c, s))
    p2 = from_dict(d)
    assert np.allclose(p2, p), "dict round-trip failed"
    assert len(BONES) == 18 and len(LR_PAIRS) == 7
    print("rig_graph self-test: PASS  (19 joints, 18 bones, 7 L/R pairs)")
    print("normalized hips:", pn[HIPS], " scale:", round(s, 4))
