"""
Lightweight multi-view rendering utilities for training data generation.

Uses trimesh + pyrender to render depth, normals, and silhouette from
multiple viewpoints around a mesh.  This is the Python equivalent of
the C++ SimpleRasterizer used in the engine.
"""

import math
import numpy as np

try:
    import trimesh
    import pyrender
    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False

# Standard humanoid joint names (must match rig_types.h).
STANDARD_JOINT_NAMES = [
    "hips", "spine", "chest", "neck", "head",
    "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
]

STANDARD_JOINT_PARENTS = [
    -1, 0, 1, 2, 3,     # hips→spine→chest→neck→head
     2, 5, 6, 7,         # chest→L_shoulder→L_upper→L_lower→L_hand
     2, 9, 10, 11,       # chest→R_shoulder→R_upper→R_lower→R_hand
     0, 13, 14,           # hips→L_upper_leg→L_lower_leg→L_foot
     0, 16, 17,           # hips→R_upper_leg→R_lower_leg→R_foot
]


def look_at(eye: np.ndarray, target: np.ndarray,
            up: np.ndarray = np.array([0, 1, 0], dtype=np.float32)
            ) -> np.ndarray:
    """Compute a 4x4 view (camera-to-world) matrix."""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-8)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(r, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def orbit_cameras(center: np.ndarray, radius: float, num_views: int,
                  elevation_deg: float = 15.0
                  ) -> list[dict]:
    """Generate orbit camera parameters (azimuth sweep).

    Returns
    -------
    cameras : list of dicts with keys:
        'view_matrix' (4,4), 'camera_pos' (3,), 'azimuth_deg', 'elevation_deg'
    """
    cameras = []
    elev = math.radians(elevation_deg)
    for i in range(num_views):
        azim = 2.0 * math.pi * i / num_views
        x = center[0] + radius * math.cos(elev) * math.sin(azim)
        y = center[1] + radius * math.sin(elev)
        z = center[2] + radius * math.cos(elev) * math.cos(azim)
        eye = np.array([x, y, z], dtype=np.float32)
        vm = look_at(eye, center)
        cameras.append({
            "view_matrix": vm,
            "camera_pos": eye,
            "azimuth_deg": math.degrees(azim),
            "elevation_deg": elevation_deg,
        })
    return cameras


def render_multiview(mesh: "trimesh.Trimesh", num_views: int = 8,
                     resolution: int = 256, elevation_deg: float = 15.0
                     ) -> list[dict]:
    """Render multi-view captures of a trimesh mesh.

    Each capture contains: color (H,W,3 uint8), depth (H,W float),
    normal (H,W,3 float), silhouette (H,W bool), and camera matrices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    num_views : int
    resolution : int
    elevation_deg : float

    Returns
    -------
    captures : list of dicts
    """
    if not HAS_PYRENDER:
        raise ImportError("pyrender is required for render_multiview()")

    bounds = mesh.bounds                           # (2, 3) — min, max
    center = (bounds[0] + bounds[1]) / 2.0
    extent = bounds[1] - bounds[0]
    max_ext = max(extent)
    radius = max_ext * 1.8

    cameras = orbit_cameras(center, radius, num_views, elevation_deg)

    # Build pyrender scene.
    py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(py_mesh)

    # Directional light.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, :3] = trimesh.transformations.euler_matrix(
        -0.3, 0.5, 0)[:3, :3]
    scene.add(light, pose=light_pose)

    # Offscreen renderer.
    renderer = pyrender.OffscreenRenderer(resolution, resolution)

    captures = []
    for cam_info in cameras:
        # Orthographic camera that fits the model.
        ortho_scale = max_ext * 0.65
        cam = pyrender.OrthographicCamera(xmag=ortho_scale, ymag=ortho_scale,
                                           znear=0.01, zfar=radius * 3)

        # Camera pose (camera-to-world).
        eye = cam_info["camera_pos"]
        fwd = center - eye
        fwd /= np.linalg.norm(fwd) + 1e-8
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, fwd)
        cam_pose = np.eye(4, dtype=np.float32)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -fwd
        cam_pose[:3, 3] = eye

        cam_node = scene.add(cam, pose=cam_pose)

        # Render.
        color, depth = renderer.render(scene,
                                       flags=pyrender.RenderFlags.RGBA)

        scene.remove_node(cam_node)

        # Silhouette from depth.
        silhouette = depth > 0

        # Approximate world normals from depth gradient.
        normals = _normals_from_depth(depth, silhouette)

        # Build projection matrix (orthographic).
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / ortho_scale
        proj[1, 1] = 1.0 / ortho_scale
        proj[2, 2] = -2.0 / (radius * 3 - 0.01)
        proj[2, 3] = -(radius * 3 + 0.01) / (radius * 3 - 0.01)
        proj[3, 3] = 1.0

        captures.append({
            "color": color[:, :, :3],                          # (H, W, 3) uint8
            "depth": depth.astype(np.float32),                 # (H, W)
            "normal": normals,                                 # (H, W, 3)
            "silhouette": silhouette.astype(np.float32),       # (H, W)
            "view_matrix": cam_info["view_matrix"],
            "proj_matrix": proj,
            "camera_pos": cam_info["camera_pos"],
            "azimuth_deg": cam_info["azimuth_deg"],
            "elevation_deg": cam_info["elevation_deg"],
            "cam_pose": cam_pose,
        })

    renderer.delete()
    return captures


def _normals_from_depth(depth: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
    """Estimate surface normals from a depth map via finite differences."""
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    # Central differences.
    dz_dx = np.zeros_like(depth)
    dz_dy = np.zeros_like(depth)
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) * 0.5
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) * 0.5

    normals[:, :, 0] = -dz_dx
    normals[:, :, 1] = -dz_dy
    normals[:, :, 2] = 1.0

    # Normalise.
    length = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True)) + 1e-8
    normals /= length

    # Zero out background.
    normals[~mask] = 0.0
    return normals


def pack_input_tensor(capture: dict) -> np.ndarray:
    """Pack a single capture into (7, H, W) float32 for the network.

    Channels: RGB(3) normalised [0,1] + Normal(3) [-1,1] + Silhouette(1) [0,1]
    """
    color = capture["color"].astype(np.float32) / 255.0          # (H,W,3)
    normal = capture["normal"].astype(np.float32)                 # (H,W,3)
    sil = capture["silhouette"].astype(np.float32)                # (H,W)

    # Stack into (H, W, 7) then transpose to (7, H, W).
    tensor = np.concatenate([
        color,
        normal,
        sil[:, :, None],
    ], axis=2).transpose(2, 0, 1)
    return tensor
