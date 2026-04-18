"""
Heatmap generation and peak extraction utilities.

Ground-truth heatmaps are isotropic 2D Gaussians centred on the projected
joint location.  At inference time we extract peaks via soft-argmax or
simple argmax.
"""

import numpy as np
import torch


def generate_heatmap(width: int, height: int, cx: float, cy: float,
                     sigma: float = 2.5) -> np.ndarray:
    """Generate a single 2D Gaussian heatmap.

    Parameters
    ----------
    width, height : int
        Heatmap resolution.
    cx, cy : float
        Centre of the Gaussian in **pixel** coordinates.
    sigma : float
        Standard deviation in pixels.

    Returns
    -------
    heatmap : (H, W) float32 in [0, 1]
    """
    xs = np.arange(width,  dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)   # (H, W) each
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    heatmap = np.exp(-d2 / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def generate_heatmaps(width: int, height: int,
                      joint_uvs: np.ndarray,
                      sigma: float = 2.5,
                      visibility: np.ndarray | None = None) -> np.ndarray:
    """Generate heatmaps for all joints.

    Parameters
    ----------
    width, height : int
        Heatmap resolution.
    joint_uvs : (J, 2) float
        Joint positions in pixel coords (col, row).
    sigma : float
        Gaussian sigma.
    visibility : (J,) bool or None
        If provided, invisible joints get zero heatmaps.

    Returns
    -------
    heatmaps : (J, H, W) float32
    """
    J = joint_uvs.shape[0]
    heatmaps = np.zeros((J, height, width), dtype=np.float32)
    for j in range(J):
        if visibility is not None and not visibility[j]:
            continue
        cx, cy = joint_uvs[j]
        # Skip joints that are way outside the image.
        if cx < -sigma * 4 or cx > width + sigma * 4:
            continue
        if cy < -sigma * 4 or cy > height + sigma * 4:
            continue
        heatmaps[j] = generate_heatmap(width, height, cx, cy, sigma)
    return heatmaps


def extract_peaks(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract peak (argmax) positions from predicted heatmaps.

    Parameters
    ----------
    heatmaps : (B, J, H, W)

    Returns
    -------
    peaks : (B, J, 2) float — (col, row) in pixel coordinates
    confidences : (B, J) float — peak values
    """
    B, J, H, W = heatmaps.shape
    flat = heatmaps.view(B, J, -1)
    confidences, indices = flat.max(dim=2)
    rows = indices // W
    cols = indices % W
    peaks = torch.stack([cols.float(), rows.float()], dim=2)
    return peaks, confidences


def soft_argmax(heatmaps: torch.Tensor,
                temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable soft-argmax for sub-pixel peak extraction.

    Parameters
    ----------
    heatmaps : (B, J, H, W)
    temperature : float
        Sharpness of the softmax; higher = closer to hard argmax.

    Returns
    -------
    peaks : (B, J, 2) float — (col, row) in pixel coordinates
    confidences : (B, J) float — max heatmap value
    """
    B, J, H, W = heatmaps.shape

    # Create coordinate grids.
    device = heatmaps.device
    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)

    # Softmax over spatial dimensions.
    flat = heatmaps.view(B, J, -1) * temperature
    weights = torch.softmax(flat, dim=2).view(B, J, H, W)

    # Weighted sum of coordinates.
    col = (weights.sum(dim=2) * xs.view(1, 1, W)).sum(dim=2)   # (B, J)
    row = (weights.sum(dim=3) * ys.view(1, 1, H)).sum(dim=2)   # (B, J)
    peaks = torch.stack([col, row], dim=2)

    confidences = heatmaps.view(B, J, -1).max(dim=2).values
    return peaks, confidences


def project_joints_to_2d(joint_positions_3d: np.ndarray,
                         view_matrix: np.ndarray,
                         proj_matrix: np.ndarray,
                         width: int, height: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D joint positions into 2D pixel coordinates.

    Parameters
    ----------
    joint_positions_3d : (J, 3) float
    view_matrix : (4, 4) float — world-to-camera
    proj_matrix : (4, 4) float — camera-to-clip
    width, height : int

    Returns
    -------
    uvs : (J, 2) float — pixel coordinates (col, row)
    visible : (J,) bool — True if in front of camera and inside image
    """
    J = joint_positions_3d.shape[0]
    ones = np.ones((J, 1), dtype=np.float32)
    pos_h = np.hstack([joint_positions_3d, ones])   # (J, 4)

    vp = proj_matrix @ view_matrix
    clip = (vp @ pos_h.T).T   # (J, 4)

    # Perspective divide.
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None]

    # NDC [-1, 1] → pixel coords.
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height   # flip Y
    uvs = np.stack([px, py], axis=1)

    # Visibility: in front of camera and inside image bounds (with margin).
    margin = 10
    visible = (
        (w > 0.01) &
        (px > -margin) & (px < width + margin) &
        (py > -margin) & (py < height + margin)
    )
    return uvs.astype(np.float32), visible
