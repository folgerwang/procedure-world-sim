"""
Evaluation metrics for joint prediction.

PCK  — Percentage of Correct Keypoints (2D, per-view)
MPJPE — Mean Per-Joint Position Error (3D, after back-projection)
"""

import numpy as np
import torch


def pck(pred_peaks: torch.Tensor, gt_peaks: torch.Tensor,
        threshold_frac: float = 0.2,
        image_size: int = 256) -> float:
    """Percentage of Correct Keypoints.

    A predicted joint is "correct" if its 2D distance to the ground truth
    peak is within `threshold_frac * image_size` pixels.

    Parameters
    ----------
    pred_peaks : (B, J, 2)  — predicted (col, row)
    gt_peaks : (B, J, 2)  — ground truth (col, row)
    threshold_frac : float
    image_size : int

    Returns
    -------
    pck_score : float in [0, 1]
    """
    dist = torch.norm(pred_peaks - gt_peaks, dim=2)       # (B, J)
    threshold = threshold_frac * image_size
    correct = (dist < threshold).float()
    return correct.mean().item()


def pck_per_joint(pred_peaks: torch.Tensor, gt_peaks: torch.Tensor,
                  threshold_frac: float = 0.2,
                  image_size: int = 256) -> torch.Tensor:
    """Per-joint PCK scores.

    Returns
    -------
    scores : (J,) float
    """
    dist = torch.norm(pred_peaks - gt_peaks, dim=2)       # (B, J)
    threshold = threshold_frac * image_size
    correct = (dist < threshold).float()
    return correct.mean(dim=0)                              # (J,)


def mpjpe(pred_3d: np.ndarray, gt_3d: np.ndarray) -> float:
    """Mean Per-Joint Position Error in 3D.

    Parameters
    ----------
    pred_3d : (J, 3)
    gt_3d : (J, 3)

    Returns
    -------
    error : float — mean L2 distance across joints
    """
    return float(np.mean(np.linalg.norm(pred_3d - gt_3d, axis=1)))


def mpjpe_batch(pred_3d: np.ndarray, gt_3d: np.ndarray) -> float:
    """Batched MPJPE.

    Parameters
    ----------
    pred_3d : (N, J, 3)
    gt_3d : (N, J, 3)

    Returns
    -------
    error : float
    """
    diffs = pred_3d - gt_3d
    per_joint = np.sqrt(np.sum(diffs ** 2, axis=2))        # (N, J)
    return float(np.mean(per_joint))


def procrustes_align(pred: np.ndarray,
                     gt: np.ndarray) -> np.ndarray:
    """Rigid Procrustes alignment (translation + rotation + uniform scale).

    Parameters
    ----------
    pred : (J, 3) — predicted joint positions
    gt : (J, 3)   — ground truth

    Returns
    -------
    aligned : (J, 3) — pred after rigid alignment to gt
    """
    mu_pred = pred.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    p = pred - mu_pred
    g = gt - mu_gt

    # Optimal rotation via SVD.
    H = p.T @ g
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    S[2, 2] = np.sign(d)
    R = Vt.T @ S @ U.T

    # Optimal scale.
    scale = np.trace(g.T @ (R @ p.T).T) / np.trace(p.T @ p)

    aligned = scale * (R @ p.T).T + mu_gt
    return aligned.astype(np.float32)


def pa_mpjpe(pred_3d: np.ndarray, gt_3d: np.ndarray) -> float:
    """Procrustes-Aligned MPJPE."""
    aligned = procrustes_align(pred_3d, gt_3d)
    return mpjpe(aligned, gt_3d)
