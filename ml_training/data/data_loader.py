"""
PyTorch Dataset / DataLoader wrapping the HDF5 training data.

Each sample is one (model, view) pair:
    input:   (7, H, W) float32 — RGB + Normal + Silhouette
    target:  (J, H, W) float32 — ground-truth joint heatmaps
    meta:    dict with view_matrix, proj_matrix, joint_uvs, joint_visible
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import h5py
except ImportError:
    h5py = None


class JointHeatmapDataset(Dataset):
    """PyTorch dataset that reads (input, heatmap) pairs from HDF5.

    The HDF5 file is expected to have structure:
        /model_XXXX/view_YY/input     — (7, H, W)
        /model_XXXX/view_YY/heatmaps  — (J, H, W)
        /model_XXXX/view_YY/joint_uvs — (J, 2)
        /model_XXXX/view_YY/joint_visible — (J,)
        /model_XXXX/view_YY/view_matrix — (4, 4)
        /model_XXXX/view_YY/proj_matrix — (4, 4)
    """

    def __init__(self, h5_path: str, split: str = "train",
                 train_frac: float = 0.70, val_frac: float = 0.15,
                 augment: bool = False, aug_config: dict | None = None):
        """
        Parameters
        ----------
        h5_path : str
            Path to dataset.h5.
        split : str
            One of "train", "val", "test".
        train_frac, val_frac : float
            Split ratios (test_frac = 1 - train - val).
        augment : bool
            Whether to apply data augmentation (train only).
        aug_config : dict
            Augmentation parameters from config YAML.
        """
        if h5py is None:
            raise ImportError("h5py is required")

        self.h5_path = h5_path
        self.split = split
        self.augment = augment and (split == "train")
        self.aug_config = aug_config or {}
        self._h5 = None   # lazy-open per worker

        # Index all (model, view) pairs from the HDF5.
        with h5py.File(h5_path, "r") as hf:
            model_keys = sorted([k for k in hf.keys() if k.startswith("model_")])
            self.num_models = len(model_keys)
            self.resolution = int(hf.attrs.get("resolution", 256))
            self.num_joints = int(hf.attrs.get("num_joints", 19))

        # Deterministic split by model index.
        n = self.num_models
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if split == "train":
            model_range = range(0, n_train)
        elif split == "val":
            model_range = range(n_train, n_train + n_val)
        else:
            model_range = range(n_train + n_val, n)

        # Build flat index: list of (model_key, view_key).
        self.samples = []
        with h5py.File(h5_path, "r") as hf:
            for mi in model_range:
                mk = f"model_{mi:04d}"
                if mk not in hf:
                    continue
                view_keys = sorted([
                    k for k in hf[mk].keys() if k.startswith("view_")
                ])
                for vk in view_keys:
                    self.samples.append((mk, vk))

        print(f"[Dataset] {split}: {len(self.samples)} samples "
              f"from {len(model_range)} models")

    def __len__(self) -> int:
        return len(self.samples)

    def _open_h5(self):
        """Lazy-open HDF5 (safe for DataLoader workers)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx: int) -> dict:
        self._open_h5()
        mk, vk = self.samples[idx]
        vgrp = self._h5[mk][vk]

        inp = vgrp["input"][:].astype(np.float32)           # (7, H, W)
        heatmaps = vgrp["heatmaps"][:].astype(np.float32)   # (J, H, W)

        # Data augmentation.
        if self.augment:
            inp, heatmaps = self._augment(inp, heatmaps)

        return {
            "input": torch.from_numpy(inp),
            "heatmaps": torch.from_numpy(heatmaps),
            "model_key": mk,
            "view_key": vk,
        }

    def _augment(self, inp: np.ndarray,
                 heatmaps: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        cfg = self.aug_config

        # Horizontal flip (50% chance).
        if cfg.get("horizontal_flip", True) and np.random.random() < 0.5:
            inp = inp[:, :, ::-1].copy()
            heatmaps = heatmaps[:, :, ::-1].copy()
            # Swap left/right joint channels.
            heatmaps = self._swap_lr_joints(heatmaps)

        # Brightness/contrast on RGB channels (first 3).
        br = cfg.get("brightness_range", 0.2)
        if br > 0:
            factor = 1.0 + np.random.uniform(-br, br)
            inp[:3] = np.clip(inp[:3] * factor, 0, 1)

        # Gaussian noise.
        noise_std = cfg.get("gaussian_noise_std", 0.02)
        if noise_std > 0:
            noise = np.random.randn(*inp.shape).astype(np.float32) * noise_std
            inp = inp + noise

        # Depth dropout (channel index 6 is silhouette, depth is implicit
        # in the normal channel — optionally zero out patches).
        dropout_rate = cfg.get("depth_dropout_rate", 0.05)
        if dropout_rate > 0 and np.random.random() < dropout_rate:
            # Zero out silhouette channel for a random patch.
            h, w = inp.shape[1], inp.shape[2]
            ph = np.random.randint(h // 8, h // 3)
            pw = np.random.randint(w // 8, w // 3)
            py = np.random.randint(0, h - ph)
            px = np.random.randint(0, w - pw)
            inp[6, py:py+ph, px:px+pw] = 0   # silhouette

        return inp, heatmaps

    @staticmethod
    def _swap_lr_joints(heatmaps: np.ndarray) -> np.ndarray:
        """Swap left/right joint heatmap channels for horizontal flip."""
        # Indices based on STANDARD_JOINT_NAMES:
        # left_shoulder(5)↔right_shoulder(9),
        # left_upper_arm(6)↔right_upper_arm(10), etc.
        swap_pairs = [
            (5, 9), (6, 10), (7, 11), (8, 12),    # arms
            (13, 16), (14, 17), (15, 18),           # legs
        ]
        out = heatmaps.copy()
        for a, b in swap_pairs:
            out[a] = heatmaps[b]
            out[b] = heatmaps[a]
        return out


def create_dataloaders(h5_path: str, batch_size: int = 16,
                       num_workers: int = 4,
                       train_frac: float = 0.70,
                       val_frac: float = 0.15,
                       aug_config: dict | None = None
                       ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders."""
    train_ds = JointHeatmapDataset(
        h5_path, split="train", train_frac=train_frac, val_frac=val_frac,
        augment=True, aug_config=aug_config)
    val_ds = JointHeatmapDataset(
        h5_path, split="val", train_frac=train_frac, val_frac=val_frac,
        augment=False)
    test_ds = JointHeatmapDataset(
        h5_path, split="test", train_frac=train_frac, val_frac=val_frac,
        augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
