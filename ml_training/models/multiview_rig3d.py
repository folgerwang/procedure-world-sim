"""
multiview_rig3d.py — direct multi-view -> 3D node-graph regressor.

Takes the full set of orbit views for one character (each view = a few
geometry channels: grayscale + depth + silhouette) plus each view's camera
embedding, and outputs the 19 joint positions in normalized model space in
one shot.  No per-view 2D heatmaps, no triangulation — the network learns
cross-view correspondence and occlusion implicitly, which is exactly the
stage that was fragile in the geometric pipeline.

Interface mirrors the 2D models:
    build_model(num_joints=19, in_channels=3, num_views=8, device="cpu")
forward(images, cams) -> (B, num_joints, 3)
    images : (B, V, C, H, W)
    cams   : (B, V, cam_dim)   # per-view camera embedding (see camera_embed)
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:                       # allow import without torch (format/tests)
    _HAS_TORCH = False


CAM_DIM = 6   # [sin az, cos az, sin elev, cos elev, elev/90, 1]


def camera_embed(azimuth_deg, elevation_deg):
    """Numpy helper: per-view camera angles -> (CAM_DIM,) embedding."""
    a = np.radians(azimuth_deg); e = np.radians(elevation_deg)
    return np.array([np.sin(a), np.cos(a), np.sin(e), np.cos(e),
                     elevation_deg / 90.0, 1.0], dtype=np.float32)


if _HAS_TORCH:
    class ResBlock(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm2d(cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm2d(cout)
            self.sc = (nn.Sequential() if stride == 1 and cin == cout
                       else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout)))

        def forward(self, x):
            y = F.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return F.relu(y + self.sc(x))

    class ViewEncoder(nn.Module):
        """Shared CNN: (C,H,W) -> feature vector."""
        def __init__(self, in_channels, feat=256):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, 7, 2, 3, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
            self.layers = nn.Sequential(
                ResBlock(32, 64, 2), ResBlock(64, 64),
                ResBlock(64, 128, 2), ResBlock(128, 128),
                ResBlock(128, feat, 2), ResBlock(feat, feat))
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = self.layers(self.stem(x))
            return self.pool(x).flatten(1)            # (N, feat)

    class MultiViewRig3D(nn.Module):
        def __init__(self, num_joints=19, in_channels=3, num_views=8, feat=256):
            super().__init__()
            self.num_joints = num_joints
            self.encoder = ViewEncoder(in_channels, feat)
            self.cam_mlp = nn.Sequential(nn.Linear(CAM_DIM, 32), nn.ReLU(inplace=True))
            vfeat = feat + 32
            # Permutation-invariant fusion: mean + max over views.
            self.head = nn.Sequential(
                nn.Linear(2 * vfeat, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
                nn.Linear(256, num_joints * 3))

        def forward(self, images, cams):
            B, V, C, H, W = images.shape
            f = self.encoder(images.reshape(B * V, C, H, W)).reshape(B, V, -1)
            c = self.cam_mlp(cams.reshape(B * V, -1)).reshape(B, V, -1)
            vf = torch.cat([f, c], dim=-1)            # (B, V, vfeat)
            pooled = torch.cat([vf.mean(1), vf.max(1).values], dim=-1)
            out = self.head(pooled)                   # (B, J*3)
            return out.reshape(B, self.num_joints, 3)


def build_model(num_joints=19, in_channels=3, num_views=8, device="cpu", **kw):
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required to build the model.")
    m = MultiViewRig3D(num_joints=num_joints, in_channels=in_channels,
                       num_views=num_views)
    return m.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("torch not installed — printing shape contract only.")
        print("images (B,V,C,H,W)=(2,8,3,128,128), cams (B,V,6) -> out (2,19,3)")
        print("camera_embed(45,10):", camera_embed(45, 10))
    else:
        torch.manual_seed(0)
        m = build_model()
        img = torch.randn(2, 8, 3, 128, 128)
        cam = torch.randn(2, 8, CAM_DIM)
        out = m(img, cam)
        assert out.shape == (2, 19, 3), out.shape
        # overfit one batch to prove gradients flow
        tgt = torch.randn(2, 19, 3)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        l0 = None
        for _ in range(60):
            opt.zero_grad(); loss = ((m(img, cam) - tgt) ** 2).mean()
            loss.backward(); opt.step()
            l0 = l0 or loss.item()
        print(f"params={count_params(m):,}  loss {l0:.4f} -> {loss.item():.4f}",
              "PASS" if loss.item() < l0 * 0.5 else "CHECK")
