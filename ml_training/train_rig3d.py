"""
train_rig3d.py — train the multi-view -> 3D node-graph regressor.

Dataset layout (one dir per character):
    data_root/<sample>/<base>_view<N>_meta.json      (camera + view_proj)
    data_root/<sample>/<base>_view<N>_gray.png        (grayscale)   [optional]
    data_root/<sample>/<base>_view<N>_depth.png       (depth)       [optional]
    data_root/<sample>/<base>_view<N>_silhouette.png  (silhouette)
    data_root/<sample>/<base>_view<N>_color.png       (fallback -> gray)
    data_root/<sample>/rig3d.json                     (3D label; see build_3d_labels.py)

Input channels per view: [grayscale, depth, silhouette].
Target: 19x3 normalized joint positions (rig_graph.normalize).

Run a CPU smoke test (no data needed) with:  python train_rig3d.py --selftest
"""

from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig_graph as rg
from models.multiview_rig3d import camera_embed, CAM_DIM

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object


def _load_gray(base, res):
    """Grayscale channel: prefer *_gray.png, else BT.709 luminance of color."""
    g = base + "_gray.png"; c = base + "_color.png"
    if os.path.exists(g):
        im = Image.open(g).convert("L").resize((res, res))
        return np.asarray(im, np.float32) / 255.0
    if os.path.exists(c):
        im = np.asarray(Image.open(c).convert("RGB").resize((res, res)), np.float32) / 255.0
        return im @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.zeros((res, res), np.float32)


def _load_chan(path, res):
    if os.path.exists(path):
        return np.asarray(Image.open(path).convert("L").resize((res, res)), np.float32) / 255.0
    return np.zeros((res, res), np.float32)


class RigDataset(Dataset):
    def __init__(self, root, num_views=8, res=128):
        self.samples = [d for d in sorted(glob.glob(os.path.join(root, "*")))
                        if os.path.isfile(os.path.join(d, "rig3d.json"))]
        self.V, self.res = num_views, res

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        d = self.samples[i]
        metas = sorted(glob.glob(os.path.join(d, "*_meta.json")))[: self.V]
        imgs, cams = [], []
        for mp in metas:
            m = json.load(open(mp)); base = mp.replace("_meta.json", "")
            gray = _load_gray(base, self.res)
            depth = _load_chan(base + "_depth.png", self.res)
            sil = _load_chan(base + "_silhouette.png", self.res)
            imgs.append(np.stack([gray, depth, sil], 0))         # (3,H,W)
            cams.append(camera_embed(m.get("azimuth_deg", 0), m.get("elevation_deg", 0)))
        # pad/truncate to V views
        while len(imgs) < self.V:
            imgs.append(np.zeros_like(imgs[0])); cams.append(np.zeros(CAM_DIM, np.float32))
        pos = rg.load(os.path.join(d, "rig3d.json"))
        posn, _, _ = rg.normalize(pos)
        return (torch.tensor(np.stack(imgs), dtype=torch.float32),
                torch.tensor(np.stack(cams), dtype=torch.float32),
                torch.tensor(posn, dtype=torch.float32))


def rig_loss(pred, tgt):
    """Coordinate MSE + L/R-symmetric bone length + bone-length consistency."""
    mse = ((pred - tgt) ** 2).mean()
    # bone vectors
    child = torch.tensor([b[0] for b in rg.BONES])
    par = torch.tensor([b[1] for b in rg.BONES])
    bl_p = (pred[:, child] - pred[:, par]).norm(dim=-1)
    bl_t = (tgt[:, child] - tgt[:, par]).norm(dim=-1)
    bone = ((bl_p - bl_t) ** 2).mean()
    # left/right symmetry of bone lengths in the prediction
    name2bone = {c: k for k, (c, p) in enumerate(rg.BONES)}
    sym = pred.new_zeros(())
    for l, r in rg.LR_PAIRS:
        if l in name2bone and r in name2bone:
            sym = sym + (bl_p[:, name2bone[l]] - bl_p[:, name2bone[r]]).pow(2).mean()
    return mse + 0.5 * bone + 0.1 * sym, mse


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from models.multiview_rig3d import build_model
    model = build_model(num_joints=rg.NUM_JOINTS, in_channels=3,
                        num_views=args.num_views, device=dev)
    ds = RigDataset(args.data_root, args.num_views, args.res)
    if len(ds) == 0:
        print("No samples (need rig3d.json per dir). Run build_3d_labels.py first."); return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    print(f"[DEVICE] {dev}  samples={len(ds)}")
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for img, cam, tgt in dl:
            img, cam, tgt = img.to(dev), cam.to(dev), tgt.to(dev)
            opt.zero_grad(); loss, mse = rig_loss(model(img, cam), tgt)
            loss.backward(); opt.step(); tot += mse.item()
        print(f"epoch {ep+1}/{args.epochs}  coordMSE={tot/len(dl):.5f}")
    torch.save(model.state_dict(), args.out)
    # TorchScript export for the C++ runtime
    model.eval()
    ex_i = torch.randn(1, args.num_views, 3, args.res, args.res, device=dev)
    ex_c = torch.randn(1, args.num_views, CAM_DIM, device=dev)
    torch.jit.trace(model, (ex_i, ex_c)).save(args.out.replace(".pt", "_ts.pt"))
    print("saved", args.out)


def selftest():
    if not _HAS_TORCH:
        print("torch unavailable — selftest skipped (run where torch is installed).")
        return
    from models.multiview_rig3d import build_model
    torch.manual_seed(0)
    m = build_model(num_views=4)
    img = torch.randn(3, 4, 3, 64, 64); cam = torch.randn(3, 4, CAM_DIM)
    tgt = torch.randn(3, rg.NUM_JOINTS, 3)
    opt = torch.optim.Adam(m.parameters(), 1e-3); first = None
    for _ in range(80):
        opt.zero_grad(); loss, mse = rig_loss(m(img, cam), tgt)
        loss.backward(); opt.step(); first = first or mse.item()
    print(f"selftest coordMSE {first:.4f} -> {mse.item():.4f}",
          "PASS" if mse.item() < first * 0.5 else "CHECK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root"); ap.add_argument("--out", default="rig3d_model.pt")
    ap.add_argument("--num_views", type=int, default=8); ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.data_root: selftest()
    else: train(a)
