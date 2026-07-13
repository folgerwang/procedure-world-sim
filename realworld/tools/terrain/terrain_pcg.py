#!/usr/bin/env python3
"""
terrain_pcg.py — segmentation-driven procedural content generation.

Pipeline stage 3 (after the colour orthophoto + derived heightmap):
    1. SEGMENT the albedo into classes.  `segment()` is the ML hook —
       currently colour heuristics + connected components (scipy); swap
       in SAM / a land-cover net behind the same mask interface later.
    2. EXTRACT instances: building footprints (oriented boxes via PCA),
       individual trees (small blobs + jittered grid sampling inside
       forest regions).  Roads/water stay terrain features (flattened /
       carved by the height pipeline), no meshes.
    3. PCG proxy meshes: WHITE untextured geometry — buildings are
       extruded boxes, trees are cones — all merged into ONE .glb placed
       in world coordinates on the terrain surface.  Import it from the
       Content Browser / place next to the terrain scene object.

Usage:
    python terrain_pcg.py --color <..._color.png> --height <....png>
        --out <..._pcg.glb>
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

WORLD_SIZE_M = 4096.0    # kTerrainMapMeters
HEIGHT_AMP_M = 250.0     # kTerrainHeightAmpMeters


# ── 1. Segmentation (ML hook) ───────────────────────────────────────────────
def segment(rgb_u8):
    """Return {class: float mask [H,W]} from the albedo.  Heuristic
    version; replace with a learned model behind the same interface."""
    a = rgb_u8.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    water = ((b > r * 1.15) & (b >= g * 0.90) & (g > r) & (mx > 0.15))
    roof = (((r > g * 1.25) & (r > b * 1.25) & (r > 0.30)) |
            ((sat < 0.25) & (mx < 0.35) & (mx > 0.15)))
    road = ((sat < 0.18) & (mx > 0.35) & (mx < 0.85)) & ~roof
    veg = ((g > r) & (g > b)) & ~water
    return {"water": water, "roof": roof & ~water,
            "road": road & ~water, "veg": veg}


# ── 2. Instance extraction ──────────────────────────────────────────────────
def _det_rand(x, y, lo, hi):
    h = (int(x) * 374761393 + int(y) * 668265263) & 0xFFFFFF
    return lo + (h / float(0x1000000)) * (hi - lo)


def extract_buildings(roof_mask, px_m):
    from scipy import ndimage as ndi
    m = ndi.binary_opening(roof_mask, np.ones((3, 3)))
    labels, n = ndi.label(m)
    out = []
    for sl in ndi.find_objects(labels):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels[sl] > 0)
        area_m2 = ys.size * px_m * px_m
        if area_m2 < 20.0 or area_m2 > 4000.0:
            continue
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        cx, cy = xs.mean(), ys.mean()
        # PCA orientation + extents.
        pts = np.stack([xs - cx, ys - cy], 1).astype(np.float32)
        cov = pts.T @ pts / max(1, pts.shape[0])
        evals, evecs = np.linalg.eigh(cov)
        ang = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
        proj = pts @ evecs
        ext = (proj.max(0) - proj.min(0)) * px_m
        w, d = max(3.0, float(ext[1])), max(3.0, float(ext[0]))
        hgt = _det_rand(cx, cy, 3.5, 9.0)
        out.append((cx, cy, w, d, ang, hgt))
    return out


def extract_trees(veg_mask, roof_mask, px_m, spacing_m=7.0,
                  max_trees=30000):
    """Jittered-grid sampling inside vegetation (deterministic)."""
    step = max(1, int(spacing_m / px_m))
    H, W = veg_mask.shape
    out = []
    for gy in range(0, H - step, step):
        for gx in range(0, W - step, step):
            jx = gx + int(_det_rand(gx, gy, 0, step))
            jy = gy + int(_det_rand(gy, gx, 0, step))
            jx, jy = min(jx, W - 1), min(jy, H - 1)
            if veg_mask[jy, jx] and not roof_mask[jy, jx]:
                out.append((jx, jy,
                            _det_rand(jx, jy, 3.0, 9.0),      # height
                            _det_rand(jy, jx, 1.2, 3.0)))     # radius
                if len(out) >= max_trees:
                    return out
    return out


# ── 3. PCG mesh assembly (merged, white) ────────────────────────────────────
class MeshAcc:
    def __init__(self):
        self.pos, self.nrm, self.idx = [], [], []

    def add(self, verts, normals, faces):
        base = len(self.pos)
        self.pos.extend(verts)
        self.nrm.extend(normals)
        self.idx.extend(base + f for f in faces)


def add_box(acc, cx, cz, w, d, ang, hgt, y0):
    ca, sa = np.cos(ang), np.sin(ang)
    hw, hd = w * 0.5, d * 0.5
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    pts = [(cx + x * ca - z * sa, cz + x * sa + z * ca) for x, z in corners]
    v, n, f = [], [], []
    for (x, z) in pts:                       # bottom 0-3, top 4-7
        v.append((x, y0, z))
    for (x, z) in pts:
        v.append((x, y0 + hgt, z))
    n = [(0, 1, 0)] * 8                      # proxy: uniform up normals
    quads = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
             (4, 5, 6, 7)]
    for a, b, c, d2 in quads:
        f += [a, b, c, a, c, d2]
    acc.add(v, n, f)


def add_cone(acc, cx, cz, hgt, rad, y0, sides=6):
    v = [(cx, y0 + hgt, cz)]
    n = [(0, 1, 0)]
    for i in range(sides):
        t = 2.0 * np.pi * i / sides
        v.append((cx + rad * np.cos(t), y0, cz + rad * np.sin(t)))
        n.append((np.cos(t), 0.3, np.sin(t)))
    f = []
    for i in range(sides):
        f += [0, 1 + i, 1 + (i + 1) % sides]
    acc.add(v, n, f)


def write_glb(acc, path):
    pos = np.asarray(acc.pos, np.float32)
    nrm = np.asarray(acc.nrm, np.float32)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
    idx = np.asarray(acc.idx, np.uint32)
    # Double-sided by construction: emit every triangle in BOTH windings.
    rev = idx.reshape(-1, 3)[:, ::-1].reshape(-1)
    idx = np.concatenate([idx, rev]).astype(np.uint32)

    # Planar world-XZ UVs (NOT zeros): the cluster pipeline computes
    # tangents from UV deltas — degenerate UVs break it.
    uv = np.stack([pos[:, 0] / WORLD_SIZE_M + 0.5,
                   pos[:, 2] / WORLD_SIZE_M + 0.5], 1).astype(np.float32)

    def pad4(bb, fill=b"\x00"):
        return bb + fill * ((4 - len(bb) % 4) % 4)

    # Split into <=65k-vertex primitives with UINT16 indices: standard
    # game assets never exceed 16-bit indexing, and a 255k-vert single
    # primitive with u32 indices is exactly the kind of input an engine
    # loader can silently mangle (drawn, but rasterizes nothing).
    prims = []
    tris = idx.reshape(-1, 3)
    tris = tris[np.argsort(tris.max(axis=1), kind="stable")]
    start = 0
    while start < len(tris):
        lo = int(tris[start].min())
        end = start
        while end < len(tris) and int(tris[end].max()) - lo < 65000:
            end += 1
        chunk = tris[start:end]
        vlo, vhi = int(chunk.min()), int(chunk.max()) + 1
        prims.append((pos[vlo:vhi], nrm[vlo:vhi], uv[vlo:vhi],
                      (chunk - vlo).astype(np.uint16).reshape(-1)))
        start = end

    views, accs, blobs, off = [], [], [], 0

    def view(blob, target):
        nonlocal off
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(blob)}
        if target:
            v["target"] = target
        views.append(v)
        blobs.append(blob)
        off += len(blob)
        return len(views) - 1

    prim_json = []
    for (p, n, t, ix) in prims:
        pv = view(pad4(p.tobytes()), 34962)
        accs.append({"bufferView": pv, "componentType": 5126,
                     "count": len(p), "type": "VEC3",
                     "min": p.min(0).tolist(), "max": p.max(0).tolist()})
        nv = view(pad4(n.tobytes()), 34962)
        accs.append({"bufferView": nv, "componentType": 5126,
                     "count": len(n), "type": "VEC3"})
        tv = view(pad4(t.tobytes()), 34962)
        accs.append({"bufferView": tv, "componentType": 5126,
                     "count": len(t), "type": "VEC2"})
        iv = view(pad4(ix.tobytes()), 34963)
        accs.append({"bufferView": iv, "componentType": 5123,   # u16
                     "count": int(ix.size), "type": "SCALAR"})
        prim_json.append({"attributes": {"POSITION": len(accs) - 4,
                                         "NORMAL": len(accs) - 3,
                                         "TEXCOORD_0": len(accs) - 2},
                          "indices": len(accs) - 1,
                          "material": 0, "mode": 4})

    # Tiny embedded WHITE albedo: the engine's drawable path expects a
    # baseColorTexture — factor-only materials import but don't render.
    import io
    wb = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 255, 255)).save(wb, format="PNG")
    img_view = view(pad4(wb.getvalue()), None)

    gltf = {"asset": {"version": "2.0", "generator": "terrain_pcg.py"},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "pcg_proxies"}],
            "meshes": [{"name": "pcg_proxies",
                        "primitives": prim_json}],
            "materials": [{"name": "proxy_white",
                           "pbrMetallicRoughness": {
                               "baseColorTexture": {"index": 0},
                               "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                               "metallicFactor": 0.0,
                               "roughnessFactor": 0.9}}],
            "images": [{"bufferView": img_view,
                        "mimeType": "image/png"}],
            "samplers": [{"magFilter": 9729, "minFilter": 9729,
                          "wrapS": 10497, "wrapT": 10497}],
            "textures": [{"sampler": 0, "source": 0}],
            "bufferViews": views, "accessors": accs}
    bin_chunk = b"".join(blobs)
    gltf["buffers"] = [{"byteLength": len(bin_chunk)}]
    js = pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2,
                            12 + 8 + len(js) + 8 + len(bin_chunk)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)


def build_pcg_glb(color_path, height_path, out_glb):
    rgb = np.asarray(Image.open(color_path).convert("RGB"))
    H = rgb.shape[0]
    px_m = WORLD_SIZE_M / H
    hmap = np.array(Image.open(height_path), dtype=np.float32) / 65535.0
    if hmap.shape[0] != H:
        hmap = np.array(Image.fromarray(hmap).resize((H, H),
                                                     Image.BILINEAR))
    masks = segment(rgb)

    def px_to_world(px, py):
        return (px * px_m - WORLD_SIZE_M * 0.5,
                py * px_m - WORLD_SIZE_M * 0.5)

    def ground(px, py):
        return float(hmap[min(int(py), H - 1),
                          min(int(px), H - 1)]) * HEIGHT_AMP_M

    acc = MeshAcc()
    bld = extract_buildings(masks["roof"], px_m)
    for cx, cy, w, d, ang, hgt in bld:
        wx, wz = px_to_world(cx, cy)
        add_box(acc, wx, wz, w, d, ang, hgt, ground(cx, cy) - 0.3)
    trees = extract_trees(masks["veg"], masks["roof"], px_m)
    for px, py, hgt, rad in trees:
        wx, wz = px_to_world(px, py)
        add_cone(acc, wx, wz, hgt, rad, ground(px, py) - 0.2)
    write_glb(acc, out_glb)
    print(f"[pcg] {len(bld)} buildings + {len(trees)} trees -> {out_glb} "
          f"({len(acc.pos)} verts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", required=True)
    ap.add_argument("--height", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_pcg_glb(args.color, args.height, args.out)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
