#!/usr/bin/env python3
"""
verify_buildings_overlay.py — top-down placement audit for the PCG proxies.

Overlays the ACTUAL placed geometry from a <map>_pcg.glb (building footprints +
tree crowns) onto the albedo orthophoto at several ground scales, so you can
eyeball whether building/tree LOCATION and SIZE match the painted map — the same
check used to evaluate the heuristic vs the SAM2 per-instance extractor.

Usage (run from realworld/):
    python tools/terrain/verify_buildings_overlay.py \\
        --glb   content/terrain/<map>_pcg.glb \\
        --color content/terrain/<map>_color.png \\
        --out-dir content/terrain/.verify_overlays

Outputs one PNG per scale (100 m / 500 m / 1 km / 2 km / full map), buildings in
red outline, tree crowns in green. Centers the zoomed scales on the building
centroid; the full-map scale is centered on the map. WORLD_SIZE_M must match the
generator (4096 m).
"""
import argparse
import json
import struct

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
WORLD_SIZE_M = 4096.0


def parse_pcg_glb(path):
    """Return (b_corners [N,4,2] world XZ bottom-face, t_center [M,2], t_rad [M])."""
    data = open(path, "rb").read()
    assert data[:4] == b"glTF", "not a GLB"
    total = struct.unpack("<I", data[8:12])[0]
    off, G, BIN = 12, None, None
    while off < total:
        clen, ctype = struct.unpack("<II", data[off:off + 8]); off += 8
        chunk = data[off:off + clen]; off += clen
        if ctype == 0x4E4F534A:
            G = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            BIN = chunk
    accs, bvs, mats = G["accessors"], G["bufferViews"], G["materials"]

    def positions(ai):
        a = accs[ai]; bv = bvs[a["bufferView"]]
        o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(BIN, dtype="<f4", count=a["count"] * 3,
                             offset=o).reshape(a["count"], 3)

    b_corners, t_center, t_rad = [], [], []
    for prim in G["meshes"][0]["primitives"]:
        name = mats[prim["material"]]["name"]
        pos = positions(prim["attributes"]["POSITION"])
        if "building" in name:                 # 8 verts / box; bottom face = 0..3
            n = len(pos) // 8
            g = pos[:n * 8].reshape(n, 8, 3)
            b_corners.append(g[:, :4, :][:, :, [0, 2]])
        elif "tree" in name:                   # 7 verts / cone; apex=0, ring=1..6
            n = len(pos) // 7
            g = pos[:n * 7].reshape(n, 7, 3)
            c = g[:, 0, :][:, [0, 2]]
            r = np.linalg.norm(g[:, 1:, [0, 2]] - c[:, None, :], axis=2).mean(1)
            t_center.append(c); t_rad.append(r)
    b_corners = np.concatenate(b_corners) if b_corners else np.zeros((0, 4, 2))
    t_center = np.concatenate(t_center) if t_center else np.zeros((0, 2))
    t_rad = np.concatenate(t_rad) if t_rad else np.zeros((0,))
    return b_corners, t_center, t_rad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", required=True, help="<map>_pcg.glb")
    ap.add_argument("--color", required=True, help="<map>_color.png albedo")
    ap.add_argument("--out-dir", default="content/terrain/.verify_overlays")
    ap.add_argument("--scales", default="100,500,1000,2000,0",
                    help="ground extents in metres; 0 = full map")
    ap.add_argument("--disp", type=int, default=1200, help="output long side px")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    b_corners, t_center, t_rad = parse_pcg_glb(args.glb)
    # Polygon footprints sidecar (exact house outlines) supersedes glb boxes.
    sidecar = os.path.splitext(args.glb)[0] + "_footprints.json"
    b_polys = None
    if os.path.exists(sidecar):
        try:
            sj = json.load(open(sidecar, encoding="utf-8"))
            b_polys = [np.asarray(p, np.float32) for p in sj["polys"]]
            print(f"footprint sidecar: {len(b_polys)} polygons "
                  f"({os.path.basename(sidecar)})")
            b_corners = np.zeros((0, 4, 2))
        except Exception as e:                            # noqa: BLE001
            print(f"sidecar unreadable ({e}); glb boxes")
    good = (t_rad > 0.5) & (t_rad < 20.0)
    print(f"buildings {len(b_corners)}  trees {len(t_center)} "
          f"(radius-sane {int(good.sum())})")

    alb = Image.open(args.color).convert("RGB")
    H = alb.size[0]
    pxm = WORLD_SIZE_M / H

    def w2p(wx, wz):
        return (wx + WORLD_SIZE_M / 2) / pxm, (wz + WORLD_SIZE_M / 2) / pxm

    if b_polys:
        cs = np.array([[p[:, 0].mean(), p[:, 1].mean()] for p in b_polys])
        # sidecar is in PIXELS; convert to world for the shared framing math
        cx = float(cs[:, 0].mean()) * pxm - WORLD_SIZE_M / 2
        cz = float(cs[:, 1].mean()) * pxm - WORLD_SIZE_M / 2
    elif len(b_corners):
        bc = b_corners.mean(1)
        cx, cz = float(bc[:, 0].mean()), float(bc[:, 1].mean())
    else:
        cx, cz = 0.0, 0.0

    for ext in [int(x) for x in args.scales.split(",")]:
        full = (ext <= 0 or ext >= WORLD_SIZE_M)
        e = WORLD_SIZE_M if full else ext
        ccx, ccz = (0.0, 0.0) if full else (cx, cz)
        half = e / 2.0
        x0, y0 = w2p(ccx - half, ccz - half)
        x1, y1 = w2p(ccx + half, ccz + half)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(H, int(x1)), min(H, int(y1))
        crop = alb.crop((x0, y0, x1, y1)).convert("RGB")
        dr = ImageDraw.Draw(crop)
        for cc in b_corners:
            mx, mz = cc.mean(0)
            if abs(mx - ccx) > half * 1.1 or abs(mz - ccz) > half * 1.1:
                continue
            pts = [((wx + WORLD_SIZE_M / 2) / pxm - x0,
                    (wz + WORLD_SIZE_M / 2) / pxm - y0) for wx, wz in cc]
            dr.polygon(pts, outline=(255, 20, 20))
        if b_polys:
            for p in b_polys:
                mx = p[:, 0].mean() * pxm - WORLD_SIZE_M / 2
                mz = p[:, 1].mean() * pxm - WORLD_SIZE_M / 2
                if abs(mx - ccx) > half * 1.1 or abs(mz - ccz) > half * 1.1:
                    continue
                dr.polygon([(float(px - x0), float(py - y0))
                            for px, py in p], outline=(255, 20, 20))
        for (tx, tz), r, gd in zip(t_center, t_rad, good):
            if not gd or abs(tx - ccx) > half * 1.1 or abs(tz - ccz) > half * 1.1:
                continue
            px, py = (tx + WORLD_SIZE_M / 2) / pxm - x0, (tz + WORLD_SIZE_M / 2) / pxm - y0
            rp = max(1.0, r / pxm)
            dr.ellipse([px - rp, py - rp, px + rp, py + rp], outline=(20, 240, 60))
        sc = args.disp / max(crop.size)
        disp = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))))
        label = "full" if full else (f"{ext}m" if ext < 1000 else f"{ext // 1000}km")
        out = os.path.join(args.out_dir, f"overlay_{label}.png")
        disp.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
