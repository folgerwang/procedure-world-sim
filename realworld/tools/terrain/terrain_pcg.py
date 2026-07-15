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
# The RENDERED terrain surface is rock + a uniform soil layer
# (kSoilInitLevel in terrain/tile_creator.comp; the visible pass is
# SOIL_PASS).  Proxies placed at raw heightmap height end up ~5 m
# underground — buildings/trees fully buried.
SOIL_OFFSET_M = 5.0


# ── 0. Ground-truth masks from the layout-first pipeline ───────────────────
# terrain_from_text.py --layout writes <out>_seg.png: an indexed-palette PNG
# whose classes ARE the generation layout — no heuristics needed.  Keep this
# palette in sync with SEG_CLASSES in terrain_from_text.py.
SEG_CLASSES = [
    ("water",  ( 30,  60, 200)),
    ("grass",  (120, 200,  80)),
    ("forest", ( 20, 110,  40)),
    ("town",   (220,  60,  40)),
    ("road",   ( 70,  70,  70)),
    ("rock",   (150, 115,  85)),
    ("snow",   (245, 245, 245)),
    ("sand",   (215, 185, 130)),
]


def masks_from_seg(seg_path, H):
    """Exact class masks from the layout seg PNG (indexed palette, or
    nearest-palette classification for RGB copies).  Returns the same
    mask dict shape segment() produces, plus 'town'."""
    im = Image.open(seg_path)
    if im.mode == "P":
        if im.size != (H, H):
            im = im.resize((H, H), Image.NEAREST)
        cls = np.asarray(im, dtype=np.uint8)
    else:
        im = im.convert("RGB")
        if im.size != (H, H):
            im = im.resize((H, H), Image.NEAREST)
        a = np.asarray(im, dtype=np.int32)
        pal = np.array([c for _, c in SEG_CLASSES], dtype=np.int32)
        d = ((a[..., None, :] - pal[None, None]) ** 2).sum(-1)
        cls = d.argmin(-1).astype(np.uint8)
    names = [n for n, _ in SEG_CLASSES]
    idx = {n: i for i, n in enumerate(names)}
    return {"water": cls == idx["water"],
            "town": cls == idx["town"],
            "road": cls == idx["road"],
            "veg": cls == idx["forest"],
            # roof stays empty — buildings come from town-block
            # subdivision, not per-pixel roofs.
            "roof": np.zeros(cls.shape, bool)}


def buildings_from_town(town_mask, road_mask, px_m, max_buildings=20000):
    """Layout town blocks are coarse REGIONS, not per-house footprints —
    subdivide each block into house-sized oriented boxes on a jittered
    grid aligned to the block's principal axis (deterministic)."""
    from scipy import ndimage as ndi
    m = town_mask & ~road_mask
    labels, n = ndi.label(m)
    out = []
    for comp_i, sl in enumerate(ndi.find_objects(labels)):
        if sl is None:
            continue
        blk = labels[sl] == comp_i + 1
        ys, xs = np.nonzero(blk)
        if ys.size * px_m * px_m < 80.0:      # smaller than one house
            continue
        ox, oy = sl[1].start, sl[0].start
        mcx, mcy = xs.mean(), ys.mean()
        pts = np.stack([xs - mcx, ys - mcy], 1).astype(np.float32)
        cov = pts.T @ pts / max(1, len(pts))
        evals, evecs = np.linalg.eigh(cov)
        ang = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
        proj = pts @ evecs
        lo, hi = proj.min(0), proj.max(0)
        pitch = 22.0 / px_m                    # ~22 m street grid
        Hb, Wb = blk.shape
        u = lo[0]
        while u <= hi[0]:
            v = lo[1]
            while v <= hi[1]:
                ju = u + _det_rand(int(u * 7 + comp_i), int(v * 13),
                                   -5.0, 5.0) / px_m
                jv = v + _det_rand(int(v * 11), int(u * 5 + comp_i),
                                   -5.0, 5.0) / px_m
                # Rotated-frame → pixel coords.
                px = mcx + evecs[0, 0] * ju + evecs[0, 1] * jv
                py = mcy + evecs[1, 0] * ju + evecs[1, 1] * jv
                ix, iy = int(round(px)), int(round(py))
                if 0 <= ix < Wb and 0 <= iy < Hb and blk[iy, ix]:
                    gx, gy = ix + ox, iy + oy
                    w = _det_rand(gx, gy, 8.0, 16.0)
                    d = _det_rand(gy, gx, 6.0, 12.0)
                    hgt = _det_rand(gx + 1, gy + 1, 3.5, 9.0)
                    ja = ang + _det_rand(gx + 2, gy + 2, -0.08, 0.08)
                    out.append((float(gx), float(gy), w, d, ja, hgt))
                    if len(out) >= max_buildings:
                        return out
                v += pitch
            u += pitch
    return out


# ── 1. Segmentation (ML hook) ───────────────────────────────────────────────
def segment(rgb_u8, px_m=0.5):
    """Return {class: float mask [H,W]} from the albedo.  Heuristic
    version; replace with a learned model behind the same interface.
    Windows are specified in METRES (scaled by px_m) so the behaviour
    is resolution-independent.  Thresholds tuned against a real FLUX
    suburb map (terrain_1783978560) by overlay inspection."""
    from scipy import ndimage as ndi
    a = rgb_u8.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    sat = (mx - mn) / (mx + 1e-6)

    # Local (≈4.5 m) luminance variance and its wide (≈12 m) average:
    #   • loc_smooth  — roof interiors are locally smooth; tree canopy
    #     (even leafless brown forest, which fails the green-veg test)
    #     is high-texture everywhere → kills forest false positives.
    #   • big_smooth  — only WATER is smooth over a whole neighbourhood;
    #     roofs always have contrast edges within a few metres.  Also
    #     catches desaturated gray-blue rivers the colour test misses.
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    w_loc = max(3, int(4.5 / px_m))
    w_big = max(5, int(12.5 / px_m))
    m1 = ndi.uniform_filter(lum, w_loc)
    m2 = ndi.uniform_filter(lum * lum, w_loc)
    var = np.maximum(m2 - m1 * m1, 0.0)
    loc_smooth = var < 8.0e-4
    big_smooth = ndi.uniform_filter(var, w_big) < 1.5e-4

    water = ((((b > r * 1.15) & (b >= g * 0.90) & (g > r) & (mx > 0.15)) |
              ((sat < 0.22) & (mx < 0.50) & (b >= r * 0.95))) & big_smooth)
    # Three roof classes.  The gray/blue branches demand NEUTRAL-TO-COOL
    # colour (b >= r): warm gray-brown scrub/soil passes a plain
    # saturation test and used to blanket brushland with "buildings".
    roof_red = (r > g * 1.25) & (r > b * 1.25) & (r > 0.30)
    roof_gray = ((sat < 0.25) & (mx > 0.15) & (mx < 0.35) &
                 (b >= r * 0.95) & loc_smooth & ~big_smooth)
    roof_blue = ((b >= r) & (b > g * 0.95) & (sat < 0.45) &
                 (mx > 0.08) & (mx < 0.45) & loc_smooth & ~big_smooth)
    roof = roof_red | roof_gray | roof_blue
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
    # Closing (≈2 m) consolidates a roof broken up by chimneys/ridge
    # lines into one component; opening (≈3 m) then removes speckle and
    # cuts thin bridges between adjacent houses.
    cl = max(3, int(round(2.0 / px_m)) | 1)
    op = max(3, int(round(3.0 / px_m)) | 1)
    m = ndi.binary_closing(roof_mask, np.ones((cl, cl)))
    m = ndi.binary_opening(m, np.ones((op, op)))
    labels, n = ndi.label(m)
    out = []
    for sl in ndi.find_objects(labels):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels[sl] > 0)
        area_m2 = ys.size * px_m * px_m
        # 40 m² floor kills residual speckle; 12000 m² ceiling still
        # admits merged row-house blocks (one big proxy box is better
        # than a hole in the middle of a town).
        if area_m2 < 40.0 or area_m2 > 12000.0:
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
        ext = (proj.max(0) - proj.min(0) + 1) * px_m
        w, d = max(3.0, float(ext[1])), max(3.0, float(ext[0]))

        if area_m2 <= 600.0:
            # Single house: the box IS the painted footprint (PCA
            # oriented extents), so proxy size maps the map's house
            # base.  SHAPE priors: roofs are compact near-rectangles —
            # ragged/ribbon components (scrub, shadows, tree lines)
            # have low fill ratio or extreme aspect.
            fill = area_m2 / max(w * d, 1e-3)
            aspect = max(w, d) / max(min(w, d), 1e-3)
            if fill < 0.45 or aspect > 5.0:
                continue
            hgt = _det_rand(cx, cy, 3.5, 9.0)
            out.append((cx, cy, w, d, ang, hgt))
            continue

        # MERGED block (row houses / dense quarter fused into one
        # component): one giant slab wouldn't match any painted house.
        # Subdivide in the block's own rotated frame into house-sized
        # cells (~16 m); each occupied cell emits a box sized by ITS
        # OWN pixel extent — every proxy hugs the painted mass under
        # it.
        cell = max(1.0, 16.0 / px_m)
        u0, v0 = proj[:, 0].min(), proj[:, 1].min()
        ui = np.floor((proj[:, 0] - u0) / cell).astype(np.int32)
        vi = np.floor((proj[:, 1] - v0) / cell).astype(np.int32)
        key = ui.astype(np.int64) * 100000 + vi
        for k in np.unique(key):
            sel = key == k
            sub_area = int(sel.sum()) * px_m * px_m
            if sub_area < 40.0:        # cell only grazes the mask
                continue
            sub = proj[sel]
            sext = (sub.max(0) - sub.min(0) + 1) * px_m
            sw = float(np.clip(sext[1], 5.0, 18.0))
            sd = float(np.clip(sext[0], 4.0, 14.0))
            ccx = cx + float(pts[sel, 0].mean())
            ccy = cy + float(pts[sel, 1].mean())
            hgt = _det_rand(ccx, ccy, 3.5, 9.0)
            out.append((ccx, ccy, sw, sd, ang, hgt))
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
    # Over budget → thin EVENLY across the whole map.  Truncating the
    # row-major scan instead would give the map's top half all the trees
    # and leave southern forests bare (a real artifact once the canopy
    # detector widened).
    if len(out) > max_trees:
        stride = len(out) / float(max_trees)
        out = [out[int(i * stride)] for i in range(max_trees)]
        print(f"[pcg] tree budget: thinned to {max_trees} evenly")
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


def write_glb(groups, path):
    """groups: [(MeshAcc, (r, g, b) 0-255, name), ...] — ONE MATERIAL PER
    GROUP so each PCG class renders in its own debug colour (buildings vs
    trees) and mis-segmented blobs are identifiable at a glance."""
    import io

    def pad4(bb, fill=b"\x00"):
        return bb + fill * ((4 - len(bb) % 4) % 4)

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

    prim_json, materials, images, textures = [], [], [], []
    for (acc, rgb, name) in groups:
        if not acc.pos:
            continue
        pos = np.asarray(acc.pos, np.float32)
        nrm = np.asarray(acc.nrm, np.float32)
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
        idx = np.asarray(acc.idx, np.uint32)
        # Double-sided by construction: emit every triangle in BOTH
        # windings.
        rev = idx.reshape(-1, 3)[:, ::-1].reshape(-1)
        idx = np.concatenate([idx, rev]).astype(np.uint32)

        # Planar world-XZ UVs (NOT zeros): the cluster pipeline computes
        # tangents from UV deltas — degenerate UVs break it.
        uv = np.stack([pos[:, 0] / WORLD_SIZE_M + 0.5,
                       pos[:, 2] / WORLD_SIZE_M + 0.5], 1).astype(np.float32)

        # Tiny embedded solid-colour albedo: the engine's drawable path
        # expects a baseColorTexture — factor-only materials import but
        # don't render.
        mat_idx = len(materials)
        wb = io.BytesIO()
        Image.new("RGB", (4, 4), tuple(rgb)).save(wb, format="PNG")
        img_view = view(pad4(wb.getvalue()), None)
        images.append({"bufferView": img_view, "mimeType": "image/png"})
        textures.append({"sampler": 0, "source": len(images) - 1})
        materials.append({"name": name,
                          "pbrMetallicRoughness": {
                              "baseColorTexture": {
                                  "index": len(textures) - 1},
                              "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                              "metallicFactor": 0.0,
                              "roughnessFactor": 0.9}})

        # Split into <=65k-vertex primitives with UINT16 indices: standard
        # game assets never exceed 16-bit indexing, and a 255k-vert single
        # primitive with u32 indices is exactly the kind of input an
        # engine loader can silently mangle (drawn, rasterizes nothing).
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
            p, n, t = pos[vlo:vhi], nrm[vlo:vhi], uv[vlo:vhi]
            ix = (chunk - vlo).astype(np.uint16).reshape(-1)
            pv = view(pad4(p.tobytes()), 34962)
            accs.append({"bufferView": pv, "componentType": 5126,
                         "count": len(p), "type": "VEC3",
                         "min": p.min(0).tolist(),
                         "max": p.max(0).tolist()})
            nv = view(pad4(n.tobytes()), 34962)
            accs.append({"bufferView": nv, "componentType": 5126,
                         "count": len(n), "type": "VEC3"})
            tv = view(pad4(t.tobytes()), 34962)
            accs.append({"bufferView": tv, "componentType": 5126,
                         "count": len(t), "type": "VEC2"})
            iv = view(pad4(ix.tobytes()), 34963)
            accs.append({"bufferView": iv, "componentType": 5123,  # u16
                         "count": int(ix.size), "type": "SCALAR"})
            prim_json.append({"attributes": {"POSITION": len(accs) - 4,
                                             "NORMAL": len(accs) - 3,
                                             "TEXCOORD_0": len(accs) - 2},
                              "indices": len(accs) - 1,
                              "material": mat_idx, "mode": 4})
            start = end

    gltf = {"asset": {"version": "2.0", "generator": "terrain_pcg.py"},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "pcg_proxies"}],
            "meshes": [{"name": "pcg_proxies",
                        "primitives": prim_json}],
            "materials": materials,
            "images": images,
            "samplers": [{"magFilter": 9729, "minFilter": 9729,
                          "wrapS": 10497, "wrapT": 10497}],
            "textures": textures,
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


def build_pcg_glb(color_path, height_path, out_glb, seg_path=None):
    rgb = np.asarray(Image.open(color_path).convert("RGB"))
    H = rgb.shape[0]
    px_m = WORLD_SIZE_M / H
    hmap = np.array(Image.open(height_path), dtype=np.float32) / 65535.0
    if hmap.shape[0] != H:
        hmap = np.array(Image.fromarray(hmap).resize((H, H),
                                                     Image.BILINEAR))

    # GROUND-TRUTH masks from the layout-first pipeline when available;
    # colour heuristics otherwise.
    use_seg = bool(seg_path) and os.path.exists(seg_path)
    if use_seg:
        masks = masks_from_seg(seg_path, H)
        print(f"[pcg] using layout masks: {seg_path}")
        # SAFETY NET: a stylistic FLUX layout (antique-map browns) can
        # quantize to an EMPTY town class — which silently produced a
        # map with zero buildings.  The albedo usually still paints the
        # towns (red roofs), so fall back to the tuned colour
        # heuristics rather than emitting nothing.
        if masks["town"].mean() < 0.0005:
            print("[pcg] layout town class is EMPTY (<0.05%) — layout "
                  "quantization likely failed; falling back to colour-"
                  "heuristic segmentation of the albedo")
            use_seg = False
            masks = segment(rgb, px_m)
    else:
        masks = segment(rgb, px_m)

    def px_to_world(px, py):
        return (px * px_m - WORLD_SIZE_M * 0.5,
                py * px_m - WORLD_SIZE_M * 0.5)

    def ground(px, py):
        return float(hmap[min(int(py), H - 1),
                          min(int(px), H - 1)]) * HEIGHT_AMP_M + SOIL_OFFSET_M

    if use_seg:
        from scipy import ndimage as ndi
        # The FLUX albedo pass is SOFT-conditioned on the layout: it
        # repaints towns with its own micro-layout (streets + roofs)
        # that drifts from the mask grid.  Boxes placed on the layout
        # grid therefore don't line up with the houses PAINTED on the
        # ground.  So: use the layout as the WHERE (no false positives
        # outside towns), but trace the individual footprints FROM THE
        # ALBEDO so proxies match the visible texture.
        heur = segment(rgb, px_m)
        town_r = ndi.uniform_filter(
            masks["town"].astype(np.float32),
            size=max(3, int(80.0 / px_m) | 1)) > 1e-6   # ~±40 m slack
        roof = (heur["roof"] & town_r &
                ~masks["water"] & ~masks["road"])
        bld = extract_buildings(roof, px_m)

        # Fallback: SMALL town components whose painted-roof trace came
        # up empty (albedo forgot a village) get the grid subdivision so
        # no town renders empty.  LARGE unpainted components are the
        # albedo saying "no town here" (e.g. FLUX drawing a giant red
        # region the albedo then renders as bare rock) — grid-filling
        # those would carpet empty terrain with thousands of boxes, so
        # trust the albedo and skip them.
        labels, ncomp = ndi.label(masks["town"])
        if ncomp > 0:
            index = np.arange(1, ncomp + 1)
            comp_mean = np.atleast_1d(
                ndi.mean(roof.astype(np.float32), labels, index))
            comp_area = np.atleast_1d(
                ndi.sum(masks["town"], labels, index)) * px_m * px_m
            # The ALBEDO is ground truth for what the player sees: a town
            # component with (near-)zero painted roofs gets NO boxes —
            # grid-filling it puts proxies on visibly empty land, which
            # reads as a bug in-engine.  (The old grid-fill fallback for
            # small unpainted components did exactly that.)  Components
            # with at least a few painted roofs keep their traced boxes.
            unpainted = int((comp_mean < 0.02).sum())
            if unpainted:
                print(f"[pcg] {unpainted} town region(s) unpainted in "
                      "albedo -> skipped (albedo is ground truth; no "
                      "boxes on visibly empty land)")

        # TREES from the PAINTED CANOPY, not just the layout forest
        # class: FLUX layouts draw forests as scattered tree ICONS
        # (a few % cover) while the albedo paints dense canopy carpets
        # — placing trees only on the icons leaves painted forests
        # empty.  Dark-green albedo pixels are canopy; light green is
        # lawn/grassland and stays treeless.
        af = rgb.astype(np.float32) / 255.0
        # Two-tier canopy detection.  Tier 1 (dense/dark forest) is the
        # old strict test.  Tier 2 catches the LIGHTER, hazier greens
        # FLUX paints for distant / dry forests — green-dominant by a
        # small margin, some saturation (excludes gray rock), and below
        # lawn brightness.  Painted forests previously failed tier 1
        # wholesale and rendered treeless.
        mx = af.max(-1)
        mn = af.min(-1)
        canopy_dark = ((af[..., 1] > af[..., 0] * 1.05) &
                       (af[..., 1] > af[..., 2] * 1.05) &
                       (mx < 0.50))
        canopy_lite = ((af[..., 1] > af[..., 0] * 1.02) &
                       (af[..., 1] > af[..., 2] * 1.02) &
                       ((mx - mn) > 0.06) &
                       (mx < 0.65))
        canopy = canopy_dark | canopy_lite
        print(f"[pcg] canopy cover: dark {canopy_dark.mean()*100:.1f}%  "
              f"light {canopy_lite.mean()*100:.1f}%  "
              f"(veg class {masks['veg'].mean()*100:.1f}%)")
        masks["veg"] = masks["veg"] | canopy
        tree_block = masks["town"] | masks["road"] | masks["water"]
    else:
        # ── Building placement priors (beyond per-pixel colour) ─────────
        from scipy import ndimage as ndi

        # 1. SLOPE gate: buildings sit on near-flat ground.  Gray rock
        #    faces pass the colour heuristic but live on steep terrain —
        #    reject roof pixels where the heightmap slope exceeds ~11°
        #    (rise/run 0.20).
        gy, gx = np.gradient(hmap * HEIGHT_AMP_M,
                             WORLD_SIZE_M / hmap.shape[0])
        flat = np.hypot(gx, gy) < 0.20

        # 2. TOWN-CLUSTER gate: buildings exist in settlements, not as
        #    lone blobs.  Local roof-pixel density over a ~120 m window
        #    must exceed 6% or the candidate is dropped.
        # 3. FOREST gate: candidates embedded in dense (green) vegetation
        #    are canopy shadows, not houses.
        roof = masks["roof"] & flat
        dens_win = max(3, int(120.0 / px_m))
        dens = ndi.uniform_filter(roof.astype(np.float32), size=dens_win)
        veg_d = ndi.uniform_filter(masks["veg"].astype(np.float32),
                                   size=dens_win)
        roof &= (dens > 0.06) & (veg_d < 0.60)
        bld = extract_buildings(roof, px_m)
        tree_block = masks["roof"]

    # Per-class accumulators — written as separate primitives with their
    # own DEBUG COLOUR material, so a glance at the rendered proxies
    # tells you what the segmentation classified each blob as:
    #   buildings = orange, trees = green.
    acc_bld, acc_tree = MeshAcc(), MeshAcc()
    for cx, cy, w, d, ang, hgt in bld:
        wx, wz = px_to_world(cx, cy)
        add_box(acc_bld, wx, wz, w, d, ang, hgt, ground(cx, cy) - 0.3)
    trees = extract_trees(masks["veg"], tree_block, px_m)
    for px, py, hgt, rad in trees:
        wx, wz = px_to_world(px, py)
        add_cone(acc_tree, wx, wz, hgt, rad, ground(px, py) - 0.2)
    write_glb([(acc_bld,  (255,  96,  32), "proxy_building"),
               (acc_tree, ( 48, 208,  80), "proxy_tree")],
              out_glb)
    print(f"[pcg] {len(bld)} buildings (orange) + {len(trees)} trees "
          f"(green) [{'layout masks' if use_seg else 'color heuristics'}]"
          f" -> {out_glb} "
          f"({len(acc_bld.pos) + len(acc_tree.pos)} verts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", required=True)
    ap.add_argument("--height", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default=None,
                    help="layout seg PNG (<map>_seg.png) — exact class "
                         "masks from the layout-first pipeline; when "
                         "given, colour heuristics are bypassed")
    args = ap.parse_args()
    build_pcg_glb(args.color, args.height, args.out, seg_path=args.seg)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
# (touched to force workspace-mount resync 2)
