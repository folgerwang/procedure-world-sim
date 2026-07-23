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


def extract_buildings_shaped(roof_mask, px_m):
    """Per-building oriented footprints that TRACE the painted roofs instead
    of tiling a uniform grid: each connected roof blob is fit with a min-area
    rotated rectangle, and large merged blocks are split by a distance-transform
    watershed into house-sized pieces, each fit with its own oriented box. So a
    proxy's shape/orientation follows the roof under it. Falls back to the grid
    extractor when OpenCV is unavailable (no hard cv2 dependency)."""
    try:
        import cv2
    except Exception:
        return extract_buildings(roof_mask, px_m)
    from scipy.ndimage import maximum_filter, label as _ndlabel
    m = roof_mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []

    def emit(region):
        cnts, _ = cv2.findContours(region.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) * px_m * px_m < 30.0:
                continue
            (bx, by), (bw, bh), ang = cv2.minAreaRect(c)
            w, d = bw * px_m, bh * px_m
            if w < 3.0 or d < 3.0:
                continue
            if max(w, d) / max(min(w, d), 1e-3) > 6.0:   # ribbon / tree-line
                continue
            hgt = _det_rand(bx, by, 3.5, 9.0)
            out.append((float(bx), float(by),
                        max(3.0, w), max(3.0, d),
                        float(np.deg2rad(ang)), hgt))

    edge_px = max(3.0, 3.0 / px_m)      # ~3 m in from a wall = interior seed
    win = max(5, int(round(6.0 / px_m)) | 1)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] * px_m * px_m
        if area < 30.0:
            continue
        comp = (lab == i)
        if area <= 250.0:               # single house: one oriented box
            emit(comp)
            continue
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        peak = (dt == maximum_filter(dt, size=win)) & (dt > edge_px)
        mk, nmk = _ndlabel(peak)
        if nmk < 2:                     # can't split -> trace the block outline
            emit(comp)
            continue
        ws = cv2.watershed(
            cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            mk.astype(np.int32).copy())
        for lbl in range(1, nmk + 1):
            emit((ws == lbl) & comp)
    return out


def _sam2_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def _build_sam2_predictor(cfg):
    """Build a SAM2 image predictor.  Prefers a LOCAL checkpoint (offline) in
    assets/ml_models/ or SAM2_CHECKPOINT; otherwise auto-downloads weights from
    Hugging Face via SAM2ImagePredictor.from_pretrained (SAM2_HF_MODEL, default
    facebook/sam2.1-hiera-small).  Raises on any problem so the caller falls
    back to the heuristic extractor."""
    import os
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    ckpt = cfg.get("checkpoint") or os.environ.get("SAM2_CHECKPOINT")
    model_cfg = cfg.get("config") or os.environ.get("SAM2_CONFIG")
    if not ckpt:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
        mdir = os.path.join(root, "assets", "ml_models")
        for name in ("sam2.1_hiera_small.pt", "sam2.1_hiera_tiny.pt",
                     "sam2_hiera_small.pt", "sam2_hiera_tiny.pt"):
            p = os.path.join(mdir, name)
            if os.path.exists(p):
                ckpt = p
                break
    if ckpt and os.path.exists(ckpt):
        if not model_cfg:
            base = os.path.basename(ckpt).lower()
            if "tiny" in base:    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif "small" in base: model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "large" in base: model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            else:                 model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        import torch
        from sam2.build_sam import build_sam2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = build_sam2(model_cfg, ckpt, device=device)
        print(f"[pcg] SAM2 loaded (local {os.path.basename(ckpt)}) on {device}")
        return SAM2ImagePredictor(sam)
    # No local weights -> Hugging Face auto-download.
    hf = cfg.get("hf_model") or os.environ.get(
        "SAM2_HF_MODEL", "facebook/sam2.1-hiera-small")
    pred = SAM2ImagePredictor.from_pretrained(hf)
    print(f"[pcg] SAM2 loaded (hf {hf})")
    return pred

def _dedup_boxes(boxes, px_m, iou_thr=0.45):
    """Cross-tile oriented-box dedup: grid-hash centers, drop a box when its
    rotated-rect intersection with an already-kept neighbour exceeds iou_thr
    of the smaller box.  Keeps larger boxes first (they trace whole roofs)."""
    import cv2
    if len(boxes) < 2:
        return boxes
    order = sorted(range(len(boxes)),
                   key=lambda i: -(boxes[i][2] * boxes[i][3]))
    cell = 40.0 / px_m                       # 40 m neighbourhood
    grid, kept = {}, []
    for i in order:
        cx, cy, w, d, ang, hgt = boxes[i]
        r1 = ((cx, cy), (w / px_m, d / px_m), np.rad2deg(ang))
        gx, gy = int(cx / cell), int(cy / cell)
        dup = False
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for j in grid.get((nx, ny), ()):
                    c2 = kept[j]
                    r2 = ((c2[0], c2[1]),
                          (c2[2] / px_m, c2[3] / px_m), np.rad2deg(c2[4]))
                    ok, pts = cv2.rotatedRectangleIntersection(r1, r2)
                    if ok == cv2.INTERSECT_NONE or pts is None:
                        continue
                    inter = cv2.contourArea(pts)
                    amin = min(r1[1][0] * r1[1][1], r2[1][0] * r2[1][1])
                    if amin > 0 and inter / amin > iou_thr:
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            grid.setdefault((gx, gy), []).append(len(kept))
            kept.append(boxes[i])
    return kept


def extract_buildings_sam2(rgb_u8, roof_mask, px_m, **cfg):
    """Per-instance building footprints via SAM2, seeded with points inside the
    building/roof mask.  Tiles the map into 1024 px windows (native res, so
    houses are large enough for SAM2), point-prompts a grid of seeds that land
    on the roof mask, keeps house-sized masks that mostly overlap the roof
    mask, dedups by IoU-NMS, and fits a min-area oriented rectangle per
    instance.  Returns [(cx, cy, w, d, ang, hgt)] in PIXEL coords (px_m units),
    matching extract_buildings().  Raises if SAM2 / weights are unavailable."""
    import numpy as np
    import cv2
    predictor = _build_sam2_predictor(cfg)
    H, W = roof_mask.shape
    tile = int(cfg.get("tile", 1024)); ov = int(cfg.get("overlap", 128))
    step = max(1, tile - ov)
    seed_px = max(6, int(round(float(cfg.get("seed_m", 10.0)) / px_m)))
    min_m2 = float(cfg.get("min_m2", 15.0)); max_m2 = float(cfg.get("max_m2", 1200.0))
    boxes = []
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(H, y0 + tile); x1 = min(W, x0 + tile)
            rm = roof_mask[y0:y1, x0:x1]
            if rm.sum() * px_m * px_m < min_m2:
                continue
            predictor.set_image(np.ascontiguousarray(rgb_u8[y0:y1, x0:x1]))
            gy, gx = np.mgrid[0:rm.shape[0]:seed_px, 0:rm.shape[1]:seed_px]
            on = rm[gy.ravel(), gx.ravel()]
            seeds = np.stack([gx.ravel()[on], gy.ravel()[on]], 1)
            kept = []
            for sx, sy in seeds:
                try:
                    m, sc, _ = predictor.predict(
                        point_coords=np.array([[float(sx), float(sy)]]),
                        point_labels=np.array([1]), multimask_output=True)
                except Exception:
                    continue
                # SAM returns ~3 granularities (subpart/part/whole).  FLUX
                # blocks segment as ONE whole-block mask, so prefer the
                # SMALLEST house-sized mask instead of the block.
                cand = []
                for k in range(len(m)):
                    mm = np.asarray(m[k]).astype(bool)
                    a = mm.sum() * px_m * px_m
                    if a < min_m2 or a > max_m2:
                        continue
                    if (mm & rm).sum() < 0.4 * mm.sum():   # mostly roof
                        continue
                    cand.append((a, float(sc[k]), mm))
                if not cand:
                    continue
                cand.sort(key=lambda t: t[0])              # smallest first
                a, score, mm = cand[0]
                kept.append((score, mm))
            kept.sort(key=lambda t: -t[0])
            taken = []
            for _, mm in kept:
                if any(_sam2_iou(mm, tm) > 0.4 for tm in taken):
                    continue
                taken.append(mm)
            covered = np.zeros_like(rm, dtype=bool)
            for mm in taken:
                covered |= mm
                cnts, _ = cv2.findContours(mm.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                (bx, by), (bw, bh), ang = cv2.minAreaRect(c)
                w, d = bw * px_m, bh * px_m
                if w < 3.0 or d < 3.0:
                    continue
                cx, cy = bx + x0, by + y0
                boxes.append((float(cx), float(cy), max(3.0, w), max(3.0, d),
                              float(np.deg2rad(ang)), _det_rand(cx, cy, 3.5, 9.0)))
            # RESIDUAL coverage: roof area SAM2 did not resolve into
            # house-sized instances (typically whole uniform blocks) gets
            # the watershed/oriented-rect subdivision so no block is empty.
            resid = rm & ~covered
            if resid.sum() * px_m * px_m > 60.0:
                for (rcx, rcy, rw, rd, rang, rh) in \
                        extract_buildings_shaped(resid, px_m):
                    boxes.append((rcx + x0, rcy + y0, rw, rd, rang, rh))
    boxes = _dedup_boxes(boxes, px_m,
                         float(cfg.get("dedup_iou", 0.45)))
    return boxes


def extract_building_footprints(rgb_u8, roof_mask, px_m, **cfg):
    """Overlap-free house footprints as simplified contour POLYGONS.
    SAM2 instances claim roof pixels first (a global claimed bitmap ensures
    every pixel belongs to at most ONE house — polygons of a partition can
    never overlap); the residual roof is split by watershed.  Each polygon
    traces the painted house base outline (approxPolyDP, ~eps_m metres).
    Returns [(poly_px [N,2] float32, height_m)].  Raises if SAM2/cv2 are
    unavailable so the caller can fall back to boxes."""
    import cv2
    from scipy.ndimage import maximum_filter, label as _ndlabel
    predictor = _build_sam2_predictor(cfg)
    H, W = roof_mask.shape
    tile = int(cfg.get("tile", 1024)); ov = int(cfg.get("overlap", 128))
    step = max(1, tile - ov)
    seed_px = max(6, int(round(float(cfg.get("seed_m", 10.0)) / px_m)))
    min_m2 = float(cfg.get("min_m2", 20.0))
    max_m2 = float(cfg.get("max_m2", 1200.0))
    roof_frac = float(cfg.get("roof_frac", 0.4))
    eps_px = max(1.0, float(cfg.get("eps_m", 1.0)) / px_m)
    claimed = np.zeros((H, W), bool)
    out = []

    def emit(region, x0, y0):
        cnts, _ = cv2.findContours(region.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) * px_m * px_m < min_m2:
                continue
            ap = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
            if len(ap) > 24:
                ap = cv2.approxPolyDP(c, eps_px * 2.0, True).reshape(-1, 2)
            if len(ap) < 3:
                continue
            poly = ap.astype(np.float32)
            poly[:, 0] += x0; poly[:, 1] += y0
            hgt = _det_rand(float(poly[:, 0].mean()),
                            float(poly[:, 1].mean()), 3.5, 9.0)
            out.append((poly, hgt))

    er = max(1, int(round(2.5 / px_m)))
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(H, y0 + tile); x1 = min(W, x0 + tile)
            cl = claimed[y0:y1, x0:x1]
            rm = roof_mask[y0:y1, x0:x1] & ~cl
            if rm.sum() * px_m * px_m < min_m2:
                continue
            predictor.set_image(np.ascontiguousarray(rgb_u8[y0:y1, x0:x1]))
            gy, gx = np.mgrid[0:rm.shape[0]:seed_px, 0:rm.shape[1]:seed_px]
            on = rm[gy.ravel(), gx.ravel()]
            seeds = np.stack([gx.ravel()[on], gy.ravel()[on]], 1)
            cand = []
            for sx, sy in seeds:
                try:
                    m, sc, _ = predictor.predict(
                        point_coords=np.array([[float(sx), float(sy)]]),
                        point_labels=np.array([1]), multimask_output=True)
                except Exception:
                    continue
                pick = None
                for k in range(len(m)):
                    mm = np.asarray(m[k]).astype(bool)
                    a = mm.sum() * px_m * px_m
                    if a < min_m2 or a > max_m2:
                        continue
                    if (mm & rm).sum() < roof_frac * mm.sum():
                        continue
                    if pick is None or a < pick[0]:
                        pick = (a, float(sc[k]), mm)
                if pick:
                    cand.append(pick)
            cand.sort(key=lambda t: -t[1])
            for a, scv, mm in cand:
                mm = mm & rm & ~cl
                if mm.sum() * px_m * px_m < min_m2:
                    continue
                emit(mm, x0, y0)
                cl |= mm
            resid = rm & ~cl
            if resid.sum() * px_m * px_m >= min_m2:
                ncc, lab = cv2.connectedComponents(resid.astype(np.uint8), 8)
                for i in range(1, ncc):
                    comp = lab == i
                    a = comp.sum() * px_m * px_m
                    if a < min_m2:
                        continue
                    if a <= 300.0:
                        emit(comp, x0, y0); cl |= comp
                        continue
                    dt = cv2.distanceTransform(comp.astype(np.uint8),
                                               cv2.DIST_L2, 5)
                    win = max(5, int(round(6.0 / px_m)) | 1)
                    peak = (dt == maximum_filter(dt, size=win)) & (dt > er)
                    mk, nmk = _ndlabel(peak)
                    if nmk < 2:
                        emit(comp, x0, y0); cl |= comp
                        continue
                    ws = cv2.watershed(
                        cv2.cvtColor((comp * 255).astype(np.uint8),
                                     cv2.COLOR_GRAY2BGR),
                        mk.astype(np.int32).copy())
                    for lbl in range(1, nmk + 1):
                        seg_m = (ws == lbl) & comp
                        if seg_m.sum() * px_m * px_m >= min_m2:
                            emit(seg_m, x0, y0)
                    cl |= comp
    print(f"[pcg] footprint instances: {len(out)} (overlap-free partition)")
    return out


def _load_env_cfg():
    """Extractor config from the TERRAIN_PCG_CFG env json (loop hand-off)."""
    p = os.environ.get("TERRAIN_PCG_CFG")
    if p and os.path.exists(p):
        try:
            cfg = json.load(open(p, encoding="utf-8"))
            print(f"[pcg] extractor cfg from {p}: {cfg}")
            return cfg
        except Exception:
            pass
    return {}


def extract_buildings_best(roof_mask, px_m, rgb_u8=None):
    """Preferred building extractor: SAM2 per-instance masks when available
    (needs weights + a GPU and the albedo), else the oriented-rect heuristic,
    else the grid extractor.  Same return contract as extract_buildings()."""
    cfg = {}
    cfg_path = os.environ.get("TERRAIN_PCG_CFG")
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            print(f"[pcg] extractor cfg from {cfg_path}: {cfg}")
        except Exception:
            cfg = {}
    if rgb_u8 is not None:
        try:
            b = extract_buildings_sam2(rgb_u8, roof_mask, px_m, **cfg)
            if b:
                print(f"[pcg] SAM2 per-instance buildings: {len(b)}")
                return b
            print("[pcg] SAM2 produced no instances; using heuristic fallback")
        except Exception as e:                                # noqa: BLE001
            print(f"[pcg] SAM2 unavailable ({e}); oriented-rect heuristic")
    return extract_buildings_shaped(roof_mask, px_m)


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
        self.uv = []     # OPTIONAL explicit UVs, parallel to pos.  When a
        #                  group supplies them for every vertex write_glb
        #                  uses them; otherwise it falls back to planar
        #                  world-XZ UVs (fine for solid debug colours,
        #                  useless for real tiling textures).

    def add(self, verts, normals, faces, uvs=None):
        base = len(self.pos)
        self.pos.extend(verts)
        self.nrm.extend(normals)
        if uvs is not None:
            self.uv.extend(uvs)
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


def _ear_clip(poly):
    """Triangulate a simple polygon into index triples (ear clipping,
    O(n^2); footprints are capped at ~24 verts so this is instant)."""
    n = len(poly)
    if n < 3:
        return []
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    idx = list(range(n))
    if area < 0:
        idx.reverse()

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1]) -
                (a[1] - o[1]) * (b[0] - o[0]))
    tris, guard = [], 0
    while len(idx) > 3 and guard < 2000:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 0:
                continue
            if any(cross(a, b, poly[j]) >= 0 and cross(b, c, poly[j]) >= 0
                   and cross(c, a, poly[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def add_prism(acc, poly_xz, hgt, y0):
    """Extrude a house FOOTPRINT polygon: vertical walls + ear-clipped flat
    roof cap.  The proxy base traces the painted outline exactly (up to the
    contour simplification epsilon), unlike an oriented bounding box."""
    n = len(poly_xz)
    if n < 3:
        return
    v = [(x, y0, z) for x, z in poly_xz] + \
        [(x, y0 + hgt, z) for x, z in poly_xz]
    nrm = [(0, 1, 0)] * (2 * n)
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [i, j, n + j, i, n + j, n + i]
    for (a, b, c) in _ear_clip(poly_xz):
        f += [n + a, n + b, n + c]
    acc.add(v, nrm, f)


def add_house(acc, poly_w, eave_h, y0, gs=2.0, pitch=0.8, cap=4.5):
    """House proxy from a footprint polygon: vertical walls to the eave, then
    a HIP ROOF whose ridge network follows the footprint\'s medial axis.
    Construction: roof rise at an interior point = min(distance-to-outline *
    pitch, cap) — the level sets of the distance transform ARE the straight-
    skeleton offsets, so the ridge lines land exactly on the edge-connection
    skeleton of the outline.  Surface is a Delaunay triangulation of the
    boundary ring + an interior grid, clipped to the footprint."""
    import cv2
    from scipy.spatial import Delaunay
    P2 = np.asarray(poly_w, np.float32)
    n = len(P2)
    if n < 3:
        return
    mn = P2.min(0) - gs
    mx = P2.max(0) + gs
    gw = max(4, int(np.ceil((mx[0] - mn[0]) / gs)) + 1)
    gh = max(4, int(np.ceil((mx[1] - mn[1]) / gs)) + 1)
    cont = ((P2 - mn) / gs).astype(np.float32)
    mask = np.zeros((gh, gw), np.uint8)
    cv2.fillPoly(mask, [np.round(cont).astype(np.int32)], 1)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * gs   # metres
    pts = [(float(x), float(z)) for x, z in P2]
    rise = [0.0] * n                       # roof meets the eave at the wall
    ys, xs = np.nonzero(mask)
    for yy, xx in zip(ys, xs):
        if dt[yy, xx] < gs * 0.8:          # too close to the ring
            continue
        pts.append((float(mn[0] + xx * gs), float(mn[1] + yy * gs)))
        rise.append(float(min(dt[yy, xx] * pitch, cap)))
    pts_np = np.asarray(pts, np.float32)
    try:
        tri = Delaunay(pts_np)
    except Exception:
        # degenerate footprint: fall back to a flat prism
        add_prism(acc, [tuple(p) for p in P2], eave_h, y0)
        return
    cont_cv = P2.reshape(-1, 1, 2).astype(np.float32)
    v = [(float(x), y0, float(z)) for x, z in P2] + \
        [(float(x), y0 + eave_h, float(z)) for x, z in P2] + \
        [(float(p[0]), y0 + eave_h + h, float(p[1]))
         for p, h in zip(pts_np, rise)]
    nrm = [(0, 1, 0)] * len(v)
    f = []
    for i in range(n):                     # walls ground->eave
        j = (i + 1) % n
        f += [i, j, n + j, i, n + j, n + i]
    roff = 2 * n
    for simp in tri.simplices:             # roof surface
        c = pts_np[simp].mean(0)
        if cv2.pointPolygonTest(cont_cv,
                                (float(c[0]), float(c[1])), False) < 0:
            continue
        a, b, c2 = (int(t) for t in simp)
        f += [roff + a, roff + b, roff + c2]
    acc.add(v, nrm, f)


# ── Proxy upgrade: procedural albedos + clean building/tree builders ────────
_TEX_CACHE = {}

# Variant palettes — one MATERIAL per entry, chosen per building/tree by a
# deterministic hash of its map position, so towns get a mix of render
# colours instead of a single clone-stamped look.
_WALL_BASES = [(226, 221, 208),   # whitewash
               (213, 202, 178),   # warm plaster
               (208, 183, 141),   # sand / ochre
               (186, 168, 150)]   # weathered gray
_ROOF_BASES = [(167, 82, 56),     # terracotta
               (122, 63, 48),     # dark brown tile
               (109, 109, 116),   # slate gray
               (150, 96, 60)]     # faded clay
_LEAF_BASES = [(64, 116, 55),     # mid green
               (44, 90, 42),      # dark conifer green
               (109, 128, 52),    # olive / dry
               (130, 106, 44)]    # autumn tint


def _load_terrain_grid(path):
    """Height sampler built from the terrain RENDER mesh (<stamp>.glb) —
    the authoritative surface the player actually sees.  The mesh is a
    regular vertex grid over the world extent; returns (xs, zs, Y)."""
    import json as _json
    import struct as _st
    data = open(path, "rb").read()
    off, chunks = 12, {}
    while off < len(data):
        clen, ctype = _st.unpack("<II", data[off:off + 8])
        off += 8
        chunks[ctype] = data[off:off + clen]
        off += clen
    g = _json.loads(chunks[0x4E4F534A])
    buf = chunks[0x004E4942]
    pos = []
    for m in g["meshes"]:
        for pr in m["primitives"]:
            a = g["accessors"][pr["attributes"]["POSITION"]]
            bv = g["bufferViews"][a["bufferView"]]
            if a["componentType"] != 5126:
                raise ValueError("non-float positions")
            arr = np.frombuffer(
                buf, np.float32, a["count"] * 3,
                bv.get("byteOffset", 0) + a.get("byteOffset", 0))
            pos.append(arr.reshape(-1, 3))
    P = np.vstack(pos).astype(np.float64)
    xs = np.unique(np.round(P[:, 0], 3))
    zs = np.unique(np.round(P[:, 2], 3))
    if len(xs) * len(zs) != len(P):
        raise ValueError("terrain mesh is not a regular grid")
    Y = np.full((len(zs), len(xs)), np.nan, np.float32)
    xi = np.searchsorted(xs, np.round(P[:, 0], 3))
    zi = np.searchsorted(zs, np.round(P[:, 2], 3))
    Y[zi, xi] = P[:, 1]
    if np.isnan(Y).any():
        raise ValueError("terrain grid has holes")
    return xs, zs, Y


def _grid_sample(grid, wx, wz):
    """Bilinear height on the render-mesh grid (matches the rasterised
    triangles to within half a quad)."""
    xs, zs, Y = grid
    fx = int(np.clip(np.searchsorted(xs, wx) - 1, 0, len(xs) - 2))
    fz = int(np.clip(np.searchsorted(zs, wz) - 1, 0, len(zs) - 2))
    tx = float(np.clip((wx - xs[fx]) / max(xs[fx + 1] - xs[fx], 1e-6),
                       0.0, 1.0))
    tz = float(np.clip((wz - zs[fz]) / max(zs[fz + 1] - zs[fz], 1e-6),
                       0.0, 1.0))
    return float(Y[fz, fx] * (1 - tx) * (1 - tz) +
                 Y[fz, fx + 1] * tx * (1 - tz) +
                 Y[fz + 1, fx] * (1 - tx) * tz +
                 Y[fz + 1, fx + 1] * tx * tz)


def _normal_from_height(h, strength=2.0):
    """Tangent-space normal map from a height field (glTF convention)."""
    gy, gx = np.gradient(h.astype(np.float32))
    n = np.dstack([-gx * strength, gy * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    return Image.fromarray(
        np.clip((n * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8), "RGB")


def _mr_image(rough, metal=None):
    """glTF metallicRoughness map: G = roughness, B = metallic."""
    h, w = rough.shape
    out = np.zeros((h, w, 3), np.float32)
    out[..., 1] = rough
    if metal is not None:
        out[..., 2] = metal
    return Image.fromarray(
        np.clip(out * 255, 0, 255).astype(np.uint8), "RGB")


def _tex_bundle(name, size=256):
    """Deterministic PBR texture set {albedo, normal, mr} for a material,
    embedded in the GLB — albedo alone made close-up surfaces read as
    flat printed paper; the normal + roughness maps give the renderer's
    PBR path something to light.  Variant names like 'wall2' / 'roof1' /
    'leaf0' select a palette entry."""
    import zlib
    if name in _TEX_CACHE:
        return _TEX_CACHE[name]
    rng = np.random.default_rng(zlib.crc32(name.encode()))

    def _variant(prefix, bases):
        s = name[len(prefix):]
        return bases[min(int(s) if s else 0, len(bases) - 1)]

    hgt = np.zeros((size, size), np.float32)     # height field (px units)
    rough = np.full((size, size), 0.9, np.float32)
    if name.startswith("wall"):
        img = (np.array(_variant("wall", _WALL_BASES), np.float32) +
               rng.normal(0, 5, (size, size, 1)))
        img += np.linspace(6, -10, size)[:, None, None]      # weathering
        from scipy import ndimage as _ndi
        hgt = _ndi.gaussian_filter(
            rng.normal(0, 1.2, (size, size)).astype(np.float32), 3)
        rough[:] = 0.88

        def _win(x0, y0, w, h):
            img[y0:y0+h, x0:x0+w] = ([68, 80, 95] +
                                     rng.normal(0, 4, (h, w, 1)))
            fr = [156, 152, 142]
            img[y0:y0+3, x0:x0+w] = fr; img[y0+h-3:y0+h, x0:x0+w] = fr
            img[y0:y0+h, x0:x0+3] = fr; img[y0:y0+h, x0+w-3:x0+w] = fr
            img[y0+h//2-1:y0+h//2+1, x0:x0+w] = fr
            hgt[y0:y0+h, x0:x0+w] -= 5.0                     # recessed
            hgt[y0:y0+3, x0:x0+w] += 5.5                     # raised frame
            hgt[y0+h-3:y0+h, x0:x0+w] += 5.5
            hgt[y0:y0+h, x0:x0+3] += 5.5; hgt[y0:y0+h, x0+w-3:x0+w] += 5.5
            rough[y0:y0+h, x0:x0+w] = 0.25                   # glass
        # ONE centered window per bay tile (was two at 1/4 and 3/4 —
        # at a 5 m bay that put a window every 2.5 m, reading as a
        # near-continuous glass band; a single window per bay gives
        # the calm ~5 m rhythm real facades have).  Bay BOUNDARIES are
        # the guaranteed-blank strips now — door snapping relies on it.
        _win(93, 60, 70, 96)
        # (window band sits HIGH in the tile: the blank lower ~40% is a
        # plinth, so a wall sunk into a slope buries plaster, not glass)
    elif name.startswith("roof"):
        img = (np.array(_variant("roof", _ROOF_BASES), np.float32) +
               rng.normal(0, 6, (size, size, 1)))
        row = np.arange(size, dtype=np.float32) % 32
        hgt += (row[:, None] / 32.0) * 3.0                   # tile ramp
        for r in range(0, size, 32):                         # tile rows
            img[r:r+3] *= 0.62
            img[r+3:r+8] *= 1.14
            hgt[r:r+3] -= 3.0                                # row step
        for r0 in range(0, size, 32):                        # joints,
            offs = 16 if (r0 // 32) % 2 else 0               # half-offset
            for c in range(offs, size, 32):
                img[r0:r0+32, c:c+2] *= 0.82
                hgt[r0:r0+32, c:c+2] -= 1.6
            for c in range(0, size, 8):                      # per-tile tint
                img[r0:r0+32, c:c+8] *= rng.uniform(0.93, 1.07)
        rough[:] = 0.82
    elif name == "bark":
        img = (np.array([97, 74, 55], np.float32) +
               rng.normal(0, 5, (size, size, 1)))
        streak = rng.normal(0, 13, (1, size)).astype(np.float32)
        img += streak[:, :, None]                            # vertical streaks
        hgt += np.repeat(streak, size, 0) * 0.12
        for c in rng.integers(0, size, 10):
            img[:, c:c+2] *= 0.7
            hgt[:, c:c+2] -= 2.0
        rough[:] = 0.95
    elif name.startswith("leaf"):
        from scipy import ndimage as _ndi
        blotch = _ndi.gaussian_filter(
            np.kron(rng.normal(0, 1, (size // 16, size // 16)),
                    np.ones((16, 16))).astype(np.float32), 5)[:, :, None]
        img = (np.array(_variant("leaf", _LEAF_BASES), np.float32) *
               (1.0 + 0.09 * blotch) + rng.normal(0, 5, (size, size, 1)))
        hgt = blotch[..., 0] * 1.2
        rough[:] = 0.95
    elif name == "door":
        img = np.full((size, size, 3), [96, 66, 40], np.float32)
        img += rng.normal(0, 4, (size, size, 1))
        for c in range(24, size - 24, 52):                   # plank seams
            img[:, c:c+3] *= 0.72
            hgt[:, c:c+3] -= 2.5
        fr = 20                                              # frame
        img[:fr] = [64, 46, 30]; img[-fr:] = [64, 46, 30]
        img[:, :fr] = [64, 46, 30]; img[:, -fr:] = [64, 46, 30]
        hgt[:fr] += 4; hgt[-fr:] += 4
        hgt[:, :fr] += 4; hgt[:, -fr:] += 4
        img[118:138, 196:216] = [200, 190, 150]              # handle
        hgt[118:138, 196:216] += 5
        rough[:] = 0.72
    elif name == "fence":
        img = np.full((size, size, 3), [72, 62, 50], np.float32)  # shadow bg
        for c in range(0, size, 44):                         # pickets
            img[30:, c:c+28] = ([204, 198, 186] +
                                rng.normal(0, 5, (size - 30, 28, 1)))
            hgt[30:, c:c+28] += 4
            img[18:30, c+8:c+20] = [204, 198, 186]           # pointed top
            hgt[18:30, c+8:c+20] += 4
        for r in (86, 190):                                  # rails
            img[r:r+14] = np.maximum(img[r:r+14], 176)
            hgt[r:r+14] += 2
        rough[:] = 0.85
    elif name == "road":
        img = (np.full((size, size, 3), [56, 56, 59], np.float32) +
               rng.normal(0, 4, (size, size, 1)))
        from scipy import ndimage as _ndi
        patch = _ndi.gaussian_filter(
            rng.normal(0, 1, (size, size)).astype(np.float32), 9)
        img += patch[:, :, None] * 9.0
        hgt = patch * 1.0
        rough[:] = 0.96
    else:
        raise KeyError(name)
    strength = {"door": 1.5, "fence": 1.2, "road": 0.7}.get(
        name, {"w": 2.0, "r": 2.5, "b": 1.5, "l": 0.6}.get(name[0], 1.5))
    _TEX_CACHE[name] = {
        "albedo": Image.fromarray(np.clip(img, 0, 255).astype(np.uint8),
                                  "RGB"),
        "normal": _normal_from_height(hgt, strength),
        "mr": _mr_image(rough)}
    return _TEX_CACHE[name]


def _tex(name, size=256):
    """Back-compat: albedo of the PBR bundle."""
    return _tex_bundle(name, size)["albedo"]


def _poly_area_signed(p):
    a = 0.0
    for i in range(len(p)):
        x1, z1 = p[i]; x2, z2 = p[(i + 1) % len(p)]
        a += x1 * z2 - x2 * z1
    return 0.5 * a


def _wall_ring(acc, ring, y0, y1):
    """Wall quads around a CLOCKWISE-from-above ring (outward windings).
    UVs tile an INTEGER number of window bays along each wall and an
    integer number of storeys up its height, anchored to that wall's own
    edges — cumulative/metric UVs sliced windows at corners, eaves and
    ground, and scattered rows at heights that made no sense."""
    nv = max(1, int(round((y1 - y0) / 3.0)))
    for i in range(len(ring)):
        p0, p1 = ring[i], ring[(i + 1) % len(ring)]
        el = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if el < 0.3:
            continue
        nu = max(1, int(round(el / 5.0)))
        v = [(p0[0], y0, p0[1]), (p1[0], y0, p1[1]),
             (p1[0], y1, p1[1]), (p0[0], y1, p0[1])]
        uvs = [(0.0, float(nv)), (float(nu), float(nv)),
               (float(nu), 0.0), (0.0, 0.0)]
        acc.add(v, [(0, 1, 0)] * 4, [0, 1, 2, 0, 2, 3], uvs)


def _add_gable_house(acc_wall, acc_roof, c, w, d, ang, eave, rise, y0,
                     hip=False):
    """Rectangular wing with a ridge along the long axis.  hip=False:
    gable ends (vertical wall triangles).  hip=True: the ridge is inset
    from both ends and the end faces slope like the sides — the two roof
    forms real houses actually use.  eave/rise are supplied by the
    CALLER: all wings of one building share an eave (per-wing eaves let
    storey walls poke through sibling roofs as sawtooth artefacts), and
    each wing's rise comes from its own width so secondary ridges tuck
    under the main roof."""
    if w < d:
        w, d = d, w
        ang += np.pi * 0.5
    w, d = max(w, 3.0), max(d, 2.6)
    ca, sa = float(np.cos(ang)), float(np.sin(ang))

    def pt(u, v):
        return (c[0] + u * ca - v * sa, c[1] + u * sa + v * ca)
    e = min(0.45 * d, 0.33 * w) if hip else 0.0   # ridge end inset
    A, B = pt(-w / 2, -d / 2), pt(w / 2, -d / 2)
    C, D = pt(w / 2, d / 2), pt(-w / 2, d / 2)
    _wall_ring(acc_wall, [A, D, C, B], y0, y0 + eave)   # clockwise ⇒ outward
    ye, yr = y0 + eave, y0 + eave + rise
    # CLOSED roof volume (no paper sheets, no see-through slits): the
    # slopes span an extended eave frame, hip ends are sloped faces on
    # that same frame, and a flat SOFFIT cap underneath seals the body —
    # from below the overhang you see a ceiling, not the world.
    ovl, ovs, drop = (0.45 if hip else 0.35), 0.45, 0.12
    W2, D2 = w / 2 + ovl, d / 2 + ovs
    E = (e + ovl) if hip else 0.0                # ridge inset on frame
    yb = ye - drop
    R1, R2 = pt(-W2 + E, 0.0), pt(W2 - E, 0.0)
    Ea, Eb = pt(-W2, -D2), pt(W2, -D2)
    Ec, Ed = pt(W2, D2), pt(-W2, D2)
    if not hip:
        # vertical gable wall triangles at the wall plane (visible in
        # the shadow slot behind the verge overhang)
        # UVs pinned inside the texture's blank plaster strip (above the
        # window row) — a metric mapping floated half-windows up here.
        Rw1, Rw2 = pt(-w / 2, 0.0), pt(w / 2, 0.0)
        guv = [(0.0, 0.20), (d / 5.0, 0.20), (d / 10.0, 0.02)]
        for tri in (((C[0], ye, C[1]), (B[0], ye, B[1]),
                     (Rw2[0], yr, Rw2[1])),
                    ((A[0], ye, A[1]), (D[0], ye, D[1]),
                     (Rw1[0], yr, Rw1[1]))):
            acc_wall.add(list(tri), [(0, 1, 0)] * 3, [0, 1, 2], guv)
    else:
        # sloped hip end faces on the eave frame (outward windings).
        # PLANAR UVs: u tracks each vertex's position along the end
        # face, v the slope distance from the eave — constant tile size.
        sle = float(np.hypot(max(E, 0.3), rise + drop))
        for tri, uvs in (
                (([(Ec[0], yb, Ec[1]), (Eb[0], yb, Eb[1]),
                   (R2[0], yr, R2[1])]),
                 [(D2 / 3.0, sle / 3.0), (-D2 / 3.0, sle / 3.0),
                  (0.0, 0.0)]),
                (([(Ea[0], yb, Ea[1]), (Ed[0], yb, Ed[1]),
                   (R1[0], yr, R1[1])]),
                 [(D2 / 3.0, sle / 3.0), (-D2 / 3.0, sle / 3.0),
                  (0.0, 0.0)])):
            acc_roof.add(tri, [(0, 1, 0)] * 3, [0, 1, 2], uvs)
    # side slopes on the eave frame.  PLANAR UVs: u = metres along the
    # ridge axis PER VERTEX (the hip edge crops tiles diagonally like a
    # real roof), v = slope metres from the eave.  Stretching the short
    # ridge edge to the eave's u-range sheared the tiles and put an
    # affine seam across the quad diagonal.
    sl = float(np.hypot(D2, rise + drop))
    for side in (-1.0, 1.0):
        E1 = pt(-W2, side * D2)
        E2 = pt(W2, side * D2)
        Ra = pt(-W2 + E, 0.0)
        Rb = pt(W2 - E, 0.0)
        v = [(E1[0], yb, E1[1]), (E2[0], yb, E2[1]),
             (Rb[0], yr, Rb[1]), (Ra[0], yr, Ra[1])]
        uvs = [(-W2 / 3.0, sl / 3.0), (W2 / 3.0, sl / 3.0),
               ((W2 - E) / 3.0, 0.0), ((-W2 + E) / 3.0, 0.0)]
        acc_roof.add(v, [(0, 1, 0)] * 4, [0, 1, 2, 0, 2, 3], uvs)
    # soffit: flat cap under the whole eave frame
    v = [(Ea[0], yb, Ea[1]), (Eb[0], yb, Eb[1]),
         (Ec[0], yb, Ec[1]), (Ed[0], yb, Ed[1])]
    uvs = [(0.0, 0.0), (2 * W2 / 3.0, 0.0),
           (2 * W2 / 3.0, 2 * D2 / 3.0), (0.0, 2 * D2 / 3.0)]
    acc_roof.add(v, [(0, 1, 0)] * 4, [0, 1, 2, 0, 2, 3], uvs)


def _inset_ring(ring, d):
    """Miter-offset every vertex of a CLOCKWISE ring inward by edge
    distance d (the ridge ring of a hip roof).  Returns None when the
    inset degenerates — flipped orientation, collapsed area, spiked
    miter, or self-intersection — and the caller falls back."""
    n = len(ring)
    P = np.asarray(ring, np.float64)
    offs = np.zeros((n, 2))
    for i in range(n):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % n]
        e0, e1 = p1 - p0, p2 - p1
        l0, l1 = np.linalg.norm(e0), np.linalg.norm(e1)
        if l0 < 1e-6 or l1 < 1e-6:
            return None
        e0, e1 = e0 / l0, e1 / l1
        n0 = np.array([e0[1], -e0[0]])       # inward normals (CW ring)
        n1 = np.array([e1[1], -e1[0]])
        den = 1.0 + float(n0 @ n1)
        if den < 0.15:                       # near-180° spike: CLAMP the
            m = n0 + n1                      # miter instead of giving up
            ml = float(np.linalg.norm(m))
            off = (m / ml if ml > 1e-6 else
                   np.array([-e0[0], -e0[1]])) * d
        else:
            off = (n0 + n1) / den * d
            ol = float(np.linalg.norm(off))
            if ol > 3.0 * d:
                off *= 3.0 * d / ol
        offs[i] = off

    def _cr(p, q, r):
        return ((q[0] - p[0]) * (r[1] - p[1]) -
                (q[1] - p[1]) * (r[0] - p[0]))

    def _crossers(inner):
        bad = set()
        for i in range(n):                   # O(n²); rings are small
            a, b = inner[i], inner[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or (j + 1) % n == i or (i + 1) % n == j:
                    continue
                c, e = inner[j], inner[(j + 1) % n]
                if (((_cr(c, e, a) > 0) != (_cr(c, e, b) > 0)) and
                        ((_cr(a, b, c) > 0) != (_cr(a, b, e) > 0))):
                    bad.update((i, (i + 1) % n, j, (j + 1) % n))
        return bad
    # The inset is pushed toward the footprint's MEDIAL LINE (so the cap
    # collapses into a main ridge), which makes degeneracies routine —
    # REPAIR instead of rejecting: vertices whose ridge segments cross or
    # that escape the outline get their offsets shrunk locally; a ring
    # that folded straight through the collapse line (flipped signed
    # area, no crossings) gets a global shrink.
    import cv2
    cont = np.asarray(ring, np.float32).reshape(-1, 1, 2)
    a_out = _poly_area_signed(ring)
    inner = None
    for _ in range(12):
        inner = [(float(P[i, 0] + offs[i, 0]),
                  float(P[i, 1] + offs[i, 1])) for i in range(n)]
        if np.sign(_poly_area_signed(inner)) != np.sign(a_out):
            offs *= 0.65
            continue
        bad = _crossers(inner)
        for i in range(n):
            if cv2.pointPolygonTest(cont, inner[i], False) < 0:
                bad.add(i)
        if not bad:
            break
        for i in bad:
            offs[i] *= 0.45
    else:
        return None
    if (_crossers(inner) or
            np.sign(_poly_area_signed(inner)) != np.sign(a_out)):
        return None
    return inner


def _add_flat_block(acc_wall, acc_roof, P, hgt, y0):
    """Irregular footprint: straight walls + pitched HIP roof — the eave
    ring rises to an inset ridge ring topped with a small cap, so big
    blocks read as ridged roofs instead of flat slabs.  Falls back to a
    flat slab only when the inset ring degenerates."""
    import cv2
    ring = [tuple(p) for p in P]
    if _poly_area_signed(ring) > 0:          # need CLOCKWISE for outward
        ring = ring[::-1]
    tris = _ear_clip(ring)
    if len(tris) < len(ring) - 2:
        # Ear clipping bailed — the simplified outline self-intersects.
        # A fan over a bad ring sprays giant roof triangles outside the
        # building, so fall back to the convex hull instead.
        hull = cv2.convexHull(np.asarray(P, np.float32).reshape(-1, 1, 2))
        ring = [tuple(p) for p in hull.reshape(-1, 2)]
        if _poly_area_signed(ring) > 0:
            ring = ring[::-1]
        tris = _ear_clip(ring)
    eave = max(2.6, hgt)
    ye = y0 + eave
    _wall_ring(acc_wall, ring, y0, ye)
    (_, _), (rw, rh), _ = cv2.minAreaRect(
        np.asarray(ring, np.float32).reshape(-1, 1, 2))
    # Aim the inset at HALF the short side, i.e. the medial collapse
    # distance of the footprint: the cap shrinks to a spine and the roof
    # gets a MAIN RIDGE LINE instead of a low plateau.
    inset = float(np.clip(0.48 * min(rw, rh), 1.2, 14.0))
    inner = None
    for scale in (1.0, 0.7, 0.45, 0.25):     # shrink until the ridge
        trial = max(0.8, inset * scale)      # ring stops degenerating
        inner = _inset_ring(ring, trial)
        if inner is not None:
            inset = trial
            break
        if trial <= 0.8:
            break
    if inner is not None:
        # Constant-pitch roof: each ridge vertex rises with the inset
        # depth it actually reached (necks that were repaired inward sit
        # lower) — the straight-skeleton behaviour of a real hip roof.
        PITCH, RISE_MIN, RISE_MAX = 1.0, 0.8, 7.0   # ~45° pitch

        def _dist_to_ring(p):
            best = 1e9
            for i in range(len(ring)):
                a, b = np.asarray(ring[i]), np.asarray(ring[(i + 1) %
                                                           len(ring)])
                ab = b - a
                t = float(np.clip(np.dot(p - a, ab) /
                                  max(float(ab @ ab), 1e-9), 0.0, 1.0))
                best = min(best, float(np.linalg.norm(a + t * ab - p)))
            return best

        rises = [float(np.clip(PITCH * _dist_to_ring(np.asarray(q)),
                               RISE_MIN, RISE_MAX)) for q in inner]
        n = len(ring)
        for i in range(n):                   # roof band: eave -> ridge
            j = (i + 1) % n
            o0, o1, i0, i1 = ring[i], ring[j], inner[i], inner[j]
            ex, ez = o1[0] - o0[0], o1[1] - o0[1]
            el = float(np.hypot(ex, ez))
            ux, uz = ex / el, ez / el
            # PLANAR per-vertex UVs (project onto the eave edge; v =
            # slope distance from it) — stretching the shorter ridge
            # edge to the eave's u-range sheared the tiles and left an
            # affine seam across the quad diagonal.
            def _uv(p, y):
                dx, dz = p[0] - o0[0], p[1] - o0[1]
                horiz = abs(dx * uz - dz * ux)
                return ((dx * ux + dz * uz) / 3.0,
                        float(np.hypot(horiz, y - ye)) / 3.0)
            v = [(o0[0], ye, o0[1]), (o1[0], ye, o1[1]),
                 (i1[0], ye + rises[j], i1[1]),
                 (i0[0], ye + rises[i], i0[1])]
            uvs = [(0.0, 0.0), (el / 3.0, 0.0),
                   _uv(i1, ye + rises[j]), _uv(i0, ye + rises[i])]
            acc_roof.add(v, [(0, 1, 0)] * 4, [0, 1, 2, 0, 2, 3], uvs)
        ctris = _ear_clip(inner)             # ridge cap (thin spine)
        v = [(x, ye + r, z) for (x, z), r in zip(inner, rises)]
        uvs = [(x / 2.5, z / 2.5) for x, z in inner]
        acc_roof.add(v, [(0, 1, 0)] * len(v),
                     [i for t in ctris for i in t], uvs)
    else:
        v = [(x, ye, z) for x, z in ring]
        uvs = [(x / 2.5, z / 2.5) for x, z in ring]
        acc_roof.add(v, [(0, 1, 0)] * len(v),
                     [i for t in tris for i in t], uvs)


def _dedupe_footprints(bld_polys, H, px_m):
    """Resolve overlaps BETWEEN footprints: the extractor's per-instance
    polygons can still overlap after contour simplification (hundreds of
    pairs on a map, tens of m² each), which made neighbouring houses
    interpenetrate.  Rasterise footprints onto a shared claim map (1 m
    cells) largest-first, keep each footprint's biggest unclaimed
    component, and re-trace its outline."""
    import cv2
    cell = max(1, int(round(1.0 / px_m)))          # px per ~1 m cell
    G = max(1, H // cell)
    claim = np.zeros((G, G), np.uint8)
    res = [None] * len(bld_polys)
    order = sorted(range(len(bld_polys)),
                   key=lambda i: -abs(cv2.contourArea(
                       np.asarray(bld_polys[i][0], np.float32))))
    for i in order:
        poly, hgt = bld_polys[i]
        p = np.asarray(poly, np.float64) / cell
        x0 = max(int(np.floor(p[:, 0].min())) - 1, 0)
        y0 = max(int(np.floor(p[:, 1].min())) - 1, 0)
        x1 = min(int(np.ceil(p[:, 0].max())) + 2, G)
        y1 = min(int(np.ceil(p[:, 1].max())) + 2, G)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [np.round(p - [x0, y0]).astype(np.int32)], 1)
        m[claim[y0:y1, x0:x1] > 0] = 0
        n, lab = cv2.connectedComponents(m)
        if n <= 1:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = (lab == int(sizes.argmax())).astype(np.uint8)
        if keep.sum() * (cell * px_m) ** 2 < 8.0:  # < 8 m² left: drop
            continue
        claim[y0:y1, x0:x1] |= keep
        cont, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        c = max(cont, key=cv2.contourArea).reshape(-1, 2)
        if len(c) < 3:
            continue
        res[i] = (((c + [x0, y0]).astype(np.float32) + 0.5) * cell, hgt)
    return [r for r in res if r is not None], claim


def _largest_rect(mask):
    """Largest all-ones axis-aligned rectangle in a binary mask (stack-
    based histogram sweep).  Returns (area, r0, c0, r1, c1) or None."""
    Hm, Wm = mask.shape
    heights = np.zeros(Wm, np.int32)
    best = (0, None)
    for r in range(Hm):
        heights = (heights + 1) * mask[r]
        stack = []
        for cc in range(Wm + 1):
            h = int(heights[cc]) if cc < Wm else 0
            start = cc
            while stack and stack[-1][1] >= h:
                s, hh = stack.pop()
                if hh * (cc - s) > best[0]:
                    best = (hh * (cc - s), (r - hh + 1, s, r, cc - 1))
                start = s
            stack.append((start, h))
    return best[1] and (best[0],) + best[1]


def _decompose_rects(P, max_wings=4):
    """Split a traced footprint into up to `max_wings` oriented rectangles
    (REAL-WORLD reference: houses are rectangles and L/T/U compositions
    of them, not amoebas).  Rasterise the outline in its dominant-axis
    frame, peel off the largest inscribed rectangle repeatedly, and map
    the rectangles back to world space."""
    import cv2
    (rcx, rcz), (rw, rh), rang = cv2.minAreaRect(P.reshape(-1, 1, 2))
    th = float(np.deg2rad(rang))
    ca, sa = np.cos(th), np.sin(th)
    rot = np.array([[ca, sa], [-sa, ca]], np.float64)      # world -> frame
    Q = (P.astype(np.float64) - [rcx, rcz]) @ rot.T
    res = 0.5 if max(rw, rh) < 80 else 1.0
    mn = Q.min(0) - res
    cells = np.ceil((Q.max(0) - mn) / res).astype(int) + 2
    if cells[0] * cells[1] > 400000:
        return []
    mask = np.zeros((cells[1], cells[0]), np.uint8)        # rows = frame-y
    cv2.fillPoly(mask, [np.round((Q - mn) / res).astype(np.int32)], 1)
    total = int(mask.sum())
    if not total:
        return []
    rects, covered = [], 0
    for _ in range(max_wings):
        got = _largest_rect(mask)
        if not got:
            break
        a, r0, c0, r1, c1 = got
        wm, dm = (c1 - c0 + 1) * res, (r1 - r0 + 1) * res
        if min(wm, dm) < 2.4 or a * res * res < max(10.0, 0.06 * total *
                                                    res * res):
            break
        fcx = mn[0] + (c0 + c1 + 1) * 0.5 * res
        fcy = mn[1] + (r0 + r1 + 1) * 0.5 * res
        wc = np.array([fcx, fcy]) @ rot + [rcx, rcz]       # frame -> world
        rects.append((float(wc[0]), float(wc[1]), float(wm), float(dm), th))
        mask[r0:r1 + 1, c0:c1 + 1] = 0
        covered += a
        if covered >= 0.90 * total:
            break
    return rects


def add_building(acc_wall, acc_roof, poly_w, hgt, y0,
                 acc_door=None, door_dir=None):
    """House from a traced footprint, built the way real houses are
    shaped: the outline is decomposed into 1-4 oriented rectangular
    WINGS, each a gable- or hip-roofed block (ridge along its long axis);
    wings overlap slightly so L/T/U compositions read as one building
    with proper roof valleys.  Falls back to the outline-tracing hip
    block only when no clean rectangle fits."""
    import cv2
    P = np.asarray(poly_w, np.float32)
    if len(P) < 3:
        return
    ap = cv2.approxPolyDP(P.reshape(-1, 1, 2), 1.2, True).reshape(-1, 2)
    if len(ap) >= 3:
        P = ap.astype(np.float32)
    area = abs(_poly_area_signed([tuple(p) for p in P]))
    if area < 6.0:
        return
    # SEPARATION: the segmentation partitions towns wall-to-wall, so
    # neighbouring buildings built to the outline interpenetrate (wing
    # inflation + raster rounding push ~0.5 m past it).  Shrink every
    # footprint inward before massing so adjacent houses stand apart.
    ring = [tuple(p) for p in P]
    if _poly_area_signed(ring) > 0:
        ring = ring[::-1]
    shr = _inset_ring(ring, 0.8)
    if shr is None:
        shr = _inset_ring(ring, 0.4)
    if shr is not None and abs(_poly_area_signed(shr)) >= 6.0:
        P = np.asarray(shr, np.float32)
    else:
        # miter inset degenerated — uniform scale about the centroid is a
        # crude inset that always succeeds
        cen = P.mean(0)
        P = (cen + (P - cen) * 0.88).astype(np.float32)
    rects = _decompose_rects(P)
    if not rects:
        _add_flat_block(acc_wall, acc_roof, P, hgt, y0)
        # door on the longest outline edge (outward for a CW ring)
        if acc_door is not None:
            ring2 = [tuple(p) for p in P]
            if _poly_area_signed(ring2) > 0:
                ring2 = ring2[::-1]
            best = max(range(len(ring2)), key=lambda i: np.hypot(
                ring2[(i + 1) % len(ring2)][0] - ring2[i][0],
                ring2[(i + 1) % len(ring2)][1] - ring2[i][1]))
            a, b = ring2[best], ring2[(best + 1) % len(ring2)]
            el = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            # EVERY building gets a door: even when the longest edge is
            # shorter than the old 2.5 m threshold, place it centred on
            # that edge (a slightly cramped door beats a doorless house)
            if el >= 0.9:
                ux, uz = (b[0] - a[0]) / el, (b[1] - a[1]) / el
                _add_door(acc_door, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2,
                          -uz, ux, y0 + 0.3)
                return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, -uz, ux)
        return None
    # normalise wing orientation here so door math matches the builder
    rects = [(cx, cz, w, d, ang) if w >= d else
             (cx, cz, d, w, ang + np.pi * 0.5)
             for cx, cz, w, d, ang in rects]
    rises = [float(np.clip(0.55 * d, 1.2, 4.5)) for _, _, _, d, _ in rects]
    eave = max(2.4, hgt - 0.5 * max(rises))
    for k, ((cx, cz, w, d, ang), rise) in enumerate(zip(rects, rises)):
        # hip whenever the wing is wide — a 15 m gable end is a huge
        # blank triangle; hips read as real architecture at that scale
        hip = (d > 10.0 or
               _det_rand(int(cz) + 7 * k, int(cx) + 3, 0.0, 1.0) > 0.55)
        # +0.3 m inflation hides the coplanar seams where wings touch
        _add_gable_house(acc_wall, acc_roof, (cx, cz), w + 0.3, d + 0.3,
                         ang, eave, rise, y0, hip=hip)
    if acc_door is None:
        return None
    # DOOR on the main wing's long wall, on the side facing the road
    # when we know where one is (door_dir), hash-picked otherwise
    cx, cz, w, d, ang = rects[0]
    ca, sa = float(np.cos(ang)), float(np.sin(ang))
    az = (-sa, ca)                                  # short-axis direction
    if door_dir is not None:
        side = 1.0 if az[0] * door_dir[0] + az[1] * door_dir[1] >= 0 else -1.0
    else:
        side = 1.0 if _det_rand(int(cx) + 5, int(cz) + 1, 0.0, 1.0) > 0.5 \
            else -1.0
    # SNAP the door to a window-bay BOUNDARY: with one centered window
    # per bay, the boundaries are the guaranteed blank-plaster strips
    # (bay centres are glass now), so the door never overlaps a window.
    wl = w + 0.3
    nbay = max(1, int(round(wl / 5.0)))
    bayw = wl / nbay
    shift = (_det_rand(int(cz) + 2, int(cx) + 8, 0.0, 1.0) - 0.5) * 0.4 * w
    # boundaries sit at (k - nbay/2) * bayw from the wall centre; keep k
    # on boundaries INSIDE the door-margin range (plain clamping after
    # snapping would land the door mid-bay, i.e. on glass)
    lim = max(0.0, w / 2 - 1.2)
    k = round(shift / bayw + nbay * 0.5)
    k_lo = int(np.ceil(-lim / bayw + nbay * 0.5))
    k_hi = int(np.floor(lim / bayw + nbay * 0.5))
    if k_lo <= k_hi:
        k = int(np.clip(k, k_lo, k_hi))
        shift = (k - nbay * 0.5) * bayw
    else:
        # wall too short for any boundary inside the margin (small house,
        # single bay): push the door as far off the bay-centre window as
        # the margin allows — a partly overlapped frame beats dead-centre
        shift = float(lim if shift >= 0.0 else -lim)
    dxy = (cx + ca * shift + side * az[0] * (d + 0.3) / 2,
           cz + sa * shift + side * az[1] * (d + 0.3) / 2)
    nx, nz = side * az[0], side * az[1]
    _add_door(acc_door, dxy[0], dxy[1], nx, nz, y0 + 0.3)
    return (dxy[0], dxy[1], nx, nz)


def _add_door(acc_door, cx, cz, nx, nz, ybase):
    """Entrance door: a framed door quad sitting 5 cm proud of the wall,
    2.15 m tall — the texture is mapped ONCE (not tiled)."""
    tx, tz = -nz, nx
    hw, o, h = 0.60, 0.05, 2.15
    p1 = (cx + tx * hw + nx * o, cz + tz * hw + nz * o)
    p0 = (cx - tx * hw + nx * o, cz - tz * hw + nz * o)
    v = [(p1[0], ybase, p1[1]), (p0[0], ybase, p0[1]),
         (p0[0], ybase + h, p0[1]), (p1[0], ybase + h, p1[1])]
    acc_door.add(v, [(nx, 0.0, nz)] * 4, [0, 1, 2, 0, 2, 3],
                 [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])


def add_fence_rect(acc_fence, corners, ground_w, gate, step=2.6, h=0.85):
    """Terrain-following picket fence around a garden rect (CW corners),
    with a gap for the gate nearest `gate` (world x,z) — the entrance
    stays reachable."""
    ring = []
    for i in range(4):
        a = np.asarray(corners[i], np.float64)
        b = np.asarray(corners[(i + 1) % 4], np.float64)
        seg = float(np.linalg.norm(b - a))
        nseg = max(1, int(round(seg / step)))
        for s in range(nseg):
            ring.append(a + (b - a) * (s / nseg))
    n = len(ring)
    gap_a = gap_b = None
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        mid = (a + b) * 0.5
        if gate is not None and float(np.hypot(mid[0] - gate[0],
                                               mid[1] - gate[1])) < 2.2:
            gap_a = a if gap_a is None else gap_a           # gate gap
            gap_b = b
            continue
        ya, yb = ground_w(a[0], a[1]), ground_w(b[0], b[1])
        el = float(np.linalg.norm(b - a))
        ux, uz = (b - a) / max(el, 1e-6)
        onx, onz = -uz, ux                                  # CW ⇒ outward
        v = [(a[0], ya - 0.2, a[1]), (b[0], yb - 0.2, b[1]),
             (b[0], yb + h, b[1]), (a[0], ya + h, a[1])]
        uvs = [(0.0, 1.0), (el / 2.2, 1.0), (el / 2.2, 0.0), (0.0, 0.0)]
        acc_fence.add(v, [(onx, 0.0, onz)] * 4, [0, 1, 2, 0, 2, 3], uvs)
    if gap_a is None:
        return
    # GATE in the gap: two posts and a lower swing panel, so the garden
    # entrance reads as a gate rather than a missing fence piece
    ga, gb = np.asarray(gap_a), np.asarray(gap_b)
    gd = gb - ga
    gl = float(np.linalg.norm(gd))
    if gl < 0.8:
        return
    ux, uz = gd / gl
    onx, onz = -uz, ux
    for p in (ga, gb):                                      # posts
        y = ground_w(p[0], p[1])
        pa = (p[0] - ux * 0.09, p[1] - uz * 0.09)
        pb = (p[0] + ux * 0.09, p[1] + uz * 0.09)
        v = [(pa[0], y - 0.2, pa[1]), (pb[0], y - 0.2, pb[1]),
             (pb[0], y + h + 0.22, pb[1]), (pa[0], y + h + 0.22, pa[1])]
        acc_fence.add(v, [(onx, 0.0, onz)] * 4, [0, 1, 2, 0, 2, 3],
                      [(0.0, 1.0), (0.18, 1.0), (0.18, 0.0), (0.0, 0.0)])
    pa = ga + gd * (0.12 / gl) if gl > 0.5 else ga          # panel
    pb = gb - gd * (0.12 / gl) if gl > 0.5 else gb
    ya, yb = ground_w(pa[0], pa[1]), ground_w(pb[0], pb[1])
    pl = float(np.linalg.norm(pb - pa))
    v = [(pa[0], ya + 0.06, pa[1]), (pb[0], yb + 0.06, pb[1]),
         (pb[0], yb + h * 0.8, pb[1]), (pa[0], ya + h * 0.8, pa[1])]
    acc_fence.add(v, [(onx, 0.0, onz)] * 4, [0, 1, 2, 0, 2, 3],
                  [(0.0, 1.0), (pl / 2.2, 1.0), (pl / 2.2, 0.0),
                   (0.0, 0.0)])


def build_road_mesh(acc_road, road_mask, ground_w, res=2.5, lift=0.06,
                    max_quads=150000):
    """Asphalt mesh CONNECTING the houses: a draped grid of quads over
    the segmentation's painted road network.  Nodes are shared and
    sampled once each (crack-free); the coverage field is smoothed
    before thresholding (softer outline), and BOUNDARY nodes drop to
    ground level so the road's edge tapers into the terrain instead of
    ending in a floating, aliased cliff."""
    import cv2
    from scipy import ndimage as _ndi
    frac = road_mask.astype(np.float32)
    while True:
        cells = max(8, int(WORLD_SIZE_M / res))
        f = cv2.resize(frac, (cells, cells), interpolation=cv2.INTER_AREA)
        on = _ndi.gaussian_filter(f, 1.0) > 0.5
        if int(on.sum()) <= max_quads:
            break
        res *= 1.25
    ys, xs = np.nonzero(on)
    if not len(ys):
        return 0
    # per-node count of adjacent road cells: <4 ⇒ boundary node
    cnt = np.zeros((cells + 1, cells + 1), np.int16)
    o = on.astype(np.int16)
    cnt[:-1, :-1] += o
    cnt[:-1, 1:] += o
    cnt[1:, :-1] += o
    cnt[1:, 1:] += o
    nodes = {}

    def node(ix, iy):
        key = (ix, iy)
        if key not in nodes:
            wx = ix * res - WORLD_SIZE_M * 0.5
            wz = iy * res - WORLD_SIZE_M * 0.5
            lft = lift if cnt[iy, ix] >= 4 else 0.0
            y = ground_w(wx, wz) + lft
            gx = (ground_w(wx + res, wz) - ground_w(wx - res, wz)) / (2 * res)
            gz = (ground_w(wx, wz + res) - ground_w(wx, wz - res)) / (2 * res)
            nrm = np.array([-gx, 1.0, -gz])
            nrm /= np.linalg.norm(nrm)
            nodes[key] = len(nodes)
            acc_road.pos.append((wx, y, wz))
            acc_road.nrm.append(tuple(nrm))
            acc_road.uv.append((wx / 7.0, wz / 7.0))
        return nodes[key]

    for iy, ix in zip(ys, xs):
        a = node(ix, iy)
        b = node(ix + 1, iy)
        c = node(ix + 1, iy + 1)
        d = node(ix, iy + 1)
        acc_road.idx.extend([a, b, c, a, c, d])
    return int(on.sum())


def _tube(acc, cx, cz, r, y0, y1, sides=5):
    v, n, uv = [], [], []
    for i in range(sides):
        t = 2 * np.pi * i / sides
        nx, nz = np.cos(t), np.sin(t)
        for y, vv in ((y0, (y1 - y0) / 2.0), (y1, 0.0)):
            v.append((cx + r * nx, y, cz + r * nz))
            n.append((nx, 0.0, nz))
            uv.append((t / np.pi, vv))
    f = []
    for i in range(sides):
        j = (i + 1) % sides
        f += [2 * i, 2 * j, 2 * j + 1, 2 * i, 2 * j + 1, 2 * i + 1]
    acc.add(v, n, f, uv)


def _cone_lp(acc, cx, cz, r, y0, h, sides=6):
    v = [(cx, y0 + h, cz)]
    n = [(0.0, 1.0, 0.0)]
    uv = [(0.5, 0.0)]
    for i in range(sides):
        t = 2 * np.pi * i / sides
        nx, nz = np.cos(t), np.sin(t)
        nl = float(np.hypot(h, r)) + 1e-6
        v.append((cx + r * nx, y0, cz + r * nz))
        n.append((nx * h / nl, r / nl, nz * h / nl))
        uv.append((t / np.pi, h / 2.5))
    f = []
    for i in range(sides):
        f += [0, 1 + i, 1 + (i + 1) % sides]
    acc.add(v, n, f, uv)


def _octa(acc, cx, cz, y0, r, h):
    ym = y0 + 0.38 * h
    mid = []
    for i in range(5):
        t = 2 * np.pi * i / 5
        mid.append((cx + r * np.cos(t), ym, cz + r * np.sin(t)))
    v = [(cx, y0 + h, cz), (cx, y0 - 0.15 * h, cz)] + mid
    n = ([(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)] +
         [((m[0] - cx) / r, 0.0, (m[2] - cz) / r) for m in mid])
    uv = [(0.5, 0.0), (0.5, 1.0)] + [(i / 2.5, 0.5) for i in range(5)]
    f = []
    for i in range(5):
        j = (i + 1) % 5
        f += [0, 2 + i, 2 + j, 1, 2 + j, 2 + i]
    acc.add(v, n, f, uv)


def _cone_jit(acc, cx, cz, r, y0, h, seed, sides=7):
    """Canopy tier with radial ring jitter — an organically ragged cone
    instead of a perfect lampshade."""
    v = [(cx, y0 + h, cz)]
    n = [(0.0, 1.0, 0.0)]
    uv = [(0.5, 0.0)]
    for i in range(sides):
        t = 2 * np.pi * i / sides
        rr = r * (0.78 + 0.44 * _det_rand(seed + i, seed * 3 + i, 0.0, 1.0))
        yy = y0 + h * 0.08 * (_det_rand(seed * 5 + i, seed + 2 * i,
                                        0.0, 1.0) - 0.5)
        nx, nz = np.cos(t), np.sin(t)
        nl = float(np.hypot(h, rr)) + 1e-6
        v.append((cx + rr * nx, yy, cz + rr * nz))
        n.append((nx * h / nl, rr / nl, nz * h / nl))
        uv.append((t / np.pi * 2.5, h / 1.1))   # finer foliage tiling
    f = []
    for i in range(sides):
        f += [0, 1 + i, 1 + (i + 1) % sides]
    acc.add(v, n, f, uv)


def _blob(acc, cx, cy, cz, rx, ry, rz, seed):
    """Jittered subdivided octahedron — a lumpy broadleaf crown."""
    base = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
            (0, 0, 1), (0, 0, -1)]
    faces8 = [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
              (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)]
    verts = list(base)
    mid = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            m = np.asarray(verts[a], np.float64) + np.asarray(verts[b],
                                                             np.float64)
            m /= np.linalg.norm(m) + 1e-9
            mid[key] = len(verts)
            verts.append(tuple(m))
        return mid[key]

    faces = []
    for a, b, c in faces8:                    # one subdivision -> 32 tris
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    v, n, uv = [], [], []
    for i, p in enumerate(verts):
        j = 0.8 + 0.4 * _det_rand(seed + i * 7, seed * 3 + i, 0.0, 1.0)
        v.append((cx + p[0] * rx * j, cy + p[1] * ry * j,
                  cz + p[2] * rz * j))
        n.append(p)
        uv.append((p[0] * 1.6 + 0.5, p[2] * 1.6 + 0.5))  # finer tiling
    acc.add(v, n, [i for t in faces for i in t], uv)


def add_tree(acc_trunk, acc_leaf, cx, cz, hgt, rad, y0, sx, sy):
    """Low-poly TREE with real-world species variety: ~40% spruce (three
    ragged tiers), ~30% pine (high crown, two narrow tiers on a tall
    bare trunk), ~30% broadleaf (lumpy multi-blob crown).  All variation
    comes from _det_rand on the source pixel coords (deterministic)."""
    kind = _det_rand(sx, sy, 0.0, 1.0)
    s = 0.65 + 0.75 * _det_rand(sy, sx, 0.0, 1.0)   # wide size spread
    hgt, rad = hgt * s, rad * s
    rt = float(np.clip(rad * 0.16, 0.2, 0.55))
    seed = sx * 131 + sy * 17
    # minimum trunk clearances keep small trees' canopies OFF the ground
    # even when terrain detail displacement raises the surface a little
    if kind < 0.4:                                   # spruce: 3 tiers
        th = max(0.18 * hgt, 1.0)
        _tube(acc_trunk, cx, cz, rt, y0 - 0.4, y0 + th + 0.3)
        _cone_jit(acc_leaf, cx, cz, rad, y0 + th, 0.48 * hgt, seed)
        _cone_jit(acc_leaf, cx, cz, rad * 0.74, y0 + th + 0.26 * hgt,
                  0.46 * hgt, seed + 3)
        _cone_jit(acc_leaf, cx, cz, rad * 0.46, y0 + th + 0.52 * hgt,
                  0.44 * hgt, seed + 6)
    elif kind < 0.7:                                 # pine: tall bare trunk
        th = max(0.45 * hgt, 1.5)
        _tube(acc_trunk, cx, cz, rt * 0.85, y0 - 0.4, y0 + th + 0.2)
        _cone_jit(acc_leaf, cx, cz, rad * 0.8, y0 + th, 0.42 * hgt,
                  seed + 1)
        _cone_jit(acc_leaf, cx, cz, rad * 0.52, y0 + th + 0.28 * hgt,
                  0.4 * hgt, seed + 4)
    else:                                            # broadleaf: blob crown
        th = max(0.38 * hgt, 1.2)
        _tube(acc_trunk, cx, cz, rt * 1.1, y0 - 0.4, y0 + th + 0.15)
        cy = y0 + th + 0.28 * hgt
        _blob(acc_leaf, cx, cy, cz, rad * 1.2, 0.34 * hgt, rad * 1.2, seed)
        ox = rad * 0.55 * (_det_rand(seed + 21, seed + 5, 0.0, 1.0) - 0.5)
        oz = rad * 0.55 * (_det_rand(seed + 9, seed + 33, 0.0, 1.0) - 0.5)
        _blob(acc_leaf, cx + ox, cy + 0.16 * hgt, cz + oz,
              rad * 0.8, 0.24 * hgt, rad * 0.8, seed + 50)


def _flat_shade(pos, idx, uv=None):
    """Explode shared-vertex geometry into per-face vertices carrying TRUE
    face normals, so proxies actually shade under the sun (the accumulators
    only store placeholder up-normals, which renders every wall exactly as
    bright as every roof — houses read as flat melted blobs).
    Winding of the source faces is arbitrary (footprint polygons arrive in
    either orientation), so the normal SIGN is fixed by convention instead:
    roof-ish faces (|ny| >= 0.5) point up; wall faces point AWAY from their
    own building's centroid.  Per-building membership is recovered from
    shared-vertex connected components — houses never share vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    tris = idx.reshape(-1, 3).astype(np.int64)
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

    nv = len(pos)
    e0 = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    e1 = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    adj = coo_matrix((np.ones(len(e0), np.int8), (e0, e1)), shape=(nv, nv))
    _, comp = connected_components(adj, directed=False)
    ncomp = int(comp.max()) + 1
    csum = np.zeros((ncomp, 3), np.float64)
    ccnt = np.zeros(ncomp, np.int64)
    np.add.at(csum, comp, pos)
    np.add.at(ccnt, comp, 1)
    cent = (csum / ccnt[:, None])[comp[tris[:, 0]]]      # per-tri centroid

    tcen = (a + b + c) / 3.0
    roof = np.abs(fn[:, 1]) >= 0.5
    # Walls flip only when CLEARLY inward (margin in metres): the new
    # builders emit correct outward windings, and a lone wall quad is its
    # own component whose centroid lies ON the wall plane (dot ≈ 0) — a
    # zero threshold there would flip on float noise.
    flip = np.where(roof, fn[:, 1] < 0.0,
                    (fn[:, 0] * (tcen[:, 0] - cent[:, 0]) +
                     fn[:, 2] * (tcen[:, 2] - cent[:, 2])) < -0.25)
    fn[flip] = -fn[flip]

    if uv is not None:
        uv = uv[tris].reshape(-1, 2).astype(np.float32)
    pos = pos[tris].reshape(-1, 3).astype(np.float32)
    nrm = np.repeat(fn, 3, axis=0).astype(np.float32)
    idx = np.arange(len(pos), dtype=np.uint32)
    return pos, nrm, idx, uv


def write_glb(groups, path):
    """groups: [(MeshAcc, (r, g, b) 0-255, name[, flat_shade]), ...] — ONE
    MATERIAL PER GROUP so each PCG class renders in its own debug colour
    (buildings vs trees) and mis-segmented blobs are identifiable at a
    glance.  Pass flat_shade=True for groups whose accumulators only carry
    placeholder normals (buildings): faces are exploded and given true
    per-face normals so walls/roofs shade correctly."""
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
    for grp in groups:
        acc, albedo, name = grp[0], grp[1], grp[2]
        flat = bool(grp[3]) if len(grp) > 3 else False
        if not acc.pos:
            continue
        pos = np.asarray(acc.pos, np.float32)
        nrm = np.asarray(acc.nrm, np.float32)
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
        idx = np.asarray(acc.idx, np.uint32)
        uv = (np.asarray(acc.uv, np.float32)
              if len(acc.uv) == len(acc.pos) and len(acc.pos) else None)
        if flat and len(idx):
            pos, nrm, idx, uv = _flat_shade(pos, idx, uv)
        # Double-sided by construction: emit every triangle in BOTH
        # windings.  (The reversed copies share the exploded vertices, so
        # both copies of a face carry the SAME corrected normal — coplanar
        # duplicates shade identically and cannot z-fight visibly.)
        rev = idx.reshape(-1, 3)[:, ::-1].reshape(-1)
        idx = np.concatenate([idx, rev]).astype(np.uint32)

        if uv is None:
            # Planar world-XZ UVs (NOT zeros): the cluster pipeline
            # computes tangents from UV deltas — degenerate UVs break it.
            uv = np.stack([pos[:, 0] / WORLD_SIZE_M + 0.5,
                           pos[:, 2] / WORLD_SIZE_M + 0.5],
                          1).astype(np.float32)

        # Embedded PBR set: the engine's drawable path expects a
        # baseColorTexture — factor-only materials import but don't
        # render.  Accepts a solid (r, g, b), a PIL.Image (albedo only),
        # or a {albedo, normal, mr} bundle from _tex_bundle for full
        # PBR (normal + metallicRoughness maps).
        mat_idx = len(materials)

        def _emit_tex(im):
            wb = io.BytesIO()
            im.save(wb, format="PNG")
            iv = view(pad4(wb.getvalue()), None)
            images.append({"bufferView": iv, "mimeType": "image/png"})
            textures.append({"sampler": 0, "source": len(images) - 1})
            return len(textures) - 1

        bundle = albedo if isinstance(albedo, dict) else {
            "albedo": (albedo if isinstance(albedo, Image.Image)
                       else Image.new("RGB", (4, 4), tuple(albedo)))}
        pbr = {"baseColorTexture": {"index": _emit_tex(bundle["albedo"])},
               "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
               # factors MULTIPLY the mr texture; metallic 0 stays safe
               # even on loaders that ignore the texture.
               "metallicFactor": 0.0,
               "roughnessFactor": 1.0 if "mr" in bundle else 0.9}
        mat = {"name": name, "pbrMetallicRoughness": pbr}
        if "mr" in bundle:
            pbr["metallicRoughnessTexture"] = {
                "index": _emit_tex(bundle["mr"])}
        if "normal" in bundle:
            mat["normalTexture"] = {"index": _emit_tex(bundle["normal"])}
        materials.append(mat)

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


def build_pcg_glb(color_path, height_path, out_glb, seg_path=None,
                  place_towns=True, place_trees=True):
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

    # Placement heights come from the RENDER MESH when it exists (the
    # sibling <stamp>.glb): the engine draws hmap*HEIGHT_AMP_M with NO
    # soil offset, so heightmap-derived bases floated ~SOIL_OFFSET_M
    # (5 m) above the visible ground — every tree hovered mid-air.
    ground_grid = None
    ter_glb = os.path.splitext(height_path)[0] + ".glb"
    if os.path.exists(ter_glb):
        try:
            ground_grid = _load_terrain_grid(ter_glb)
            print(f"[pcg] ground heights from render mesh: {ter_glb}")
        except Exception as e:                            # noqa: BLE001
            print(f"[pcg] terrain glb unreadable ({e}); heightmap fallback")

    def ground(px, py):
        if ground_grid is not None:
            return _grid_sample(ground_grid, *px_to_world(px, py))
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
        bld, bld_polys = [], []
        try:
            bld_polys = extract_building_footprints(
                rgb, roof, px_m, **_load_env_cfg())
        except Exception as e:                            # noqa: BLE001
            print(f"[pcg] footprint extractor unavailable ({e}); boxes")
        if not bld_polys:
            bld = extract_buildings_best(roof, px_m, rgb_u8=rgb)

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
        bld, bld_polys = [], []
        try:
            bld_polys = extract_building_footprints(
                rgb, roof, px_m, **_load_env_cfg())
        except Exception as e:                            # noqa: BLE001
            print(f"[pcg] footprint extractor unavailable ({e}); boxes")
        if not bld_polys:
            bld = extract_buildings_best(roof, px_m, rgb_u8=rgb)
        tree_block = masks["roof"]

    # Per-class accumulators — written as separate primitives with their
    # own DEBUG COLOUR material, so a glance at the rendered proxies
    # tells you what the segmentation classified each blob as:
    #   buildings = orange, trees = green.
    # ── Biome gate ────────────────────
    # The colour heuristics classify dark low-saturation BASALT as roofs
    # and any faint-green tint as canopy, so barren/treeless prompts
    # (volcanic, desert, snow, rock) get hallucinated towns and forests.
    # When the caller says the biome has no civilisation / no trees,
    # suppress those proxies outright rather than trusting the pixels.
    if not place_towns and (bld or locals().get("bld_polys")):
        print("[pcg] biome gate: suppressing building proxies (barren biome)")
        bld = []
        bld_polys = []
    # Variant accumulators — a deterministic position hash assigns each
    # building a wall/roof palette and a height jitter, and each tree a
    # foliage palette, so streets stop looking clone-stamped.
    acc_walls = [MeshAcc() for _ in _WALL_BASES]
    acc_roofs = [MeshAcc() for _ in _ROOF_BASES]
    acc_leafs = [MeshAcc() for _ in _LEAF_BASES]
    acc_trunk = MeshAcc()

    def _pick(cx, cy, salt, n):
        return min(int(_det_rand(int(cx) + salt, int(cy) + 2 * salt + 1,
                                 0.0, float(n))), n - 1)

    def _jitter_h(cx, cy, h):
        h = h * (0.85 + 0.35 * _det_rand(int(cx) + 3, int(cy) + 5,
                                         0.0, 1.0))
        if _det_rand(int(cx) + 9, int(cy) + 13, 0.0, 1.0) > 0.88:
            h *= 1.45                        # occasional taller landmark
        return h

    try:
        import cv2
    except ModuleNotFoundError:
        cv2 = None                                 # PIL fallback below
    acc_door, acc_fence, acc_road = MeshAcc(), MeshAcc(), MeshAcc()

    def ground_w(wx, wz):
        return ground((wx + WORLD_SIZE_M * 0.5) / px_m,
                      (wz + WORLD_SIZE_M * 0.5) / px_m)

    # direction field toward the nearest painted road (doors face it)
    _door_field = None
    if masks["road"].any():
        from scipy.ndimage import distance_transform_edt
        if cv2 is not None:
            _rs = cv2.resize(masks["road"].astype(np.uint8), (1024, 1024),
                             interpolation=cv2.INTER_AREA) > 0
        else:
            from PIL import Image as _Im
            _rs = np.asarray(_Im.fromarray(
                (masks["road"] * 255).astype(np.uint8)).resize(
                    (1024, 1024), _Im.BILINEAR)) > 127
        _rd, _ri = distance_transform_edt(~_rs, return_indices=True)
        _door_field = (_rd, _ri, H / 1024.0)

    def door_dir_at(pcx, pcy):
        if _door_field is None:
            return None
        rd, ri, sc = _door_field
        qx = min(int(pcx / sc), 1023)
        qy = min(int(pcy / sc), 1023)
        if rd[qy, qx] * sc * px_m > 60.0:      # no road within 60 m
            return None
        dx = float(ri[1][qy, qx] - qx)
        dz = float(ri[0][qy, qx] - qy)
        n = float(np.hypot(dx, dz))
        return (dx / n, dz / n) if n > 1e-6 else None

    bld_polys = locals().get("bld_polys") or []
    claim = None
    if bld_polys:
        n0 = len(bld_polys)
        bld_polys, claim = _dedupe_footprints(bld_polys, H, px_m)
        print(f"[pcg] footprint overlap resolution: {n0} -> "
              f"{len(bld_polys)} buildings")
    for cx, cy, w, d, ang, hgt in bld:
        wx, wz = px_to_world(cx, cy)
        ca, sa = np.cos(ang), np.sin(ang)
        pts = [(wx + u * ca - v * sa, wz + u * sa + v * ca)
               for u, v in ((-w / 2, -d / 2), (w / 2, -d / 2),
                            (w / 2, d / 2), (-w / 2, d / 2))]
        add_building(acc_walls[_pick(cx, cy, 17, len(_WALL_BASES))],
                     acc_roofs[_pick(cx, cy, 31, len(_ROOF_BASES))],
                     pts, _jitter_h(cx, cy, hgt), ground(cx, cy) - 0.6,
                     acc_door, door_dir_at(cx, cy))
    _garden_src = []
    for poly, phgt in bld_polys:
        pts = [px_to_world(px_, py_) for px_, py_ in poly]
        # SLOPE-AWARE base: min ground over the whole outline (a centre
        # sample let downhill corners float in mid-air), walls extended
        # 0.4 m into the ground.
        base = min(ground(px_, py_) for px_, py_ in poly) - 0.3
        pcx, pcy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        dinfo = add_building(
            acc_walls[_pick(pcx, pcy, 17, len(_WALL_BASES))],
            acc_roofs[_pick(pcx, pcy, 31, len(_ROOF_BASES))],
            pts, _jitter_h(pcx, pcy, phgt), base,
            acc_door, door_dir_at(pcx, pcy))
        _garden_src.append((poly, dinfo))
    # ── gardens: picket fence around house-scale lots with free ground ──
    ngarden = 0
    if claim is not None:
        for poly, dinfo in _garden_src:
            P = np.asarray(poly, np.float32)
            garea = float(cv2.contourArea(P)) * px_m * px_m
            pcx, pcy = float(P[:, 0].mean()), float(P[:, 1].mean())
            if not 25.0 <= garea <= 260.0:
                continue
            if _det_rand(int(pcx) + 41, int(pcy) + 7, 0.0, 1.0) < 0.2:
                continue
            Pw = (P * px_m - WORLD_SIZE_M * 0.5).astype(np.float32)
            (rcx, rcz), (rw, rh), rang = cv2.minAreaRect(
                Pw.reshape(-1, 1, 2))
            th = float(np.deg2rad(rang))
            ca_, sa_ = np.cos(th), np.sin(th)
            m = 4.2
            cors = [(rcx + u * ca_ - v * sa_, rcz + u * sa_ + v * ca_)
                    for u, v in ((-(rw / 2 + m), -(rh / 2 + m)),
                                 ((rw / 2 + m), -(rh / 2 + m)),
                                 ((rw / 2 + m), (rh / 2 + m)),
                                 (-(rw / 2 + m), (rh / 2 + m)))]
            if _poly_area_signed(cors) > 0:
                cors = cors[::-1]
            ok = True
            for i in range(4):
                a = np.asarray(cors[i]); b = np.asarray(cors[(i + 1) % 4])
                L = float(np.linalg.norm(b - a))
                for s in range(int(L / 1.5) + 1):
                    q = a + (b - a) * (s / max(int(L / 1.5) + 1, 1))
                    gx = int(q[0] + WORLD_SIZE_M * 0.5)
                    gz = int(q[1] + WORLD_SIZE_M * 0.5)
                    px_ = min(int((q[0] + WORLD_SIZE_M * 0.5) / px_m), H - 1)
                    py_ = min(int((q[1] + WORLD_SIZE_M * 0.5) / px_m), H - 1)
                    if (not (0 <= gx < claim.shape[1] and
                             0 <= gz < claim.shape[0]) or
                            claim[gz, gx] or
                            masks["road"][py_, px_] or
                            masks["water"][py_, px_]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            gate = ((dinfo[0] + dinfo[2] * m, dinfo[1] + dinfo[3] * m)
                    if dinfo else None)
            add_fence_rect(acc_fence, cors, ground_w, gate)
            ngarden += 1
    nroad = build_road_mesh(acc_road, masks["road"] & ~masks["water"],
                            ground_w)
    print(f"[pcg] doors: {len(acc_door.pos) // 4}  gardens: {ngarden}  "
          f"road quads: {nroad}")
    trees = extract_trees(masks["veg"], tree_block, px_m) if place_trees else []
    if not place_trees:
        print("[pcg] biome gate: suppressing tree proxies (treeless biome)")
    for px, py, hgt, rad in trees:
        wx, wz = px_to_world(px, py)
        add_tree(acc_trunk, acc_leafs[_pick(px, py, 53, len(_LEAF_BASES))],
                 wx, wz, hgt, rad, ground(px, py) - 0.05, int(px), int(py))
    groups = ([(a, _tex_bundle(f"wall{i}"), f"house_wall{i}", True)
               for i, a in enumerate(acc_walls)] +
              [(a, _tex_bundle(f"roof{i}"), f"house_roof{i}", True)
               for i, a in enumerate(acc_roofs)] +
              [(acc_trunk, _tex_bundle("bark"), "tree_trunk")] +
              [(a, _tex_bundle(f"leaf{i}"), f"tree_leaf{i}")
               for i, a in enumerate(acc_leafs)] +
              [(acc_door, _tex_bundle("door"), "house_door"),
               (acc_fence, _tex_bundle("fence"), "garden_fence"),
               (acc_road, _tex_bundle("road"), "roadway")])
    write_glb([grp for grp in groups if grp[0].pos], out_glb)
    if bld_polys:
        fp = os.path.splitext(out_glb)[0] + "_footprints.json"
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump({"px_m": px_m, "world_size_m": WORLD_SIZE_M,
                       "polys": [p.tolist() for p, _ in bld_polys],
                       "heights": [h for _, h in bld_polys],
                       # tree placements too, so offline tooling can
                       # rebuild the proxy GLB without re-running
                       # segmentation
                       "trees": [[int(px), int(py), float(h), float(r)]
                                 for px, py, h, r in trees]}, fh)
        print(f"[pcg] footprints sidecar -> {fp}")
    print(f"[pcg] {len(bld) + len(bld_polys)} buildings (orange) + {len(trees)} trees "
          f"(green) [{'layout masks' if use_seg else 'color heuristics'}]"
          f" -> {out_glb} "
          f"({sum(len(a.pos) for a, *_ in groups)} verts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", required=True)
    ap.add_argument("--height", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default=None,
                    help="layout seg PNG (<map>_seg.png) — exact class "
                         "masks from the layout-first pipeline; when "
                         "given, colour heuristics are bypassed")
    ap.add_argument("--no-towns", action="store_true",
                    help="suppress building proxies (barren biome)")
    ap.add_argument("--no-trees", action="store_true",
                    help="suppress tree proxies (treeless biome)")
    args = ap.parse_args()
    build_pcg_glb(args.color, args.height, args.out, seg_path=args.seg,
                  place_towns=not args.no_towns,
                  place_trees=not args.no_trees)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
# (touched to force workspace-mount resync 2)
