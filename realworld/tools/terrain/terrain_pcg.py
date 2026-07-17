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
    acc_bld, acc_tree = MeshAcc(), MeshAcc()
    bld_polys = locals().get("bld_polys") or []
    for cx, cy, w, d, ang, hgt in bld:
        wx, wz = px_to_world(cx, cy)
        add_box(acc_bld, wx, wz, w, d, ang, hgt, ground(cx, cy) - 0.3)
    for poly, phgt in bld_polys:
        pcx, pcy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        pts = [px_to_world(px_, py_) for px_, py_ in poly]
        # walls take ~60% of the height budget; the hip roof the rest.
        eave = max(2.5, 0.6 * phgt)
        add_house(acc_bld, pts, eave, ground(pcx, pcy) - 0.3,
                  pitch=0.8, cap=max(2.0, 0.4 * phgt))
    trees = extract_trees(masks["veg"], tree_block, px_m) if place_trees else []
    if not place_trees:
        print("[pcg] biome gate: suppressing tree proxies (treeless biome)")
    for px, py, hgt, rad in trees:
        wx, wz = px_to_world(px, py)
        add_cone(acc_tree, wx, wz, hgt, rad, ground(px, py) - 0.2)
    write_glb([(acc_bld,  (255,  96,  32), "proxy_building"),
               (acc_tree, ( 48, 208,  80), "proxy_tree")],
              out_glb)
    if bld_polys:
        fp = os.path.splitext(out_glb)[0] + "_footprints.json"
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump({"px_m": px_m, "world_size_m": WORLD_SIZE_M,
                       "polys": [p.tolist() for p, _ in bld_polys],
                       "heights": [h for _, h in bld_polys]}, fh)
        print(f"[pcg] footprints sidecar -> {fp}")
    print(f"[pcg] {len(bld) + len(bld_polys)} buildings (orange) + {len(trees)} trees "
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
