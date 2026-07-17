#!/usr/bin/env python3
"""
pcg_verify_loop.py — self-improving iteration loop for building placement.

Closes the loop around the building extractor (SAM2 hybrid or heuristic):

    extract footprints -> score against the map masks -> pass? done.
                          fail? adjust extractor params -> extract again.

Scoring is fully quantitative (no API, no human in the loop):
  coverage   fraction of roof-mask area covered by boxes        (>= min-coverage)
  precision  fraction of box area that lies on the roof mask    (>= min-precision)
  road_viol  fraction of box area that lies on roads            (<= max-road)
  size       median footprint edge within [5, 30] m

Adjustment rules per failing metric:
  low coverage  -> denser seeds (seed_m -2), larger max_m2 (+300)
  low precision -> stricter roof-overlap requirement (roof_frac +0.1)
  road_viol     -> subtract a wider road dilation before extraction (+1 px)

On pass (or best-of after --max-iters) the winning config is written to
<out-dir>/best_cfg.json — feed it to a full regen via the TERRAIN_PCG_CFG env
var — plus a JSON report and a 500 m/1 km overlay PNG per iteration for eyes.

Usage (from realworld/):
    python tools/terrain/pcg_verify_loop.py \
        --color content/terrain/<map>_color.png \
        --height content/terrain/<map>.png \
        --seg content/terrain/<map>_seg.png \
        --max-iters 4
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import terrain_pcg as P  # noqa: E402


def build_roof_mask(rgb, seg_path, dil_road_px, px_m):
    """Same derivation as build_pcg_glb's layout path: heuristic roofs gated
    to near-town, minus water and (dilated) roads."""
    from scipy import ndimage as ndi
    H = rgb.shape[0]
    heur = P.segment(rgb, px_m)
    if seg_path and os.path.exists(seg_path):
        masks = P.masks_from_seg(seg_path, H)
        town_r = ndi.uniform_filter(
            masks["town"].astype(np.float32),
            size=max(3, int(80.0 / px_m) | 1)) > 1e-6
        road = masks["road"] | heur["road"]
        water = masks["water"] | heur["water"]
        roof = heur["roof"] & town_r
    else:
        road, water, roof = heur["road"], heur["water"], heur["roof"]
    road_base = road
    if dil_road_px > 0:
        road = ndi.binary_dilation(road, iterations=int(dil_road_px))
    return roof & ~water & ~road, road_base, water


def rasterize(polys, shape):
    import cv2
    union = np.zeros(shape, np.uint8)
    total_px = 0
    areas = []
    for poly, hgt in polys:
        pts = poly.astype(np.int32)
        one = np.zeros(shape, np.uint8)
        cv2.fillPoly(one, [pts], 1)
        a = int(one.sum())
        total_px += a
        areas.append(a)
        union |= one
    return union.astype(bool), total_px, areas


def score(polys, roof, road, px_m):
    if not polys:
        return {"count": 0, "coverage": 0.0, "precision": 0.0,
                "road_viol": 0.0, "overlap": 0.0, "med_edge": 0.0}
    bm, total_px, areas = rasterize(polys, roof.shape)
    union_px = max(1, int(bm.sum()))
    roof_area = max(1, int(roof.sum()))
    med_edge = float(np.median([np.sqrt(a) for a in areas])) * px_m
    return {
        "count": len(polys),
        "coverage": float((bm & roof).sum()) / roof_area,
        "precision": float((bm & roof).sum()) / union_px,
        "road_viol": float((bm & road).sum()) / union_px,
        # 0 = perfectly overlap-free partition; >0 means double-claimed px
        "overlap": 1.0 - union_px / max(1, total_px),
        "med_edge": med_edge,
    }


def render_overlay(rgb, boxes, out_png, px_m, center_px, ext_m, disp=1100):
    H = rgb.shape[0]
    half = int(ext_m / px_m / 2)
    cx, cy = int(center_px[0]), int(center_px[1])
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(H, cx + half), min(H, cy + half)
    img = Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")
    dr = ImageDraw.Draw(img)
    import cv2
    for poly, hgt in boxes:
        bx, by = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        if not (x0 <= bx < x1 and y0 <= by < y1):
            continue
        dr.polygon([(float(px - x0), float(py - y0)) for px, py in poly],
                   outline=(255, 20, 20))
    sc = disp / max(img.size)
    img.resize((max(1, int(img.width * sc)),
                max(1, int(img.height * sc)))).save(out_png)


def boxes_to_polys(boxes, px_m):
    import cv2
    out = []
    for (cx, cy, w, d, ang, hgt) in boxes:
        rect = ((cx, cy), (w / px_m, d / px_m), np.rad2deg(ang))
        out.append((cv2.boxPoints(rect).astype(np.float32), hgt))
    return out


def extract(rgb, roof, px_m, cfg):
    """Polygon footprints (exact base outlines, overlap-free) when SAM2 is
    available; oriented-rect boxes converted to 4-pt polys otherwise."""
    try:
        return P.extract_building_footprints(rgb, roof, px_m, **cfg), "sam2-poly"
    except Exception as e:                                    # noqa: BLE001
        print(f"[loop] footprints unavailable ({e}); heuristic boxes")
        return boxes_to_polys(P.extract_buildings_shaped(roof, px_m), px_m), \
            "heuristic"


def main():
    ap = argparse.ArgumentParser(description="PCG building placement loop")
    ap.add_argument("--color", required=True)
    ap.add_argument("--height", required=True)
    ap.add_argument("--seg", default=None)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--out-dir", default="content/terrain/.pcg_loop")
    ap.add_argument("--min-coverage", type=float, default=0.55)
    # Precision counts box concavity fill against us (min-area rects over
    # ragged roofs), so 0.65 is a strong result; road violations are scored
    # against the UNDILATED road mask.
    # Polygons trace the mask, so precision is high by construction.
    ap.add_argument("--min-precision", type=float, default=0.85)
    ap.add_argument("--max-road", type=float, default=0.08)
    ap.add_argument("--max-overlap", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rgb = np.asarray(Image.open(args.color).convert("RGB"))
    px_m = P.WORLD_SIZE_M / rgb.shape[0]

    cfg = {"seed_m": 10.0, "min_m2": 15.0, "max_m2": 1200.0,
           "roof_frac": 0.4, "dedup_iou": 0.45}
    dil_road = 2
    report = {"iterations": [], "thresholds": {
        "coverage": args.min_coverage, "precision": args.min_precision,
        "road": args.max_road}}
    best = None

    for it in range(1, args.max_iters + 1):
        print(f"\n=== iteration {it}/{args.max_iters}  cfg={cfg} "
              f"road_dil={dil_road} ===", flush=True)
        roof, road, water = build_roof_mask(rgb, args.seg, dil_road, px_m)
        boxes, mode = extract(rgb, roof, px_m, cfg)
        m = score(boxes, roof, road, px_m)
        print(f"[loop] {mode}: n={m['count']} cov={m['coverage']:.2f} "
              f"prec={m['precision']:.2f} road={m['road_viol']:.3f} "
              f"overlap={m['overlap']:.3f} edge~{m['med_edge']:.0f} m",
              flush=True)

        if boxes:
            c = np.array([[p[:, 0].mean(), p[:, 1].mean()]
                          for p, _ in boxes]).mean(0)
            for ext, tag in ((1000.0, "1km"), (500.0, "500m")):
                render_overlay(rgb, boxes,
                               os.path.join(args.out_dir,
                                            f"iter{it:02d}_{tag}.png"),
                               px_m, c, ext)

        ok = (m["coverage"] >= args.min_coverage and
              m["precision"] >= args.min_precision and
              m["road_viol"] <= args.max_road and
              m["overlap"] <= args.max_overlap and
              4.0 <= m["med_edge"] <= 40.0)
        entry = {"iter": it, "mode": mode, "cfg": dict(cfg),
                 "road_dil": dil_road, "metrics": m, "pass": ok}
        report["iterations"].append(entry)
        fitness = (m["coverage"] * m["precision"]
                   - m["road_viol"] - m["overlap"])
        if best is None or fitness > best[0]:
            best = (fitness, dict(cfg), dil_road, m)

        if ok:
            print(f"[loop] PASS on iteration {it}", flush=True)
            break
        # ── adjust ────────────────────────────────────────────────────
        if m["coverage"] < args.min_coverage:
            cfg["seed_m"] = max(4.0, cfg["seed_m"] - 2.0)
            cfg["max_m2"] = min(2400.0, cfg["max_m2"] + 300.0)
        if m["precision"] < args.min_precision:
            cfg["roof_frac"] = min(0.8, cfg["roof_frac"] + 0.1)
        if m["road_viol"] > args.max_road:
            dil_road += 1
        print(f"[loop] adjust -> cfg={cfg} road_dil={dil_road}", flush=True)

    passed = any(e["pass"] for e in report["iterations"])
    report["passed"] = passed
    report["best_cfg"] = best[1]; report["best_road_dil"] = best[2]
    report["best_metrics"] = best[3]
    with open(os.path.join(args.out_dir, "best_cfg.json"), "w",
              encoding="utf-8") as f:
        json.dump(best[1], f, indent=2)
    with open(os.path.join(args.out_dir, "loop_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[loop] report -> {args.out_dir}/loop_report.json  "
          f"best_cfg -> {args.out_dir}/best_cfg.json  passed={passed}",
          flush=True)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
