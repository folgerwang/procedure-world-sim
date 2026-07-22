#!/usr/bin/env python3
"""
terrain_from_text.py — text → heightmap → terrain pipeline.

Stage 1  TEXT → IMAGE   FLUX.2-klein (tools/flux/flux_generate.py) renders a
                        top-down grayscale relief image from the user's
                        description (prompt is wrapped in heightmap-specific
                        conditioning language).
Stage 2  IMAGE → TERRAIN  A torch-based conversion model turns the raw
                        diffusion image into a *plausible terrain* heightfield:
                          - grayscale extraction + histogram shaping
                          - GPU hydraulic erosion (iterative rain/flow/deposit)
                          - thermal erosion (talus slope relaxation)
                          - detail-preserving blend of the eroded base with
                            the original high-frequency relief
                        The stage is a self-contained module with a
                        `refine(heightfield_tensor) -> tensor` interface, so a
                        LEARNED erosion/refinement network can replace the
                        simulation later without touching the pipeline.
Stage 3  INSTALL        Writes a 16-bit single-channel PNG compatible with the
                        engine's terrain heightmap (assets/map.png, 2048×2048,
                        mode I;16).  --install swaps it in with a backup.

NOTE (engine): the tile SHADERS sample assets/map.png, but the CPU-side
terrainMap() used for player physics is analytic FBM — installing a generated
map changes visuals only until the engine-side sync (CPU sampling of the same
map) is wired.  See the session notes / TODO in terrain.cpp.

Usage:
    python terrain_from_text.py --prompt "jagged wuxia mountain valley with a
        river canyon" --out content/terrain/wuxia_valley.png
    python terrain_from_text.py --prompt "..." --install       # replace assets/map.png
    python terrain_from_text.py --from-image raw.png --out h.png   # skip FLUX
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
FLUX      = os.path.join(REPO_ROOT, "tools", "flux", "flux_generate.py")
MAP_PNG   = os.path.join(REPO_ROOT, "assets", "map.png")

# Terrain map footprint — MUST match kTerrainMapMeters in
# src/sim_engine/shaders/global_definition.glsl.h (decoupled from the
# 32 km world; 8192 px over 4096 m = 0.5 m/texel).  White = 2000 m.
WORLD_SIZE_M   = 4096.0
HEIGHT_AMP_M   = 250.0   # matches kTerrainHeightAmpMeters
# Detail tiling (terrain_detail_worker.py): 16x16 detail tiles of 2 km,
# each 2048^2 at 1 m/texel.
DETAIL_TILE_M   = 2048.0
DETAIL_TILE_RES = 2048


# Prompts whose biome has no settlements / no trees. The colour-heuristic
# segmentation hallucinates roofs from dark basalt and canopy from any faint
# green, so we gate the PCG proxies by intent rather than trusting pixels.
_NO_TOWN_WORDS = (
    "volcanic", "volcano", "caldera", "lava", "basalt", "magma",
    "desert", "dune", "dunes", "sahara", "canyon", "mesa", "badland",
    "badlands", "arctic", "tundra", "glacier", "glacial", "iceberg",
    "ice sheet", "snowfield", "barren", "wasteland", "wilderness",
    "uninhabited", "moon", "lunar", "mars", "martian", "alien",
    "rocky highland", "bare rock", "scree", "talus",
)
_NO_TREE_WORDS = (
    "volcanic", "volcano", "caldera", "lava", "basalt", "magma",
    "desert", "dune", "dunes", "sahara", "arctic", "tundra", "glacier",
    "glacial", "iceberg", "ice sheet", "snowfield", "snowy", "snow",
    "barren", "wasteland", "moon", "lunar", "mars", "martian",
    "bare rock", "scree", "talus", "sand",
)
# Explicit vegetation words override the treeless default (e.g. "snowy alpine
# forest" should still get trees).
_HAS_TREE_WORDS = (
    "forest", "woodland", "jungle", "rainforest", "taiga", "grove",
    "wooded", "orchard", "trees", "canopy",
)


def _biome_pcg_flags(prompt):
    """Infer (place_towns, place_trees) from the prompt text so barren /
    treeless biomes don't get bogus building and tree proxies."""
    p = prompt.lower()
    place_towns = not any(w in p for w in _NO_TOWN_WORDS)
    place_trees = not any(w in p for w in _NO_TREE_WORDS)
    if any(w in p for w in _HAS_TREE_WORDS):
        place_trees = True
    return place_towns, place_trees

# Prompt scaffold: FLUX responds well to explicit "heightmap" framing plus
# photographic-DEM vocabulary.  The user's text slots into the middle.
# The view language is deliberately redundant (satellite / nadir / straight
# down / orthographic / no horizon) — diffusion models drift toward scenic
# perspective renders of "mountains" without it, and a perspective image is
# useless as a heightfield.
PROMPT_TEMPLATE = (
    "satellite view terrain heightmap seen directly from above, "
    "nadir aerial view, straight down 90 degrees overhead, "
    "orthographic map projection, "
    # Regional-scale conditioning: the map spans the WHOLE 32 km world,
    # so the content must be drawn at wide-area scale — whole mountain
    # ranges and valley systems, not a village-photo crop.  Without
    # this, FLUX draws ~2 km of content and the engine stretches it
    # 16x (houses become 300 m blobs).
    "digital elevation model covering a 4 kilometer wide area, "
    "complete valley and ridge systems in frame, {user_prompt}, "
    "grayscale digital elevation model, white is high elevation, "
    "black is low elevation, rivers and lakes are the DARKEST lowest "
    "channels, smooth gradients, satellite DEM data, "
    "flat 2D map, no perspective, no horizon, no sky, no shading, "
    "no colors, no text, no borders, no vignette"
)


# Colour satellite pass: the finished heightmap rides along as FLUX.2's
# native reference-image conditioning, so the colour map's layout matches
# the heightfield — ridges get rock/snow, valleys vegetation, channels
# water — instead of being an unrelated pretty picture.
COLOR_PROMPT_TEMPLATE = (
    "photorealistic full-color satellite imagery of the exact same "
    "terrain as the reference heightmap, matching its layout precisely, "
    # Same regional-scale conditioning as the heightmap pass: 30 km of
    # ground in frame, like a Landsat/Sentinel tile — towns are tiny
    # pixel clusters, individual buildings and trees are NOT resolvable.
    "aerial orthophoto covering a 4 kilometer wide area, {user_prompt}, "
    "nadir aerial orthophoto seen straight down, "
    "natural earth colors, vegetation in valleys, exposed rock and snow "
    "on high peaks, water in the lowest channels, "
    "no perspective, no horizon, no sky, no clouds, no text, no borders"
)


# ── Layout-first generation: land-cover map → masks + height + albedo ──────
# Instead of guessing classes from satellite-photo colours (heuristics that
# confuse rivers with roofs and scrub with towns), FLUX renders a FLAT-COLOUR
# land-cover map in a fixed palette.  Nearest-palette quantization then gives
# EXACT class masks; the heightfield and albedo derive from those masks, so
# heightmap, segmentation, albedo and PCG placement agree by construction.
SEG_CLASSES = [
    #  name      palette RGB        base elev  relief amp   (0..1 of range)
    ("water",  ( 30,  60, 200)),
    ("grass",  (120, 200,  80)),
    ("forest", ( 20, 110,  40)),
    ("town",   (220,  60,  40)),
    ("road",   ( 70,  70,  70)),
    ("rock",   (150, 115,  85)),
    ("snow",   (245, 245, 245)),
    ("sand",   (215, 185, 130)),
]
SEG_BASE_ELEV = {"water": 0.04, "grass": 0.16, "forest": 0.22,
                 "town": 0.14, "road": 0.15, "rock": 0.55,
                 "snow": 0.75, "sand": 0.10}
SEG_RELIEF_AMP = {"water": 0.01, "grass": 0.06, "forest": 0.10,
                  "town": 0.02, "road": 0.02, "rock": 0.45,
                  "snow": 0.35, "sand": 0.04}

# NOTE style vocabulary: "cartographic game world map" made FLUX draw an
# ANTIQUE illustrated map — brown houses on tan paper, hatched mountains —
# and the palette quantizer classified every house as rock/sand (towns:
# 0%).  The wording below pushes toward a GIS classification raster with
# saturated primary colours instead.
SEG_PROMPT_TEMPLATE = (
    "GIS land cover classification raster, semantic segmentation map, "
    "computer game minimap in flat saturated primary colors, "
    "top-down orthographic view of a 4 kilometer wide region, "
    "{user_prompt}, "
    "every land cover class painted in its pure signal color: "
    "BRIGHT PURE BLUE rivers and lakes, LIGHT GREEN grassland and "
    "fields, DARK GREEN forest and woodland, BRIGHT PURE RED town and "
    "village building blocks, DARK GRAY roads and streets connecting "
    # NOTE: never say "mountains"/"peaks"/"cliffs" — those words summon
    # the map-icon archetype: SIDE-PROFILE hatched mountain drawings.
    # Highland is just another flat region class seen from above.
    "the towns, large irregular BROWN regions of rocky highland "
    "painted as flat amoeba-shaped patches, WHITE patches in the "
    "middle of the largest brown regions, TAN sand and dirt, "
    "every region a flat solid-colored patch seen straight down from "
    "above, hard edges, solid color fills, no outlines, no gradients, "
    "no shading, no texture, no hatching, no ridgelines, no 3D relief, "
    "no side view, no text, no labels, "
    "no borders, no legend, no paper style, not an antique map, "
    "flat 2D digital classification image"
)

COLOR_FROM_LAYOUT_TEMPLATE = (
    "photorealistic full-color satellite aerial orthophoto of the exact "
    "same region as the reference land cover map, matching its layout "
    "precisely: blue areas are water, dark green areas are forest, "
    "light green areas are grassland, red areas are towns with dense "
    "buildings, gray lines are roads, brown areas are bare rock "
    "mountains, white areas are snow, "
    "covering a 4 kilometer wide area, {user_prompt}, "
    "nadir view seen straight down, orthographic map projection, "
    "natural earth colors, no perspective, no horizon, no sky, "
    "no clouds, no text, no borders"
)


def quantize_to_classes(png_path: str, size: int):
    """Classification of the FLUX layout render → uint8 class-index map
    [size, size].  HSV RULES first (robust to FLUX's colour drift —
    'red' towns come out brick, 'blue' water comes out steel...), with
    nearest-palette distance only as the tiebreak for pixels no rule
    claims.  Majority-vote smoothing removes border speckle."""
    from scipy import ndimage as ndi
    im = Image.open(png_path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.NEAREST)
    rgbf = np.asarray(im, dtype=np.float32) / 255.0
    r, g, b = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    v = rgbf.max(-1)
    mn = rgbf.min(-1)
    c = v - mn
    s = c / (v + 1e-6)
    # Hue in degrees (0..360); undefined (c≈0) stays 0 but s gates it.
    # cs: divide-safe chroma — the full RHS is evaluated BEFORE the mask
    # indexes it, so a raw `/ c` warns on every gray (c == 0) pixel even
    # though those lanes are discarded.
    hue = np.zeros_like(v)
    cs = np.where(c > 1e-6, c, 1.0)
    m = (c > 1e-6) & (v == r)
    hue[m] = (60.0 * ((g - b) / cs) % 360.0)[m]
    m = (c > 1e-6) & (v == g)
    hue[m] = (60.0 * ((b - r) / cs) + 120.0)[m]
    m = (c > 1e-6) & (v == b)
    hue[m] = (60.0 * ((r - g) / cs) + 240.0)[m]

    names = [n for n, _ in SEG_CLASSES]
    idx = {n: i for i, n in enumerate(names)}
    cls = np.full(v.shape, 255, np.uint8)          # 255 = unclaimed

    def put(mask, name):
        cls[(cls == 255) & mask] = idx[name]

    put((v > 0.75) & (s < 0.18), "snow")
    put((hue >= 185) & (hue <= 265) & (s > 0.2) & (v > 0.15), "water")
    put((hue >= 65) & (hue <= 175) & (s > 0.2) & (v < 0.45), "forest")
    put((hue >= 60) & (hue <= 175) & (s > 0.2), "grass")
    put(((hue >= 335) | (hue <= 22)) & (s > 0.35) & (v > 0.25), "town")
    put((s < 0.22) & (v < 0.55), "road")           # dark achromatic
    # v < 0.65 (not 0.55): the canonical rock palette colour itself has
    # v≈0.59 — with a lower cut it fell through to "sand" and red-noise
    # speckle inside mountains became disconnected tiny "town" blobs.
    put((hue > 22) & (hue < 55) & (v < 0.65), "rock")   # brown
    put((hue >= 22) & (hue <= 65), "sand")              # tan remainder
    # Unclaimed pixels: nearest palette colour.
    if (cls == 255).any():
        a16 = np.asarray(im, dtype=np.int32)
        pal = np.array([col for _, col in SEG_CLASSES], dtype=np.int32)
        un = cls == 255
        d = ((a16[un][:, None, :] - pal[None]) ** 2).sum(-1)
        cls[un] = d.argmin(-1).astype(np.uint8)

    # ── Warm-brown disambiguation: HOUSES vs MOUNTAINS ──────────────
    # FLUX's stylized layouts paint houses AND mountains in the same
    # warm-brown family, so hue rules alone put them in one class.
    # Per warm (rock ∪ town) component, decide by STRUCTURE (rule set
    # validated against a real antique-style generation):
    #   • ≥ 20000 m²                        → rock, unconditionally
    #     (mountain range — hatch strokes inside classify as "road",
    #     so road signals must never rescue a big component);
    #   • explicitly red-painted (≥50 %)    → town (GIS-style blocks);
    #   • else town IFF road-laced (streets within ~40 m), NOT hugging
    #     a mountain mass (kills scree dots + hatch fragments), and
    #     ≥ ~7 m wide via distance transform (kills drawn boundary
    #     LINES, which survive erosion but have no interior).
    px_m = WORLD_SIZE_M / size
    warm = (cls == idx["rock"]) | (cls == idx["town"])
    if warm.any():
        red0 = (cls == idx["town"]).astype(np.float32)  # pre-reclass
        lab, ncomp = ndi.label(warm)
        if ncomp > 0:
            index = np.arange(1, ncomp + 1)
            areas = ndi.sum(warm, lab, index) * px_m * px_m
            red_frac = ndi.mean(red0, lab, index)
            bigrock = np.isin(lab, index[areas >= 20000.0])
            road_d = ndi.uniform_filter(
                (cls == idx["road"]).astype(np.float32),
                max(3, int(80.0 / px_m) | 1))
            rockm_d = ndi.uniform_filter(
                bigrock.astype(np.float32),
                max(3, int(120.0 / px_m) | 1))
            road_c = ndi.mean(road_d, lab, index)
            rockm_c = ndi.mean(rockm_d, lab, index)
            edt = ndi.distance_transform_edt(warm) * px_m
            width_c = ndi.maximum(edt, lab, index)
            # Elongation ≈ length/width (area over squared width):
            # ~1 for a house rect, ~25 for a drawn boundary STRIP that
            # is wide enough to pass the width test.
            elong_c = areas / np.maximum((2.0 * width_c) ** 2, 1.0)
            to_town = (red_frac >= 0.5) | (
                (areas < 20000.0) & (road_c > 0.04) &
                (rockm_c < 0.15) & (width_c >= 3.5) &
                (elong_c < 8.0))
            lut = np.empty(ncomp + 1, np.uint8)
            lut[0] = 0                                  # unused (bg)
            lut[1:] = np.where(to_town, idx["town"], idx["rock"])
            cls[warm] = lut[lab[warm]]

    # Majority-vote smoothing: per-class box score, argmax — kills
    # single-texel speckle without the label-averaging bug a plain
    # median on indices would have.
    score = np.zeros((len(SEG_CLASSES),) + cls.shape, np.float32)
    for i in range(len(SEG_CLASSES)):
        score[i] = ndi.uniform_filter((cls == i).astype(np.float32), 5)
    cls = score.argmax(0).astype(np.uint8)
    return cls


def seg_color_agreement(cls_map, color_png: str, names) -> float:
    """How faithfully a colour render follows the segmentation: mean
    per-class fraction of seg pixels whose colour looks like that class.
    Drives the retry loop — the PCG places from the seg, so a colour
    pass that drifts puts buildings visibly off the painted houses."""
    small = np.asarray(Image.fromarray(cls_map).resize((256, 256),
                                                       Image.NEAREST))
    a = np.asarray(Image.open(color_png).convert("RGB").resize((256, 256)),
                   dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    pred = {
        "water":  (b > r * 1.10) & (b >= g * 0.90),
        "forest": (g > r) & (g > b) & (mx < 0.45),
        "grass":  (g > r) & (g > b) & (mx >= 0.45),
        "town":   ((r > g * 1.20) & (r > b * 1.20)) |
                  ((sat < 0.20) & (mx > 0.30) & (mx < 0.80)),
        "road":   (sat < 0.18) & (mx > 0.30) & (mx < 0.85),
        "rock":   (sat < 0.45) & (r >= g) & (mx > 0.25),
        "snow":   (mx > 0.75) & (sat < 0.15),
    }
    scores = []
    for i, n in enumerate(names):
        m = small == i
        if float(m.mean()) < 0.005 or n not in pred:
            continue
        scores.append(float((pred[n] & m).sum()) / float(m.sum()))
    return float(np.mean(scores)) if scores else 0.0


def save_seg_png(cls, path: str):
    """Class-index map → indexed-palette PNG (exact colours, tiny file)."""
    pal_img = Image.fromarray(cls, mode="P")
    flat = []
    for _, c in SEG_CLASSES:
        flat += list(c)
    pal_img.putpalette(flat + [0] * (768 - len(flat)))
    pal_img.save(path)


def height_from_classes(cls, size: int, seed: int):
    """Heightfield synthesised FROM the layout: per-class base elevation
    + fractal relief whose amplitude is also class-driven (mountains
    rough, towns/water near-flat).  Class borders are feathered so
    region edges don't ring as cliffs.  Aligned with the segmentation
    by construction."""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [n for n, _ in SEG_CLASSES]
    base = np.zeros(cls.shape, np.float32)
    amp = np.zeros(cls.shape, np.float32)
    for i, n in enumerate(names):
        m = cls == i
        base[m] = SEG_BASE_ELEV[n]
        amp[m] = SEG_RELIEF_AMP[n]
    base_t = torch.from_numpy(base).to(dev)[None, None]
    amp_t = torch.from_numpy(amp).to(dev)[None, None]
    # Feather class borders over ~60 m.
    k = max(3, int(60.0 / (WORLD_SIZE_M / cls.shape[0])) | 1)
    base_t = F.avg_pool2d(base_t, k, 1, k // 2)
    amp_t = F.avg_pool2d(amp_t, k, 1, k // 2)
    # Value-noise fBm relief (deterministic per seed).
    gen = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
    fbm = torch.zeros_like(base_t)
    total, a_ = 0.0, 1.0
    for res in (8, 16, 32, 64, 128, 256, 512):
        n = torch.rand((1, 1, res, res), generator=gen).to(dev)
        n = F.interpolate(n, size=cls.shape, mode="bicubic",
                          align_corners=False)
        fbm += (n - 0.5) * a_
        total += a_
        a_ *= 0.55
    fbm /= total
    h = torch.clamp(base_t + fbm * amp_t * 2.0, 0.0, 1.0)
    if cls.shape[0] != size:
        h = F.interpolate(h, size=(size, size), mode="bicubic",
                          align_corners=False)
    return torch.clamp(h[0, 0], 0.0, 1.0).cpu()


def masks_height_pass(h, cls):
    """Mask-driven replacement for semantic_height_pass + carve_water:
    towns/roads pulled flat, water pinned below its banks — using the
    TRUE class masks instead of colour guesses."""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = h.shape[-1]
    names = [n for n, _ in SEG_CLASSES]
    cls_r = np.asarray(Image.fromarray(cls).resize((size, size),
                                                   Image.NEAREST))
    civ_np = ((cls_r == names.index("town")) |
              (cls_r == names.index("road"))).astype(np.float32)
    wat_np = (cls_r == names.index("water")).astype(np.float32)
    hd = h.to(dev)[None, None]
    civ = torch.from_numpy(civ_np).to(dev)[None, None]
    civ = torch.clamp(F.avg_pool2d(civ, 9, 1, 4) * 2.0, 0.0, 1.0)
    tgt = F.avg_pool2d(hd, 49, 1, 24)
    hd = hd + (tgt - hd) * civ * 0.9
    for _ in range(4):
        nb = F.avg_pool2d(hd, 3, 1, 1)
        hd = hd + (nb - hd) * civ * 0.7
    # Water: below the local bank level, flat.
    w = torch.from_numpy(wat_np).to(dev)[None, None]
    wb = torch.clamp(F.avg_pool2d(w, 9, 1, 4) * 2.0, 0.0, 1.0)
    hb = -F.max_pool2d(-hd, 51, 1, 25)
    hd = torch.where(w > 0.5, torch.minimum(hd, hb - 0.02), hd)
    band = wb - (w > 0.5).float()
    hd = hd * (1.0 - band) + torch.minimum(hd, hb) * band
    return torch.clamp(hd[0, 0], 0.0, 1.0).cpu()


def colorize_from_classes(cls, size: int, seed: int, path: str):
    """Procedural albedo fallback: per-class base colour + value noise.
    Used when the FLUX layout-conditioned colour pass is disabled or
    fails — guarantees the pipeline always emits a usable albedo."""
    base_rgb = {"water": (52, 84, 130), "grass": (116, 142, 86),
                "forest": (58, 92, 55), "town": (128, 120, 116),
                "road": (110, 108, 104), "rock": (138, 128, 118),
                "snow": (235, 238, 242), "sand": (196, 176, 138)}
    names = [n for n, _ in SEG_CLASSES]
    cls_r = np.asarray(Image.fromarray(cls).resize((size, size),
                                                   Image.NEAREST))
    out = np.zeros((size, size, 3), np.float32)
    for i, n in enumerate(names):
        out[cls_r == i] = base_rgb[n]
    rng = np.random.default_rng(seed & 0x7FFFFFFF)
    noise = rng.random((size // 4, size // 4, 1), dtype=np.float32)
    noise = np.asarray(Image.fromarray(
        (noise[..., 0] * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR), np.float32)[..., None] / 255.0
    out *= 0.85 + noise * 0.3
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


# ── Progress reporting for the editor ───────────────────────────────────
# The popup's progress bar polls "<out>.progress": a single line
# "<fraction 0..1> <stage label>".  Overwritten in place at every update.
_PROGRESS_PATH = None


def report(frac: float, label: str):
    print(f"[terrain] {frac * 100.0:5.1f}%  {label}")
    if _PROGRESS_PATH:
        try:
            with open(_PROGRESS_PATH, "w", encoding="utf-8") as pf:
                pf.write(f"{max(0.0, min(1.0, frac)):.4f} {label}")
        except OSError:
            pass


def run_flux(full_prompt: str, out_png: str, size: int, steps: int,
             seed: int, ref_image: str = None, label: str = "text->image",
             prog_lo: float = 0.0, prog_hi: float = 1.0):
    """Runs flux_generate.py, streaming its tqdm output to map diffusion
    steps onto the [prog_lo, prog_hi] slice of the overall progress."""
    import re
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(full_prompt)
        prompt_file = f.name
    try:
        cmd = [sys.executable, FLUX,
               "--prompt-file", prompt_file,
               "--out", out_png,
               "--width", str(size), "--height", str(size),
               "--steps", str(steps), "--seed", str(seed),
               # Quantize the FLUX transformer to fp8 (quanto qfloat8): ~half
               # the VRAM of the bf16 weights with no visible quality loss on
               # this DiT.  Passed explicitly so terrain generation stays fp8
               # regardless of flux_generate.py's (drifting) default.
               "--transformer-quant", "fp8"]
        if ref_image:
            cmd += ["--ref-image", ref_image]
        print(f"[terrain] FLUX {label}:", " ".join(cmd))
        report(prog_lo, f"{label}: loading model")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        step_re = re.compile(rb"(\d+)/(\d+)")
        buf = b""
        while True:
            chunk = proc.stdout.read(64)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            buf += chunk
            # tqdm redraws with \r; split on both terminators.
            while b"\r" in buf or b"\n" in buf:
                cut = min(x for x in (buf.find(b"\r"), buf.find(b"\n"))
                          if x >= 0)
                line, buf = buf[:cut], buf[cut + 1:]
                m = step_re.search(line)
                if m and int(m.group(2)) > 0:
                    frac = int(m.group(1)) / int(m.group(2))
                    report(prog_lo + (prog_hi - prog_lo) * frac,
                           f"{label}: step {m.group(1).decode()}/"
                           f"{m.group(2).decode()}")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"flux_generate exited {proc.returncode}")
        if os.path.exists(out_png + ".err"):
            raise RuntimeError(open(out_png + ".err").read())
        report(prog_hi, f"{label}: done")
    finally:
        os.unlink(prompt_file)


# ── Stage 2: heightmap → terrain conversion ────────────────────────────────
# Torch implementation so it runs on the GPU and can be swapped for a learned
# model behind the same interface.
def refine(h, iters_hydraulic=160, iters_thermal=60, talus=0.008,
           rain=0.01, evap=0.5, capacity=0.08, detail_keep=0.05):
    # detail_keep dropped 0.25 -> 0.05: the "high-frequency relief" it
    # preserved was mostly diffusion speckle (see condition_base); the
    # runtime detail tiles provide the close-range relief instead.
    """h: float32 tensor [H, W] in [0, 1] -> refined tensor [H, W] in [0, 1]."""
    import torch
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h = h.to(dev)
    original = h.clone()

    def shifts(x):
        # 4-neighbourhood via padding (replicate edges).
        xp = F.pad(x[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        return (xp[:-2, 1:-1], xp[2:, 1:-1],   # up, down
                xp[1:-1, :-2], xp[1:-1, 2:])   # left, right

    # ── Hydraulic erosion (simplified pipe model) ──────────────────────
    water    = torch.zeros_like(h)
    sediment = torch.zeros_like(h)
    for _ in range(iters_hydraulic):
        water = water + rain
        total = h + water
        up, dn, lf, rt = shifts(total)
        # Outflow toward the lowest neighbour (bulk approximation).
        lowest = torch.minimum(torch.minimum(up, dn),
                               torch.minimum(lf, rt))
        drop  = torch.clamp(total - lowest, min=0.0)
        flow  = torch.minimum(water, drop * 0.5)
        # Erode proportional to flow, capped by carry capacity.
        erode = torch.clamp(flow * capacity, max=0.002)
        can_carry = torch.clamp(flow * capacity - sediment, min=0.0)
        erode = torch.minimum(erode, can_carry)
        h        = h - erode
        sediment = sediment + erode
        # Deposit where flow stalls, evaporate.
        deposit  = sediment * 0.25 * torch.exp(-flow * 40.0)
        h        = h + deposit
        sediment = sediment - deposit
        water    = (water - flow) * evap

    # ── Thermal erosion (talus relaxation) ─────────────────────────────
    for _ in range(iters_thermal):
        up, dn, lf, rt = shifts(h)
        for nb in (up, dn, lf, rt):
            d = h - nb                      # positive where we tower over nb
            move = torch.clamp((d - talus) * 0.125, min=0.0)
            h = h - move                    # material leaves the peak…
        # (bulk model: matching deposits land via the neighbours' own pass)

    # Detail-preserving blend: keep a fraction of the diffusion image's
    # high-frequency relief so ridges stay characterful.
    blur = F.avg_pool2d(original[None, None], 5, 1, 2)[0, 0]
    high = original - blur
    h = h + high * detail_keep

    h = torch.clamp(h, 0.0, 1.0)
    return h.cpu()


def condition_base(h):
    """Kill diffusion speckle before erosion.  FLUX draws forests and
    photo texture as texel-scale dots; interpreted as elevation each dot
    becomes a 50-150 m pit/spike — rendered as an endless field of sharp
    cones ("egg-carton" terrain).  A 3x3 median removes the dots and a
    small Gaussian removes residual texel noise; real landforms are
    10-100x larger and pass through untouched.  The runtime detail
    worker re-adds controlled 1 m relief, so the base map should carry
    ONLY landforms."""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = h.to(dev)[None, None]
    # Grayscale CLOSING (dilate -> erode, 9x9): fills dark pits smaller
    # than ~9 texels — exactly the forest-dot speckle — while large
    # valleys/rivers pass through.  (Validated on a real generation:
    # mean gradient 9.0 -> 4.0 m/texel, landforms intact.)
    x = F.max_pool2d(x, 9, stride=1, padding=4)
    x = -F.max_pool2d(-x, 9, stride=1, padding=4)
    # Gentle OPENING (erode -> dilate, 5x5): shaves isolated bright
    # spikes the closing can't touch.
    x = -F.max_pool2d(-x, 5, stride=1, padding=2)
    x = F.max_pool2d(x, 5, stride=1, padding=2)
    # 3x3 median for the remaining single-texel outliers.
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    p = xp.unfold(2, 3, 1).unfold(3, 3, 1)
    x = p.reshape(p.shape[0], p.shape[1], p.shape[2], p.shape[3], 9) \
         .median(dim=-1).values
    # Separable Gaussian, sigma ~1.2 texels.
    k = torch.tensor([0.061, 0.242, 0.383, 0.242, 0.061],
                     device=dev, dtype=torch.float32)
    k = (k / k.sum()).view(1, 1, 1, 5)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"), k)
    x = F.conv2d(F.pad(x, (0, 0, 2, 2), mode="replicate"),
                 k.transpose(2, 3))
    # Slope-adaptive flattening: valley floors, fields and roads are
    # naturally SMOOTH — keep relief on steep rock, strip the residual
    # diffusion noise from the flats (this is what makes roads drivable
    # instead of corrugated).
    texel_m = WORLD_SIZE_M / x.shape[-1]
    gs = HEIGHT_AMP_M / (2.0 * texel_m)
    gx = (torch.roll(x, -1, -1) - torch.roll(x, 1, -1)) * gs
    gy = (torch.roll(x, -1, -2) - torch.roll(x, 1, -2)) * gs
    slope = torch.sqrt(gx * gx + gy * gy)          # m per m
    flat_w = torch.clamp(1.0 - slope / 0.12, 0.0, 1.0)
    flat_w = flat_w * flat_w
    wide = F.avg_pool2d(x, 13, 1, 6)
    x = x + (wide - x) * flat_w * 0.85
    return x[0, 0].cpu()


def image_to_heightfield(path: str, size: int):
    import torch
    im = Image.open(path).convert("F")   # float grayscale (luma of RGB ok)
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    # np.array (not asarray): PIL hands out a READ-ONLY buffer view and the
    # in-place ops below need an owning, writable copy.
    a = np.array(im, dtype=np.float32)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    # Gentle histogram shaping: push mid-tones down so valleys dominate
    # (raw diffusion output tends to hover around mid-gray).
    a = a ** 1.4
    return torch.from_numpy(a)


# ── Colour-first heightfield derivation ─────────────────────────────────────
# The two-pass pipeline (FLUX heightmap, then ref-conditioned FLUX colour)
# keeps producing MISMATCHED pairs — reference conditioning is a soft
# prior that many seeds ignore.  Colour-first inverts it: FLUX renders
# only the colour orthophoto (its strong suit), and the heightfield is
# DERIVED from that image — aligned by construction.
COLOR_FIRST_TEMPLATE = (
    # View language FIRST and redundant.  CAUTION: avoid the word "map"
    # — it steers FLUX toward DRAWN cartographic illustrations
    # (isometric mountains, flat colors, panel borders).  Photography
    # vocabulary only.
    "real aerial photograph taken by a survey drone camera pointing "
    "straight down at the ground, nadir orthorectified satellite "
    "photography like Google Earth imagery, seen directly from above "
    "at 90 degrees, photorealistic, covering a 4 kilometer wide area, "
    "{user_prompt}, real photographic texture, natural earth colors, "
    "no illustration, no drawing, no cartoon, no painting, no "
    "isometric view, no perspective, no side view, no horizon, no sky, "
    "no text, no borders"
)


def looks_bad_ortho(png_path: str) -> str:
    """Reject non-orthophoto renders.  Returns '' if OK, else a reason.
    (a) SIDE VIEW: sky band along the top (bright blue rows).
    (b) ILLUSTRATION: drawn/cartoon maps are mostly FLAT color fills —
        real aerial photography has texture everywhere, so a high
        fraction of near-constant patches marks a drawing."""
    a = np.asarray(Image.open(png_path).convert("RGB").resize((256, 256)),
                   dtype=np.float32) / 255.0
    top = a[:32]
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    mx = top.max(-1)
    if float(((mx > 0.62) & (b >= g) & (b > r * 1.02)).mean()) > 0.5:
        return "sky along the top (side view)"
    lum = a.mean(-1)
    # Local flatness: std within 4x4 blocks.
    blocks = lum[:256 - 256 % 4].reshape(64, 4, 64, 4).transpose(
        0, 2, 1, 3).reshape(64, 64, 16)
    flat = float((blocks.std(-1) < 0.008).mean())
    # Calibrated on real outputs: the isometric drawn map measured 39%
    # flat; photorealistic aerials sit well under 20%.
    if flat > 0.30:
        return f"flat-color illustration ({flat * 100.0:.0f}% flat)"
    return ""


def height_from_color(color_path: str, size: int):
    """Derive a plausible heightfield (torch [size,size], 0..1) from the
    colour orthophoto: MiDaS monocular depth when available, else a
    semantic heuristic (water low+flat, vegetation mid, bare rock high,
    snow highest, low-frequency luminance for relief)."""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    im = Image.open(color_path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).to(dev)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]

    h = None
    try:  # MiDaS small: relative inverse-depth; for a nadir ortho higher
        #  ground is closer to the camera => higher inverse depth.
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                               trust_repo=True).to(dev).eval()
        tfm = torch.hub.load("intel-isl/MiDaS", "transforms",
                             trust_repo=True).small_transform
        with torch.no_grad():
            inp = tfm((a.cpu().numpy() * 255).astype(np.uint8)).to(dev)
            pred = midas(inp)
            pred = F.interpolate(pred[None], size=(size, size),
                                 mode="bicubic",
                                 align_corners=False)[0, 0]
        h = pred
        print("[terrain] height_from_color: MiDaS depth")
    except Exception as e:                                # noqa: BLE001
        print(f"[terrain] MiDaS unavailable ({e}) — semantic heuristic")

    if h is None:
        lum = (0.3 * r + 0.5 * g + 0.2 * b)
        mx = a.max(-1).values
        mn = a.min(-1).values
        sat = (mx - mn) / (mx + 1e-6)
        veg = ((g > r) & (g > b)).float()
        snow = ((mx > 0.75) & (sat < 0.15)).float()
        rock = ((sat < 0.35) & (mx > 0.3)).float() * (1.0 - snow)
        base = 0.25 + 0.10 * veg + 0.30 * rock + 0.55 * snow
        blur = F.avg_pool2d(lum[None, None], 65, 1, 32)[0, 0]
        h = base * 0.6 + blur * 0.4

    # Normalize, then force water LOW and FLAT (strict blue-dominant
    # mask, same as carve_water).
    h = h - h.min()
    h = h / (h.max() + 1e-6)
    water = ((b > r * 1.15) & (b >= g * 0.90) & (g > r) &
             (a.max(-1).values > 0.15)).float()
    w = F.avg_pool2d(water[None, None], 9, 1, 4)[0, 0]
    hb = -F.max_pool2d(-h[None, None], 51, 1, 25)[0, 0]
    h = torch.where(w > 0.5, torch.minimum(h, hb - 0.02), h)
    band = torch.clamp(w * 2.0, 0.0, 1.0) - (w > 0.5).float()
    h = h * (1.0 - band) + torch.minimum(h, hb) * band
    return torch.clamp(h, 0.0, 1.0).cpu()


def semantic_height_pass(h, color_path: str):
    """Segmentation-guided height cleanup.  Classes come from the albedo
    (colour heuristics): ROADS/plazas = low-saturation gray/tan;
    BUILDINGS = red/orange roofs or dark slate blocks.  Those areas are
    pulled onto wide, slope-limited ground and relaxed — roads read as
    graded, drivable surfaces instead of noise ribbons.  Water handling
    lives in height_from_color/carve_water; rock keeps its relief."""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = h.shape[-1]
    im = Image.open(color_path).convert("RGB").resize(
        (size, size), Image.BILINEAR)
    a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).to(dev)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1).values
    mn = a.min(-1).values
    sat = (mx - mn) / (mx + 1e-6)
    road = ((sat < 0.18) & (mx > 0.35) & (mx < 0.85)).float()
    roof = (((r > g * 1.25) & (r > b * 1.25) & (r > 0.3)) |
            ((sat < 0.25) & (mx < 0.35) & (mx > 0.15))).float()
    civ = torch.clamp(road + roof, 0.0, 1.0)[None, None]
    civ = torch.clamp(F.avg_pool2d(civ, 9, 1, 4) * 2.0, 0.0, 1.0)
    frac = float(civ.mean())
    print(f"[terrain] semantic pass: {frac * 100.0:.1f}% road/settlement")

    hd = h.to(dev)[None, None]
    # Pull onto wide, gently-graded ground...
    tgt = F.avg_pool2d(hd, 49, 1, 24)
    hd = hd + (tgt - hd) * civ * 0.9
    # ...then relax residual spikes inside the mask.
    for _ in range(4):
        nb = F.avg_pool2d(hd, 3, 1, 1)
        hd = hd + (nb - hd) * civ * 0.7
    return torch.clamp(hd[0, 0], 0.0, 1.0).cpu()


def carve_water(h, color_path: str):
    """Make water LOW.  FLUX heightmaps frequently draw rivers/lakes as
    BRIGHT channels (photo glint / map-style linework), which extrudes
    them into RIDGES in-engine — inverted terrain layout.  The colour
    satellite pass is far more reliable about WHERE water is, so: detect
    the water mask from colour, then carve the heightfield so water sits
    a few meters below its local banks (and is flat-ish)."""
    import torch
    import torch.nn.functional as F
    dev = h.device if h.is_cuda else (
        "cuda" if torch.cuda.is_available() else "cpu")
    a = np.asarray(Image.open(color_path).convert("RGB"),
                   dtype=np.float32) / 255.0
    if a.shape[0] != h.shape[0] or a.shape[1] != h.shape[1]:
        a = np.asarray(Image.fromarray(
            (a * 255).astype(np.uint8)).resize(
            (h.shape[1], h.shape[0]), Image.BILINEAR),
            dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    # Water is BLUE-dominant (teal/steel blue).  The earlier looser test
    # (b > r*1.08 & g > r*1.05) also matched dark FOREST green (slight
    # blue component), carving whole forested mountainsides down to
    # valley level — mesa/table-mountain artifacts.  Requiring b to rival
    # g excludes vegetation: forest g >> b, water b ≈ g or higher.
    water_np = ((b > r * 1.15) & (b >= g * 0.90) & (g > r) &
                (mx > 0.15)).astype(np.float32)
    frac = float(water_np.mean())
    if frac > 0.30:
        print(f"[terrain] carve_water: implausible water fraction "
              f"({frac * 100.0:.0f}%) — skipping carve (bad colour map?)")
        return h
    if frac < 0.001:
        print("[terrain] carve_water: no significant water in colour map")
        return h
    hd = h.to(dev)
    w = torch.from_numpy(water_np).to(dev)
    w = F.avg_pool2d(w[None, None], 9, 1, 4)[0, 0]   # despeckle/soften
    wm = (w > 0.5).float()
    # Bank level: local minimum of the surroundings (51-texel erosion).
    hb = -F.max_pool2d(-hd[None, None], 51, 1, 25)[0, 0]
    depth = 4.0 / HEIGHT_AMP_M                              # ~4 m below banks
    target = hb - depth
    hd = hd * (1.0 - wm) + torch.minimum(hd, target) * wm
    # Feathered shoreline band eases the banks down to water level.
    band = torch.clamp(w * 2.0, 0.0, 1.0) - wm
    hd = hd * (1.0 - band) + torch.minimum(hd, hb) * band
    print(f"[terrain] carve_water: {frac * 100.0:.1f}% of the map carved "
          "to water level")
    return torch.clamp(hd, 0.0, 1.0).cpu()


def save_u16(h, path: str):
    a = (h.numpy() * 65535.0 + 0.5).astype(np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(a, mode="I;16").save(path)
    print(f"[terrain] wrote {path}  ({a.shape[1]}x{a.shape[0]}, 16-bit)")


# ── Terrain mesh export ─────────────────────────────────────────────────────
# Bakes the (already height-scaled) heightfield into a textured grid mesh
# saved NEXT TO the textures with the SAME base name: <out>.glb.  GLB so the
# editor's Content Browser can thumbnail it and the scene importer can place
# it directly.  Self-contained: the albedo rides inside the GLB buffer, and
# the writer is dependency-free (json + struct + numpy).
#
# World mapping matches the engine (tile_creator.comp): the map spans
# kWorldMapSize = 16384 m centered on the origin, white = 2000 m.
def save_mesh_glb(h, out_png: str, color_png: str = None, mesh_res: int = 256):
    import json
    import struct

    WORLD_SIZE = WORLD_SIZE_M   # kWorldMapSize (global_definition.glsl.h)
    HEIGHT_AMP = HEIGHT_AMP_M   # tile_creator.comp heightmap multiplier

    a = h.numpy().astype(np.float32)
    src_h, src_w = a.shape
    res = max(2, int(mesh_res))

    # Decimated grid sample of the heightfield.
    ys = np.round(np.linspace(0, src_h - 1, res)).astype(np.int64)
    xs = np.round(np.linspace(0, src_w - 1, res)).astype(np.int64)
    grid = a[np.ix_(ys, xs)] * HEIGHT_AMP                     # [res, res] m

    half = WORLD_SIZE * 0.5
    lin = np.linspace(-half, half, res, dtype=np.float32)
    xw = np.broadcast_to(lin[None, :], (res, res))            # +u -> +x
    zw = np.broadcast_to(lin[:, None], (res, res))            # +v -> +z
    pos = np.stack([xw, grid, zw], axis=-1).reshape(-1, 3).astype(np.float32)

    # Normals via central differences on the sampled grid.
    step = WORLD_SIZE / (res - 1)
    dx = np.gradient(grid, step, axis=1)
    dz = np.gradient(grid, step, axis=0)
    n = np.stack([-dx, np.ones_like(grid), -dz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = n.reshape(-1, 3).astype(np.float32)

    # UVs: glTF texcoord origin is the image top-left, which is exactly
    # how the engine maps the heightmap (row 0 = world_min.z edge).
    uu = np.broadcast_to(np.linspace(0.0, 1.0, res, dtype=np.float32)[None, :],
                         (res, res))
    vv = np.broadcast_to(np.linspace(0.0, 1.0, res, dtype=np.float32)[:, None],
                         (res, res))
    uv = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    # Two CCW (viewed from +Y) triangles per quad.
    i0 = (np.arange(res - 1)[:, None] * res + np.arange(res - 1)[None, :])
    i0 = i0.reshape(-1)
    quad = np.stack([i0, i0 + res, i0 + 1,
                     i0 + 1, i0 + res, i0 + res + 1], axis=-1)
    idx = quad.reshape(-1).astype(np.uint32)

    def pad4(b, fill=b"\x00"):
        return b + fill * ((4 - len(b) % 4) % 4)

    pos_b, nrm_b, uv_b, idx_b = (pad4(x.tobytes())
                                 for x in (pos, nrm, uv, idx))
    img_b = b""
    if color_png and os.path.exists(color_png):
        # Embed the albedo at <=2048 px — the full 8k PNG would bloat the
        # GLB by ~60 MB for a preview/placeable asset.
        import io
        cim = Image.open(color_png).convert("RGB")
        if max(cim.size) > 2048:
            cim = cim.resize((2048, 2048), Image.LANCZOS)
        cbuf = io.BytesIO()
        cim.save(cbuf, format="PNG")
        img_b = pad4(cbuf.getvalue())

    views, accessors, offset = [], [], 0

    def add_view(blob, target=None):
        nonlocal offset
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
        if target:
            v["target"] = target
        views.append(v)
        offset += len(blob)
        return len(views) - 1

    vcount = res * res
    add_view(pos_b, 34962)
    accessors.append({"bufferView": 0, "componentType": 5126,
                      "count": vcount, "type": "VEC3",
                      "min": pos.min(axis=0).tolist(),
                      "max": pos.max(axis=0).tolist()})
    add_view(nrm_b, 34962)
    accessors.append({"bufferView": 1, "componentType": 5126,
                      "count": vcount, "type": "VEC3"})
    add_view(uv_b, 34962)
    accessors.append({"bufferView": 2, "componentType": 5126,
                      "count": vcount, "type": "VEC2"})
    add_view(idx_b, 34963)
    accessors.append({"bufferView": 3, "componentType": 5125,
                      "count": int(idx.size), "type": "SCALAR"})

    material = {"name": "terrain",
                "pbrMetallicRoughness": {"metallicFactor": 0.0,
                                         "roughnessFactor": 1.0},
                "doubleSided": False}
    gltf = {
        "asset": {"version": "2.0", "generator": "terrain_from_text.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "terrain"}],
        "meshes": [{"name": "terrain", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0, "mode": 4}]}],
        "materials": [material],
        "bufferViews": views,
        "accessors": accessors,
    }
    if img_b:
        img_view = add_view(img_b)
        gltf["images"] = [{"bufferView": img_view, "mimeType": "image/png"}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987,
                             "wrapS": 33071, "wrapT": 33071}]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    else:
        material["pbrMetallicRoughness"]["baseColorFactor"] = \
            [0.45, 0.5, 0.4, 1.0]

    bin_chunk = pos_b + nrm_b + uv_b + idx_b + img_b
    gltf["buffers"] = [{"byteLength": len(bin_chunk)}]
    json_chunk = pad4(json.dumps(gltf, separators=(",", ":"))
                      .encode("utf-8"), b" ")

    glb_path = os.path.splitext(out_png)[0] + ".glb"
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(glb_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))          # glTF v2
        f.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))    # JSON
        f.write(json_chunk)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))     # BIN
        f.write(bin_chunk)
    print(f"[terrain] wrote {glb_path}  ({res}x{res} grid, "
          f"{idx.size // 3} tris"
          f"{', albedo embedded' if img_b else ''})")
    return glb_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="terrain description (stage 1 input)")
    ap.add_argument("--prompt-file",
                    help="read the terrain description from a file "
                         "(editor integration — avoids shell quoting)")
    ap.add_argument("--from-image",
                    help="skip FLUX; use this image as the raw heightmap")
    ap.add_argument("--out", default="content/terrain/generated_map.png")
    ap.add_argument("--size", type=int, default=8192,
                    help="final BASE map resolution (8192 over the 32 km "
                         "world = 4 m/texel; 1 m detail comes from the "
                         "runtime detail tiles, see "
                         "terrain_detail_worker.py)")
    ap.add_argument("--gen-size", type=int, default=2048,
                    help="FLUX RENDER resolution.  The heightfield is upsampled "
                         "from this to --size, so lowering it cuts FLUX VRAM + "
                         "time ~quadratically with almost no loss on a heightmap.")
    ap.add_argument("--refine-size", type=int, default=2048,
                    help="resolution the erosion/refinement model runs at; "
                         "the result is bicubic-upsampled to --size (erosion "
                         "at 8192^2 is ~16x the cost for little visible gain "
                         "under the runtime 1 m detail pass)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the stage-2 terrain conversion model")
    ap.add_argument("--naturalize", dest="naturalize", action="store_true",
                    default=True,
                    help="match real-DEM statistics: despike/de-terrace, "
                         "biome hypsometry, slope budget, stream-power "
                         "erosion with a natural slope-area law (default on)")
    ap.add_argument("--no-naturalize", dest="naturalize",
                    action="store_false")
    ap.add_argument("--nat-biome", default="auto",
                    choices=["auto", "hills", "mountains", "plains"],
                    help="hypsometry/slope target biome (auto = from prompt)")
    ap.add_argument("--nat-strength", type=float, default=1.0,
                    help="0..1 blend of the naturalize correction")
    ap.add_argument("--mesh-res", type=int, default=512,
                    help="terrain mesh grid resolution (verts per side) for "
                         "the .glb export (512 over 32 km = 64 m quads); "
                         "0 disables the mesh")
    ap.add_argument("--height-scale", type=float, default=0.25,
                    help="peak height as a fraction of the engine's full "
                         "range (tile_creator.comp maps white to 2000 m, "
                         "far too dramatic for a full-range normalized "
                         "map).  0.25 => ~500 m peaks.  Applied when "
                         "writing the 16-bit PNG so engine constants and "
                         "CPU/GPU sync stay untouched.")
    ap.add_argument("--color", action="store_true",
                    help="also generate a matching COLOR satellite map "
                         "(<out>_color.png) via a second FLUX pass "
                         "conditioned on the heightfield")
    ap.add_argument("--pcg", action="store_true", default=True,
                    help="segmentation-driven PCG proxies (buildings + "
                         "trees as white meshes) -> <out>_pcg.glb")
    ap.add_argument("--no-pcg", dest="pcg", action="store_false")
    ap.add_argument("--height-first", action="store_true",
                    help="legacy two-pass order (FLUX heightmap, then "
                         "ref-conditioned colour).  Default is COLOUR-"
                         "FIRST: FLUX renders only the orthophoto and "
                         "the heightfield derives from it — heightmap "
                         "and albedo match by construction.")
    ap.add_argument("--layout", action="store_true",
                    help="LAYOUT-FIRST: FLUX renders a flat-color land-"
                         "cover map (fixed palette), quantized into "
                         "exact class masks (<out>_seg.png).  Heightmap "
                         "AND albedo derive from the masks, and PCG "
                         "places buildings/trees from ground-truth "
                         "classes instead of color heuristics.")
    ap.add_argument("--carve-water", action="store_true",
                    help="reshape the heightfield so the colour map's "
                         "water sits below its banks (OFF by default: "
                         "the original ML heightmap is used as-is)")
    ap.add_argument("--install", action="store_true",
                    help="ALSO replace assets/map.png (backs up the original "
                         "to assets/map.png.bak once)")
    args = ap.parse_args()

    if args.prompt_file and not args.prompt:
        with open(args.prompt_file, encoding="utf-8") as pf:
            args.prompt = pf.read().strip()
    if not args.prompt and not args.from_image:
        ap.error("need --prompt / --prompt-file (FLUX) or --from-image")

    global _PROGRESS_PATH
    _PROGRESS_PATH = (args.out if os.path.isabs(args.out)
                      else os.path.join(REPO_ROOT, args.out)) + ".progress"

    # Overall progress budget: heightmap FLUX 2%..48%, erosion ..55%,
    # color FLUX 58%..95%, final write/install ..100%.  Without --color
    # the heightmap pass stretches to 90%.
    hm_hi = 0.48 if args.color else 0.90

    refine_res = min(args.refine_size, args.size)
    layout_mode = args.layout and not args.from_image
    color_first = (args.color and not args.height_first
                   and not args.from_image and not layout_mode)
    color_raw = None
    cls_map = None
    seg_raw = None
    if layout_mode:
        # ── LAYOUT-FIRST: FLUX renders the flat-color land-cover map;
        # classes, heightfield and albedo all derive from it.
        seg_raw = os.path.join(tempfile.gettempdir(),
                               "terrain_seg_raw.png")

        def layout_side_view(png_path: str) -> str:
            """'' if the layout looks top-down, else the reason.
            Side-profile map-icon mountains have two signatures a real
            classification raster never has: (a) black ridgeline
            HATCHING, (b) snow at the TOP EDGE of brown regions (peaks
            drawn upward) instead of centred inside them."""
            a = np.asarray(Image.open(png_path).convert("RGB").resize(
                (256, 256)), dtype=np.float32) / 255.0
            mx = a.max(-1)
            ink = float((mx < 0.18).mean())
            if ink > 0.02:
                return f"ridgeline hatching ({ink * 100.0:.1f}% ink)"
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            snow = (mx > 0.85) & ((a.max(-1) - a.min(-1)) < 0.12)
            brown = (r > g) & (g > b) & (r > 0.35) & (r < 0.8)
            if snow.mean() > 0.01 and brown.mean() > 0.02:
                ys, _ = np.nonzero(snow)
                yb, _ = np.nonzero(brown)
                # snow centroid clearly ABOVE brown centroid => profile
                if float(yb.mean() - ys.mean()) > 20.0:
                    return "snow above rock (side-profile mountains)"
            return ""

        import random as _rndseg
        seg_seed = args.seed if args.seed >= 0 \
            else _rndseg.randint(0, 1 << 30)
        for attempt in range(3):
            run_flux(SEG_PROMPT_TEMPLATE.format(user_prompt=args.prompt),
                     seg_raw, args.gen_size, args.steps,
                     seg_seed + attempt * 7919,
                     label="land-cover layout", prog_lo=0.02, prog_hi=0.42)
            reason = layout_side_view(seg_raw)
            if not reason:
                break
            print(f"[terrain] layout attempt {attempt + 1}: rejected — "
                  f"{reason}; retrying with a new seed")
        report(0.44, "quantizing layout into class masks")
        cls_map = quantize_to_classes(seg_raw, args.gen_size)
        names = [n for n, _ in SEG_CLASSES]
        stats = {n: float((cls_map == i).mean() * 100.0)
                 for i, n in enumerate(names)}
        print("[terrain] layout classes: " +
              "  ".join(f"{n} {v:.1f}%" for n, v in stats.items()))
        report(0.46, "heightfield from class layout")
        h = height_from_classes(cls_map, refine_res,
                                args.seed if args.seed >= 0 else 12345)
    elif color_first:
        # ── COLOUR-FIRST: one FLUX pass renders the orthophoto; the
        # heightfield DERIVES from it (height_from_color) — heightmap
        # and albedo cannot disagree.
        color_raw = os.path.join(tempfile.gettempdir(),
                                 "terrain_color_raw.png")
        import random as _rnd0
        cf_seed = args.seed if args.seed >= 0 else _rnd0.randint(0, 1 << 30)
        for attempt in range(3):
            run_flux(COLOR_FIRST_TEMPLATE.format(user_prompt=args.prompt),
                     color_raw, args.gen_size, args.steps,
                     cf_seed + attempt * 7919,
                     label="color orthophoto", prog_lo=0.02, prog_hi=0.70)
            reason = looks_bad_ortho(color_raw)
            if not reason:
                break
            print(f"[terrain] attempt {attempt + 1}: rejected — {reason}; "
                  "retrying with a new seed")
        report(0.72, "deriving heightfield from color")
        h = height_from_color(color_raw, refine_res)
    else:
        raw_png = args.from_image
        if not raw_png:
            raw_png = os.path.join(tempfile.gettempdir(),
                                   "terrain_raw.png")
            run_flux(PROMPT_TEMPLATE.format(user_prompt=args.prompt),
                     raw_png, args.gen_size, args.steps, args.seed,
                     label="heightmap", prog_lo=0.02, prog_hi=hm_hi)
        h = image_to_heightfield(raw_png, refine_res)

    print("[terrain] stage 2: heightmap -> terrain conversion")
    report(hm_hi + 0.02, "terrain conversion (erosion)")
    # Refine (erosion) at --refine-size, then bicubic-upsample the result
    # to the final --size base map (2048 -> 8192 by default).  The 1 m
    # detail comes from the runtime detail-tile worker, so refining at
    # full 8k is wasted work.
    h = condition_base(h)          # despeckle BEFORE erosion
    if not args.no_refine:
        h = refine(h)

    # ── Natural-distribution correction (terrain_naturalize) ───────────────
    # Measures the heightfield against real-DEM statistics (spectral slope,
    # slope distribution, hypsometry, clamp/terrace mass, drainage concavity)
    # and corrects deviations: despike, de-terrace, de-clamp, biome-target
    # hypsometric remap, and stream-power erosion that carves dendritic
    # drainage obeying the S~A^-theta slope-area law.  Runs BEFORE the
    # masks/semantic pass so town/water flattening stays authoritative.
    nat_report = None
    if args.naturalize:
        import terrain_naturalize as tn
        import torch as _torch
        nat_biome = (tn.biome_from_prompt(args.prompt or "")
                     if args.nat_biome == "auto" else args.nat_biome)
        wm = None
        if layout_mode:
            _names = [n for n, _ in SEG_CLASSES]
            _cls_r = np.asarray(Image.fromarray(cls_map).resize(
                (h.shape[-1], h.shape[-1]), Image.NEAREST))
            wm = _cls_r == _names.index("water")
        report(hm_hi + 0.05, "naturalize (distribution match)")
        _h_np, nat_report = tn.naturalize(
            h.numpy().astype(np.float64),
            biome=nat_biome,
            strength=args.nat_strength,
            water_mask=wm,
            world_m=WORLD_SIZE_M,
            amp_m=HEIGHT_AMP_M * args.height_scale,  # EFFECTIVE world amp
            verbose=True)
        h = _torch.from_numpy(_h_np.astype(np.float32))

    if layout_mode:
        # Mask-driven: towns/roads flat, water below its banks —
        # enforced AFTER erosion so the hydraulic pass can't re-roughen
        # them.  Uses the TRUE class masks.
        h = masks_height_pass(h, cls_map)
    elif color_first:
        # Roads flat, settlements level — enforced AFTER erosion so the
        # hydraulic pass can't re-roughen them.
        h = semantic_height_pass(h, color_raw)
    if args.size != refine_res:
        import torch
        import torch.nn.functional as F
        h = F.interpolate(h[None, None], size=(args.size, args.size),
                          mode="bicubic", align_corners=False)[0, 0]
        h = torch.clamp(h, 0.0, 1.0)
        print(f"[terrain] base map upscaled {refine_res} -> {args.size}")
    report(hm_hi + 0.07, "terrain conversion done")

    out = args.out if os.path.isabs(args.out) \
        else os.path.join(REPO_ROOT, args.out)

    # ── Stage 3 (optional): colour satellite map, conditioned on the
    # refined heightfield so its layout matches the terrain.  Runs BEFORE
    # the final heightmap write: the editor treats the heightmap PNG's
    # existence as "job complete", so it must be the LAST artifact.
    if layout_mode:
        stem, _ = os.path.splitext(out)
        # 1. Segmentation map at final resolution (exact palette PNG).
        seg_out = stem + "_seg.png"
        save_seg_png(np.asarray(Image.fromarray(cls_map).resize(
            (args.size, args.size), Image.NEAREST)), seg_out)
        print(f"[terrain] wrote {seg_out}  (land-cover class masks)")
        # Keep the raw FLUX layout render beside the map: quantization
        # problems are impossible to debug from the quantized PNG alone.
        try:
            shutil.copy2(seg_raw, stem + "_layout.png")
        except OSError:
            pass
        # 2. Albedo: FLUX conditioned on the layout render when --color,
        #    procedural class colorization otherwise / on failure.
        color_out = stem + "_color.png"
        wrote_color = False
        if args.color:
            try:
                # Ref conditioning is soft: score each attempt's CLASS
                # AGREEMENT against the seg (PCG places from the seg, so
                # colour drift = buildings visibly off their painted
                # bases) and retry with new seeds, keeping the best.
                import random as _rndl
                seed0 = args.seed if args.seed >= 0 \
                    else _rndl.randint(0, 1 << 30)
                seg_names = [n for n, _ in SEG_CLASSES]
                best_s, best_bytes = -1.0, None
                for attempt in range(3):
                    run_flux(COLOR_FROM_LAYOUT_TEMPLATE.format(
                                 user_prompt=args.prompt),
                             color_out, args.gen_size, args.steps,
                             seed0 + attempt * 7919,
                             ref_image=seg_raw, label="albedo from layout",
                             prog_lo=0.60, prog_hi=0.93)
                    s = seg_color_agreement(cls_map, color_out, seg_names)
                    print(f"[terrain] albedo attempt {attempt + 1}: "
                          f"seg agreement {s:.2f}")
                    if s > best_s:
                        with open(color_out, "rb") as bf:
                            best_s, best_bytes = s, bf.read()
                    if s >= 0.55:
                        break
                if best_bytes is not None:
                    with open(color_out, "wb") as bf:
                        bf.write(best_bytes)
                print(f"[terrain] albedo seg agreement: {best_s:.2f} "
                      f"{'(aligned)' if best_s >= 0.55 else '(WEAK)'}")
                Image.open(color_out).convert("RGB").resize(
                    (args.size, args.size), Image.LANCZOS).save(color_out)
                wrote_color = True
            except Exception as e:                        # noqa: BLE001
                print(f"[terrain] layout albedo FLUX pass failed ({e}) "
                      "— procedural colorization")
        if not wrote_color:
            colorize_from_classes(cls_map, args.size,
                                  args.seed if args.seed >= 0 else 12345,
                                  color_out)
        print(f"[terrain] wrote {color_out}  (albedo, layout-aligned)")
    elif args.color and color_first:
        # COLOUR-FIRST: the orthophoto ALREADY exists (the heightfield
        # was derived from it) — just upscale/install it.  No second
        # FLUX pass, no retry loop, no possible mismatch.
        stem, _ = os.path.splitext(out)
        color_out = stem + "_color.png"
        Image.open(color_raw).convert("RGB").resize(
            (args.size, args.size), Image.LANCZOS).save(color_out)
        print(f"[terrain] wrote {color_out}  (color orthophoto, "
              "height-aligned by construction)")
    elif args.color:
        stem, _ = os.path.splitext(out)
        color_out = stem + "_color.png"
        ref8 = os.path.join(tempfile.gettempdir(), "terrain_ref8.png")
        Image.fromarray(
            (h.numpy() * 255.0 + 0.5).astype(np.uint8), mode="L"
        ).convert("RGB").save(ref8)
        # FLUX's reference conditioning is soft — some seeds ignore the
        # heightmap and paint an unrelated scene (water on ridges etc.).
        # Score each attempt by edge-structure correlation against the
        # heightfield and retry with a new seed; keep the best.
        import random as _rnd
        base_seed = args.seed if args.seed >= 0 else _rnd.randint(0, 1 << 30)
        h_small = np.array(Image.fromarray(
            h.numpy()).resize((256, 256), Image.BILINEAR), dtype=np.float32)

        def _edges(x):
            gy, gx = np.gradient(x)
            e = np.hypot(gx, gy)
            return (e - e.mean()) / (e.std() + 1e-6)

        best_score, best_bytes = -1e9, None
        for attempt in range(3):
            run_flux(COLOR_PROMPT_TEMPLATE.format(
                         user_prompt=args.prompt or "natural terrain"),
                     color_out, args.gen_size, args.steps,
                     base_seed + attempt * 7919,
                     ref_image=ref8, label="color satellite",
                     prog_lo=0.58, prog_hi=0.95)
            ci = np.asarray(Image.open(color_out).convert("L").resize(
                (256, 256), Image.BILINEAR), dtype=np.float32)
            score = float((_edges(h_small) * _edges(ci)).mean())
            print(f"[terrain] color attempt {attempt + 1}: "
                  f"layout match {score:+.3f}")
            if score > best_score:
                with open(color_out, "rb") as bf:
                    best_score, best_bytes = score, bf.read()
            if score >= 0.06:
                break
        if best_bytes is not None:
            with open(color_out, "wb") as bf:
                bf.write(best_bytes)
        print(f"[terrain] color layout match: {best_score:+.3f} "
              f"{'(followed heightmap)' if best_score >= 0.06 else '(WEAK — consider regenerating)'}")
        # FLUX renders the albedo/colour map at --gen-size; upsample it to
        # --size so it matches the heightmap output resolution (the heightfield
        # is already upsampled in image_to_heightfield).  LANCZOS upscale of a
        # 1024 render — no new detail, but the map_mask matches the terrain.
        if args.gen_size != args.size:
            _ci = Image.open(color_out).convert("RGB")
            if _ci.size != (args.size, args.size):
                _ci.resize((args.size, args.size),
                           Image.LANCZOS).save(color_out)
                print(f"[terrain] upscaled colour map "
                      f"{args.gen_size} -> {args.size}")
        print(f"[terrain] wrote {color_out}  (color satellite map)")
        # OPT-IN water carve (--carve-water): reshapes the heightfield
        # along the colour map's water.  Off by default — the original
        # ML heightmap is used untouched.
        if args.carve_water:
            report(0.955, "carving water into heightfield")
            h = carve_water(h, color_out)
        # Roads/settlements flat regardless of pipeline order.
        h = semantic_height_pass(h, color_out)

    report(0.97, "writing terrain map")
    # World-height scaling happens HERE (not in the engine): the refined
    # heightfield spans the full [0,1] range, which the tile creator maps
    # to 0..2000 m.  NOTE: applied after the colour pass, whose FLUX
    # conditioning wants the full-contrast reference.
    scale = max(0.0, min(1.0, args.height_scale))
    if scale < 1.0:
        h = h * scale
        print(f"[terrain] height scale {scale:g} "
              f"(peaks ~{HEIGHT_AMP_M * scale:.0f} m)")
    save_u16(h, out)

    # Natural-distribution report sidecar: <out>.stats.json with the
    # before/after metrics from the naturalize pass — the verify loop and
    # the editor can gate on `after.natural` without recomputing.
    if nat_report is not None:
        import json as _json
        try:
            with open(out + ".stats.json", "w", encoding="utf-8") as _f:
                _json.dump(nat_report, _f, indent=2)
            print(f"[terrain] wrote {out}.stats.json "
                  f"(natural={nat_report['after']['natural']})")
        except OSError as _e:
            print(f"[terrain] stats sidecar failed (non-fatal): {_e}")

    # Terrain mesh: same folder, same base name as the textures
    # (<out>.glb), baked from the scaled heightfield with the albedo
    # embedded.  Before the .done sentinel, so the editor's auto-apply
    # only fires once every artifact is complete.
    if args.mesh_res > 0:
        report(0.98, "baking terrain mesh")
        color_out = None
        if args.color:
            color_out = os.path.splitext(out)[0] + "_color.png"
        save_mesh_glb(h, out, color_out, args.mesh_res)

    # PCG proxies: segmentation-driven white placeholder meshes
    # (buildings + trees) placed on the terrain — <out>_pcg.glb.
    if args.pcg and (args.color or layout_mode):
        report(0.985, "PCG proxies from segmentation")
        pcg_out = os.path.splitext(out)[0] + "_pcg.glb"
        try:
            import terrain_pcg
            seg_png = (os.path.splitext(out)[0] + "_seg.png"
                       if layout_mode else None)
            place_towns, place_trees = _biome_pcg_flags(args.prompt or "")
            print(f"[pcg] stage START -> {pcg_out}"
                  + (f" (masks: {seg_png})" if seg_png else
                     " (color heuristics)")
                  + f" [towns={place_towns} trees={place_trees}]", flush=True)
            terrain_pcg.build_pcg_glb(
                os.path.splitext(out)[0] + "_color.png", out, pcg_out,
                seg_path=seg_png,
                place_towns=place_towns, place_trees=place_trees)
            print(f"[pcg] stage DONE -> {pcg_out} "
                  f"({os.path.getsize(pcg_out)} B)", flush=True)
        except Exception as e:                            # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[terrain] PCG stage FAILED (non-fatal): {e}",
                  flush=True)
    elif args.pcg:
        print("[pcg] stage SKIPPED: no --color albedo "
              "(segmentation needs the colour map)", flush=True)

    if args.install:
        if os.path.exists(MAP_PNG) and not os.path.exists(MAP_PNG + ".bak"):
            shutil.copy2(MAP_PNG, MAP_PNG + ".bak")
            print(f"[terrain] backed up original -> {MAP_PNG}.bak")
        shutil.copy2(out, MAP_PNG)
        print(f"[terrain] installed -> {MAP_PNG}")
        # The tile fragment shader's terrain albedo IS assets/map_mask.png
        # sampled at full-world UV — installing the colour satellite map
        # there textures the terrain directly, no engine changes needed.
        if args.color:
            stem, _ = os.path.splitext(out)
            color_out = stem + "_color.png"
            mask_png = os.path.join(REPO_ROOT, "assets", "map_mask.png")
            if os.path.exists(color_out):
                if os.path.exists(mask_png) and \
                   not os.path.exists(mask_png + ".bak"):
                    shutil.copy2(mask_png, mask_png + ".bak")
                    print(f"[terrain] backed up original -> {mask_png}.bak")
                shutil.copy2(color_out, mask_png)
                print(f"[terrain] installed color -> {mask_png} "
                      "(terrain surface texture)")
        print("[terrain] NOTE: engine restart required; visuals use the new "
              "map, but CPU physics (terrainMap) still runs the analytic "
              "FBM until the engine-side sync is implemented.")

    # Progress file served its purpose.
    try:
        if _PROGRESS_PATH and os.path.exists(_PROGRESS_PATH):
            os.remove(_PROGRESS_PATH)
    except OSError:
        pass

    # Completion sentinel — written LAST, after every artifact (heightmap,
    # colour map, --install copies) is fully on disk.  The editor polls
    # THIS file, not the output PNG: polling the PNG races the auto-apply
    # against PIL still flushing it (partial read -> stbi returns null ->
    # engine runtime_error).
    with open(out + ".done", "w", encoding="utf-8") as df:
        df.write("ok")


if __name__ == "__main__":
    # Error sidecar for the editor: on ANY failure write "<out>.err" with the
    # traceback so the C++ polling loop reports it (same contract as
    # flux_generate.py) instead of waiting forever for the output PNG.
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        err = traceback.format_exc()
        sys.stderr.write(err)
        try:
            for i, a in enumerate(sys.argv):
                if a == "--out" and i + 1 < len(sys.argv):
                    with open(sys.argv[i + 1] + ".err", "w",
                              encoding="utf-8") as ef:
                        ef.write(err)
                    break
        except OSError:
            pass
        sys.exit(1)