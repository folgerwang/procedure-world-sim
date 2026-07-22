#!/usr/bin/env python3
"""
terrain_detail_worker.py — on-demand 1 m terrain detail tile generator.

The engine renders the terrain from an 8192^2 BASE heightmap covering the
32768 m world (4 m/texel).  Near the camera it wants 1 m resolution: the
world is split into 16x16 DETAIL TILES of 2048 m, each 2048^2 texels.
This worker generates those tiles on request, at runtime, in the
background.

Protocol (dead simple, file-based, race-free):
    tiles_dir/req_XX_YY        engine touches this to request tile (XX,YY)
    tiles_dir/tile_XX_YY.png   worker's answer: 16-bit, 2048^2, SAME 0..1
                               height scale as the base map (engine
                               multiplies by 2000 m).  Written atomically
                               (tmp + os.replace) so the engine can load
                               it the moment it exists.
    tiles_dir/worker.alive     heartbeat (mtime refreshed every loop) so
                               the engine can detect a dead worker.
The request file is deleted after the tile is written.  Tile indices:
XX, YY in 0..15, tile (0,0) at world (-16384, -16384).

Detail synthesis ("procedural torch" stage):
    1. BASE: Catmull-Rom (bicubic) upsample of the tile's window of the
       8k base map — C1-smooth, and because it samples the GLOBAL map it
       is IDENTICAL along shared tile borders: seamless by construction.
    2. DETAIL: multi-octave value noise evaluated on lattices anchored at
       WORLD-integer coordinates (not tile-local), hashed from lattice
       position only — deterministic and continuous across tile borders,
       so neighbouring tiles agree exactly without any overlap blending.
       Amplitude is slope-modulated (rocky relief on slopes, calm
       valleys) and fades to ZERO at the transition band so a detail
       tile meets the raw base map smoothly where no neighbour is
       loaded.
    3. ml_detail() hook: drop-in point for a learned SR model — return a
       tensor to REPLACE the procedural detail (same seamlessness rules:
       it must be a function of world position; use >=64 px context
       margin).

Usage (spawned by the engine on terrain apply):
    python terrain_detail_worker.py --map content/terrain/xxx.png
        --tiles-dir content/terrain/xxx_tiles [--color content/.._color.png]
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # 8k PNGs

# kTerrainMapMeters: the map's world footprint (decoupled from the 32 km
# world so photo-scale FLUX content renders at natural size).
WORLD_SIZE_M    = 4096.0
HEIGHT_AMP_M    = 250.0  # matches kTerrainHeightAmpMeters
DETAIL_TILE_M   = WORLD_SIZE_M / 16.0    # 256 m
# 2049, not 2048: texel CENTERS sit on integer world meters, so the last
# texel row/column of tile (i) and the first of tile (i+1) are the SAME
# world positions — and since generation is world-deterministic they hold
# bit-identical values.  Border vertices then sample identical heights
# from either tile: exact seamlessness, no gutters or overlap blending.
DETAIL_TILE_RES = 2049
TILES_PER_SIDE  = int(WORLD_SIZE_M / DETAIL_TILE_M)   # 16
MARGIN_TEXELS   = 64    # context margin so filtering/relax can't seam


# ── ML upscaler (Real-ESRGAN x4) ────────────────────────────────────────────
# The global maps are 4 m/texel; a 2 km tile is a 512^2 crop, so one x4
# SR pass lands exactly on the 2048^2 @ 1 m tile the engine streams.
# Soft dependency: pip install realesrgan basicsr — if missing, the
# worker logs once and falls back to bicubic (albedo) + procedural
# noise (height).
class MlUpscaler:
    def __init__(self):
        self.ok = False
        self.err = None
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                   "download/v0.1.0/RealESRGAN_x4plus.pth")
            self.engine = RealESRGANer(
                scale=4, model_path=url, model=model,
                tile=512, tile_pad=16,
                half=torch.cuda.is_available())
            self.ok = True
            print("[detail-worker] Real-ESRGAN x4 ready")
        except Exception as e:                        # noqa: BLE001
            self.err = str(e)
            print(f"[detail-worker] Real-ESRGAN unavailable ({e}) — "
                  "bicubic/procedural fallback "
                  "(pip install realesrgan basicsr)")

    def up4_rgb(self, rgb_u8):
        """[H,W,3] uint8 -> [4H,4W,3] uint8."""
        out, _ = self.engine.enhance(rgb_u8[:, :, ::-1], outscale=4)  # BGR
        return out[:, :, ::-1]


# Deterministic edge feather: 1 in the tile interior, 0 at the tile
# border, over `width` texels.  ML SR is not bit-exact across
# independently processed tiles, so its contribution must vanish at
# shared borders — the world-anchored procedural detail (bit-exact)
# carries the border band instead.
def _edge_feather(full, margin, width, dev):
    import torch
    idx = torch.arange(full, device=dev, dtype=torch.float32)
    lo = (idx - margin) / width
    hi = (full - 1 - margin - idx) / width
    f1 = torch.clamp(torch.minimum(lo, hi), 0.0, 1.0)
    f1 = f1 * f1 * (3.0 - 2.0 * f1)
    return f1[None, :] * f1[:, None]


# ── Procedural detail ───────────────────────────────────────────────────────
def _hash01(ix, iy, seed):
    """Deterministic [0,1) hash of integer lattice coords (torch tensors)."""
    import torch
    # Pre-wrap the seed constant into signed int64 range — the raw Python
    # product (seed * 1.44e18) exceeds int64 and raises OverflowError when
    # mixed with int64 tensors.
    sk = (int(seed) * 1442695040888963407) % (1 << 64)
    if sk >= (1 << 63):
        sk -= (1 << 64)
    h = ix.to(torch.int64) * 374761393 + iy.to(torch.int64) * 668265263 + sk
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).to(torch.float32) / float(0x1000000)


def _value_noise(wx, wy, cell, seed):
    """Value noise at world positions wx, wy (meters, [H,W] tensors) with
    lattice spacing `cell` meters, anchored at world origin => continuous
    across tile borders by construction."""
    import torch
    gx = wx / cell
    gy = wy / cell
    ix0 = torch.floor(gx)
    iy0 = torch.floor(gy)
    fx = gx - ix0
    fy = gy - iy0
    # smoothstep weights (C1)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    v00 = _hash01(ix0,     iy0,     seed)
    v10 = _hash01(ix0 + 1, iy0,     seed)
    v01 = _hash01(ix0,     iy0 + 1, seed)
    v11 = _hash01(ix0 + 1, iy0 + 1, seed)
    return ((v00 * (1 - ux) + v10 * ux) * (1 - uy) +
            (v01 * (1 - ux) + v11 * ux) * uy) * 2.0 - 1.0   # [-1, 1]


def procedural_detail(base_m, wx, wy, dev):
    """Slope-guided multi-octave world-anchored noise, in METERS."""
    import torch
    # Local slope from the upsampled base (per-texel gradient, 1 m cells).
    dhx = torch.zeros_like(base_m)
    dhy = torch.zeros_like(base_m)
    dhx[:, 1:-1] = (base_m[:, 2:] - base_m[:, :-2]) * 0.5
    dhy[1:-1, :] = (base_m[2:, :] - base_m[:-2, :]) * 0.5
    slope = torch.sqrt(dhx * dhx + dhy * dhy)          # m per m
    # 0.2 on flats -> 1.6 on steep rock; keeps valleys/water calm.
    amp_mod = torch.clamp(slope * 8.0, 0.2, 1.6)

    detail = torch.zeros_like(base_m)
    # (cell meters, amplitude meters) — ridged large forms, granular
    # small.  Amplitudes scaled 1/8 with the map extent (4 km world,
    # 250 m amp): flats get ~±5 cm — walkable/drivable — and steep rock
    # tops out around ±0.6 m instead of boulder-field ±5 m.
    octaves = [(64.0, 0.40), (24.0, 0.22), (8.0, 0.11),
               (3.0, 0.055), (1.0, 0.028)]
    for i, (cell, amp) in enumerate(octaves):
        n = _value_noise(wx, wy, cell, seed=101 + i * 7919)
        if cell >= 96.0:
            n = 1.0 - 2.0 * torch.abs(n)   # ridged for the big shapes
        detail = detail + n * amp
    return detail * amp_mod


# ── Catmull-Rom window upsample of the global base map ─────────────────────
def catmull_rom_window(base01, u0, v0, u1, v1, out_res, dev):
    """Sample the [u0,v0]..[u1,v1] uv window of base01 ([H,W] tensor, 0..1)
    at out_res^2 with Catmull-Rom, matching the engine's tile_creator
    filter.  Implemented with grid_sample(bicubic) — same C1 family."""
    import torch
    import torch.nn.functional as F
    ys = torch.linspace(v0, v1, out_res, device=dev)
    xs = torch.linspace(u0, u1, out_res, device=dev)
    # grid_sample wants [-1, 1] coords of the SOURCE texture.
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx * 2.0 - 1.0, gy * 2.0 - 1.0], dim=-1)[None]
    src = base01[None, None]
    out = F.grid_sample(src, grid, mode="bicubic",
                        padding_mode="border", align_corners=False)
    return out[0, 0]


# ── Tile generation ─────────────────────────────────────────────────────────
def civ_mask(rgb_u8):
    """Road/settlement mask from an albedo tile (same heuristics as the
    pipeline's semantic_height_pass): detail noise is damped there so
    roads stay drivable at 1 m resolution too."""
    from PIL import ImageFilter
    a = rgb_u8.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    sat = (mx - a.min(-1)) / (mx + 1e-6)
    road = (sat < 0.18) & (mx > 0.35) & (mx < 0.85)
    roof = ((r > g * 1.25) & (r > b * 1.25) & (r > 0.3)) | \
           ((sat < 0.25) & (mx < 0.35) & (mx > 0.15))
    m = ((road | roof) * 255).astype(np.uint8)
    m = Image.fromarray(m).filter(ImageFilter.GaussianBlur(4))
    return np.clip(np.asarray(m, dtype=np.float32) / 255.0 * 2.0, 0.0, 1.0)


def generate_tile(base01, ix, iy, dev, sr=None, civ=None):
    import torch
    res = DETAIL_TILE_RES                                  # 2049 texels
    margin = MARGIN_TEXELS
    full = res + 2 * margin
    tile_m = DETAIL_TILE_M
    cell_m = tile_m / (res - 1)                            # exactly 1 m

    # Texel centers on integer world meters: texel k of the un-margined
    # tile sits at world origin + k * 1 m (see DETAIL_TILE_RES comment).
    wx0 = -WORLD_SIZE_M / 2 + ix * tile_m - margin * cell_m
    wy0 = -WORLD_SIZE_M / 2 + iy * tile_m - margin * cell_m
    span = (full - 1) * cell_m

    # uv window into the base map at those texel-center world positions.
    u0 = (wx0 + WORLD_SIZE_M / 2) / WORLD_SIZE_M
    v0 = (wy0 + WORLD_SIZE_M / 2) / WORLD_SIZE_M
    u1 = (wx0 + span + WORLD_SIZE_M / 2) / WORLD_SIZE_M
    v1 = (wy0 + span + WORLD_SIZE_M / 2) / WORLD_SIZE_M

    base = catmull_rom_window(base01, u0, v0, u1, v1, full, dev)
    base_m = base * HEIGHT_AMP_M

    # World coordinates per texel (for world-anchored noise) — texel
    # centers exactly AT wx0 + k*cell, no half-texel offset.
    xs = wx0 + torch.arange(full, device=dev, dtype=torch.float32) * cell_m
    ys = wy0 + torch.arange(full, device=dev, dtype=torch.float32) * cell_m
    wy, wx = torch.meshgrid(ys, xs, indexing="ij")

    det = procedural_detail(base_m, wx, wy, dev)
    if civ is not None:
        import torch.nn.functional as F
        cv = torch.from_numpy(civ).to(dev)
        cv = F.interpolate(cv[None, None], size=(full, full),
                           mode="bilinear", align_corners=False)[0, 0]
        det = det * (1.0 - 0.9 * cv)     # roads/settlements stay smooth

    # ── ML height detail (Real-ESRGAN on the raw base window) ─────────
    # SR the 4 m/texel source crop to 1 m and keep only the DELTA vs the
    # smooth upsample (bounded to ±4 m: the u8 round-trip quantizes, and
    # unbounded deltas can hallucinate cliff lines).  The delta is edge-
    # feathered to zero at tile borders (SR is not bit-exact across
    # tiles); the procedural noise — which IS bit-exact — carries the
    # border band, so seams stay perfect.
    if sr is not None and sr.ok:
        import torch
        import torch.nn.functional as F
        src_px = int(round(span / WORLD_SIZE_M * base01.shape[1]))  # ~4m px
        sx0 = int(round((wx0 + WORLD_SIZE_M / 2) / WORLD_SIZE_M
                        * base01.shape[1]))
        sy0 = int(round((wy0 + WORLD_SIZE_M / 2) / WORLD_SIZE_M
                        * base01.shape[0]))
        sx0 = max(0, min(base01.shape[1] - src_px, sx0))
        sy0 = max(0, min(base01.shape[0] - src_px, sy0))
        win = base01[sy0:sy0 + src_px, sx0:sx0 + src_px]
        lo, hi = float(win.min()), float(win.max())
        if hi - lo > 1e-6:
            wn = ((win - lo) / (hi - lo) * 255.0).clamp(0, 255)
            wu8 = wn.to(torch.uint8).cpu().numpy()
            rgb = np.stack([wu8, wu8, wu8], axis=-1)
            up = sr.up4_rgb(rgb)[:, :, 0].astype(np.float32)
            up_m = (torch.from_numpy(up).to(dev) / 255.0
                    * (hi - lo) + lo) * HEIGHT_AMP_M
            smooth_m = F.interpolate(
                (win * HEIGHT_AMP_M)[None, None],
                size=up_m.shape, mode="bicubic",
                align_corners=False)[0, 0]
            # +-0.5 m: enough for SR-crisped ridges/banks, small enough
            # that u8-quantization artifacts can't hallucinate ledges.
            delta = torch.clamp(up_m - smooth_m, -0.5, 0.5)
            # Resample the delta onto the tile texel grid.
            delta = F.interpolate(delta[None, None], size=(full, full),
                                  mode="bilinear",
                                  align_corners=False)[0, 0]
            det = det + delta * _edge_feather(full, margin,
                                              MARGIN_TEXELS, dev)

    # Light thermal relax on base+detail (margin absorbs the boundary).
    h_m = base_m + det
    for _ in range(4):
        hp = torch.nn.functional.pad(h_m[None, None], (1, 1, 1, 1),
                                     mode="replicate")[0, 0]
        nb = (hp[:-2, 1:-1] + hp[2:, 1:-1] + hp[1:-1, :-2] + hp[1:-1, 2:]) * 0.25
        h_m = h_m + torch.clamp(nb - h_m, -0.15, 0.15) * 0.5

    h_m = h_m[margin:margin + res, margin:margin + res]
    h01 = torch.clamp(h_m / HEIGHT_AMP_M, 0.0, 1.0)
    return h01.cpu().numpy()


def generate_color_tile(color_np, ix, iy, sr=None):
    """1 m albedo detail tile (2048^2).  Real-ESRGAN x4 of the tile's
    512^2 window of the global colour map when available (with source
    context margin + border feather back to bicubic so independently
    processed neighbours agree at shared edges); plain bicubic
    otherwise."""
    H, W = color_np.shape[:2]
    tile_px = W // TILES_PER_SIDE                        # 512 for 8k
    m = 32                                               # source margin px
    x0, y0 = ix * tile_px, iy * tile_px
    rows = np.clip(np.arange(y0 - m, y0 + tile_px + m), 0, H - 1)
    cols = np.clip(np.arange(x0 - m, x0 + tile_px + m), 0, W - 1)
    win = color_np[np.ix_(rows, cols)]                   # (tile+2m)^2

    if sr is not None and sr.ok and tile_px * 4 == 2048:
        up = sr.up4_rgb(win).astype(np.float32)          # x4
        bic = np.asarray(Image.fromarray(win).resize(
            (win.shape[1] * 4, win.shape[0] * 4), Image.BICUBIC),
            dtype=np.float32)
        # Feather SR -> bicubic over the outer 64 out-px: SR isn't
        # bit-exact across tiles; bicubic is, so borders match.
        n = up.shape[0]
        ramp = np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n))
                       / 64.0, 0.0, 1.0).astype(np.float32)
        f = (ramp[None, :] * ramp[:, None])[..., None]
        blend = up * f + bic * (1.0 - f)
        out = blend[m * 4:m * 4 + 2048, m * 4:m * 4 + 2048]
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))

    crop = np.asarray(Image.fromarray(win).resize(
        ((tile_px + 2 * m) * 4,) * 2, Image.BICUBIC))
    return Image.fromarray(crop[m * 4:m * 4 + 2048, m * 4:m * 4 + 2048])


def save_u16_atomic(a01, path):
    a = (a01 * 65535.0 + 0.5).astype(np.uint16)
    tmp = path + ".tmp"
    Image.fromarray(a, mode="I;16").save(tmp, format="PNG")
    os.replace(tmp, path)      # atomic on the same volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True,
                    help="the installed BASE heightmap (16-bit PNG, 8192^2)")
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--color", default=None,
                    help="global color map; if given, also emits "
                         "tile_XX_YY_color.png albedo detail tiles")
    ap.add_argument("--idle-exit", type=float, default=900.0,
                    help="exit after this many seconds with no requests "
                         "(the engine respawns the worker on demand)")
    ap.add_argument("--ml", choices=["off", "albedo", "full"],
                    default="full",
                    help="Real-ESRGAN x4 detail: 'full' = heightmap + "
                         "albedo, 'albedo' = colour only (heights stay "
                         "procedural), 'off' = bicubic/procedural")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.tiles_dir, exist_ok=True)
    # The editor can re-apply/install a terrain WHILE a worker is starting,
    # briefly replacing the map file.  Retry a few times; if the map is
    # really gone (superseded generation), exit quietly instead of dying
    # with a traceback — the editor will spawn a fresh worker for the new
    # map.
    base_np = None
    for _attempt in range(5):
        try:
            base_np = np.array(Image.open(args.map).convert("I"),
                               dtype=np.float32) / 65535.0
            break
        except (FileNotFoundError, OSError):
            time.sleep(1.0)
    if base_np is None:
        print(f"[detail-worker] map vanished ({args.map}) — superseded; "
              "exiting cleanly")
        return
    base01 = torch.from_numpy(base_np).to(dev)
    color_np = None
    if args.color and os.path.exists(args.color):
        color_np = np.asarray(Image.open(args.color).convert("RGB"))

    sr = MlUpscaler() if args.ml != "off" else None
    sr_height = sr if args.ml == "full" else None

    alive = os.path.join(args.tiles_dir, "worker.alive")
    print(f"[detail-worker] up  map={args.map}  dev={dev}  "
          f"tiles={TILES_PER_SIDE}x{TILES_PER_SIDE} @ {DETAIL_TILE_RES}^2")
    last_work = time.time()
    while True:
        # The editor deletes the tiles dir when the terrain is re-applied;
        # treat that as "this worker is superseded", not a crash.
        try:
            os.makedirs(args.tiles_dir, exist_ok=True)
            with open(alive, "w") as f:
                f.write(str(os.getpid()))
        except OSError:
            print("[detail-worker] tiles dir revoked — superseded; exiting")
            return
        reqs = sorted(fn for fn in os.listdir(args.tiles_dir)
                      if fn.startswith("req_"))
        if not reqs:
            if time.time() - last_work > args.idle_exit:
                print("[detail-worker] idle — exiting")
                break
            time.sleep(0.2)
            continue
        for fn in reqs:
            try:
                _, sx, sy = fn.split("_")
                ix, iy = int(sx), int(sy)
            except ValueError:
                os.remove(os.path.join(args.tiles_dir, fn))
                continue
            out = os.path.join(args.tiles_dir, f"tile_{ix:02d}_{iy:02d}.png")
            if not os.path.exists(out):
                t0 = time.time()
                civ = None
                if color_np is not None:
                    # Colour FIRST: the engine keys residency on the
                    # height PNG, so the colour must be complete when
                    # the height tile appears — and its road/settlement
                    # mask damps the height detail noise.
                    cimg = generate_color_tile(color_np, ix, iy, sr=sr)
                    civ = civ_mask(np.asarray(cimg))
                    cout = os.path.join(
                        args.tiles_dir, f"tile_{ix:02d}_{iy:02d}_color.png")
                    ctmp = cout + ".tmp"
                    cimg.save(ctmp, format="PNG")
                    os.replace(ctmp, cout)
                h01 = generate_tile(base01, ix, iy, dev, sr=sr_height,
                                    civ=civ)
                save_u16_atomic(h01, out)
                print(f"[detail-worker] tile ({ix},{iy}) in "
                      f"{time.time() - t0:.2f}s")
            try:
                os.remove(os.path.join(args.tiles_dir, fn))
            except OSError:
                pass
            last_work = time.time()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
