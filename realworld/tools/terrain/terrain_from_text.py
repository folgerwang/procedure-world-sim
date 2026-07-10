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

# Prompt scaffold: FLUX responds well to explicit "heightmap" framing plus
# photographic-DEM vocabulary.  The user's text slots into the middle.
# The view language is deliberately redundant (satellite / nadir / straight
# down / orthographic / no horizon) — diffusion models drift toward scenic
# perspective renders of "mountains" without it, and a perspective image is
# useless as a heightfield.
PROMPT_TEMPLATE = (
    "satellite view terrain heightmap seen directly from above, "
    "nadir aerial view, straight down 90 degrees overhead, "
    "orthographic map projection, {user_prompt}, "
    "grayscale digital elevation model, white is high elevation, "
    "black is low elevation, smooth gradients, satellite DEM data, "
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
    "{user_prompt}, nadir aerial orthophoto seen straight down, "
    "natural earth colors, vegetation in valleys, exposed rock and snow "
    "on high peaks, water in the lowest channels, "
    "no perspective, no horizon, no sky, no clouds, no text, no borders"
)


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
           rain=0.01, evap=0.5, capacity=0.08, detail_keep=0.25):
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

    WORLD_SIZE = 16384.0   # kWorldMapSize (global_definition.glsl.h)
    HEIGHT_AMP = 2000.0    # tile_creator.comp heightmap multiplier

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
        with open(color_png, "rb") as f:
            img_b = pad4(f.read())

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
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--gen-size", type=int, default=1024,
                    help="FLUX RENDER resolution.  The heightfield is upsampled "
                         "from this to --size, so lowering it cuts FLUX VRAM + "
                         "time ~quadratically with almost no loss on a heightmap "
                         "(1024 => ~4x less activation memory than 2048).")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the stage-2 terrain conversion model")
    ap.add_argument("--mesh-res", type=int, default=256,
                    help="terrain mesh grid resolution (verts per side) for "
                         "the .glb export; 0 disables the mesh")
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

    raw_png = args.from_image
    if not raw_png:
        raw_png = os.path.join(tempfile.gettempdir(), "terrain_raw.png")
        run_flux(PROMPT_TEMPLATE.format(user_prompt=args.prompt),
                 raw_png, args.gen_size, args.steps, args.seed,
                 label="heightmap", prog_lo=0.02, prog_hi=hm_hi)

    print("[terrain] stage 2: heightmap -> terrain conversion")
    report(hm_hi + 0.02, "terrain conversion (erosion)")
    h = image_to_heightfield(raw_png, args.size)
    if not args.no_refine:
        h = refine(h)
    report(hm_hi + 0.07, "terrain conversion done")

    out = args.out if os.path.isabs(args.out) \
        else os.path.join(REPO_ROOT, args.out)

    # ── Stage 3 (optional): colour satellite map, conditioned on the
    # refined heightfield so its layout matches the terrain.  Runs BEFORE
    # the final heightmap write: the editor treats the heightmap PNG's
    # existence as "job complete", so it must be the LAST artifact.
    if args.color:
        stem, _ = os.path.splitext(out)
        color_out = stem + "_color.png"
        ref8 = os.path.join(tempfile.gettempdir(), "terrain_ref8.png")
        Image.fromarray(
            (h.numpy() * 255.0 + 0.5).astype(np.uint8), mode="L"
        ).convert("RGB").save(ref8)
        run_flux(COLOR_PROMPT_TEMPLATE.format(
                     user_prompt=args.prompt or "natural terrain"),
                 color_out, args.gen_size, args.steps, args.seed,
                 ref_image=ref8, label="color satellite",
                 prog_lo=0.58, prog_hi=0.95)
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

    report(0.97, "writing terrain map")
    # World-height scaling happens HERE (not in the engine): the refined
    # heightfield spans the full [0,1] range, which the tile creator maps
    # to 0..2000 m.  NOTE: applied after the colour pass, whose FLUX
    # conditioning wants the full-contrast reference.
    scale = max(0.0, min(1.0, args.height_scale))
    if scale < 1.0:
        h = h * scale
        print(f"[terrain] height scale {scale:g} "
              f"(peaks ~{2000.0 * scale:.0f} m)")
    save_u16(h, out)

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
