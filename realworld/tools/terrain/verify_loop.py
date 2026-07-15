#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terrain generation verification loop.

Closes the loop around the procedural terrain generator:

    generate heightmap  ->  engine applies + dumps a clean viewport frame
                        ->  Claude analyzes the frame against a rubric
                        ->  pass? done.  fail? Claude adjusts the prompt /
                            height-scale and we regenerate.

Each iteration launches the engine headless in one-shot "verify-dump" mode:

    RealWorld.exe --verify-apply <heightmap.png> [--verify-color <png>] \
                  --verify-dump <out.png> --verify-delay <sec>

The engine applies the heightmap, waits `--verify-delay` seconds, dumps a clean
(UI-free) render-buffer frame to <out.png>, writes
`screenshots/terrain_verify_done.txt`, and closes itself. We then read the PNG
and hand it to Claude for a verdict.

Prerequisites
-------------
  * A built RealWorld.exe (pass its path with --engine-exe; default ./RealWorld.exe).
  * ANTHROPIC_API_KEY must be set in the environment.
  * Run from the `realworld/` directory so relative paths resolve
    (content/, screenshots/, tools/terrain/).

Examples
--------
  python tools/terrain/verify_loop.py --prompt "volcanic island with a caldera lake"
  python tools/terrain/verify_loop.py --prompt "desert canyon" --max-iters 5 --color
  python tools/terrain/verify_loop.py --analyze-only screenshots/frame_latest.png \\
        --prompt "snowy alpine range"
"""

import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GEN_SCRIPT = Path("tools/terrain/terrain_from_text.py")
DONE_MARKER = Path("screenshots/terrain_verify_done.txt")
DEFAULT_ENGINE_EXE = "RealWorld.exe"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"

# The rubric the analyzer scores against.  Keep in sync with the README.
RUBRIC = {
    "not_degenerate": "The frame shows real terrain, not a flat/blank/all-black "
                      "image, a failed load, NaN speckle, or obvious corruption.",
    "plausible_landforms": "Elevation reads as natural: coherent mountains, "
                           "valleys and slopes, no impossible spikes, no hard "
                           "unnatural stair-stepping or repeating stamped tiles.",
    "matches_prompt": "The terrain reflects the requested description "
                      "(biome, dominant landform, notable features).",
    "no_artifacts": "No obvious render artifacts: seams/tiling, holes, "
                    "z-fighting, stretched or corrupted texturing.",
}

SYSTEM_PROMPT = (
    "You are a technical art director reviewing procedurally generated 3D game "
    "terrain. You are given ONE screenshot of a terrain rendered in-engine and "
    "the text prompt it was generated from. Judge whether the terrain 'makes "
    "sense' using the rubric. Be strict but fair: minor stylistic differences "
    "are fine; reject only real problems. Reply with STRICT JSON only, no prose "
    "outside the JSON."
)


# --------------------------------------------------------------------------- #
# Engine one-shot launch
# --------------------------------------------------------------------------- #

def parse_kv(path):
    kv = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
    except OSError:
        pass
    return kv


def run_engine_oneshot(engine_exe, height_png, color_png, capture_png,
                       delay_s, timeout_s):
    """Launch the engine in verify-dump mode; it self-closes after the dump.

    Returns the resolved capture path on success, or None on failure/timeout.
    """
    # Clear a stale done-marker so we only react to this run's.
    try:
        DONE_MARKER.unlink()
    except FileNotFoundError:
        pass

    cmd = [
        str(engine_exe),
        "--verify-apply", str(height_png),
        "--verify-dump", str(capture_png),
        "--verify-delay", f"{delay_s:.1f}",
    ]
    if color_png:
        cmd.extend(["--verify-color", str(color_png)])
    print(f"  [engine] {' '.join(cmd)}", flush=True)

    try:
        proc = subprocess.Popen(cmd)
    except OSError as e:
        print(f"  [engine] could not launch {engine_exe}: {e}", flush=True)
        return None

    # The engine quits itself once the dump is written; wait for that, with a
    # hard timeout as a safety net (then terminate it).
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        if DONE_MARKER.exists():
            # Give the process a moment to exit on its own.
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
            break
        time.sleep(0.5)
    else:
        print("  [engine] TIMEOUT; terminating.", flush=True)
        proc.terminate()
        return None

    if DONE_MARKER.exists():
        cap = parse_kv(DONE_MARKER).get("capture") or str(capture_png)
    else:
        cap = str(capture_png)
    return cap if Path(cap).exists() else None


# --------------------------------------------------------------------------- #
# Terrain generation
# --------------------------------------------------------------------------- #

def generate_terrain(prompt, out_png, height_scale, seed, color, extra_args):
    """Run terrain_from_text.py synchronously. Returns (ok, color_png_or_none)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    prompt_file = out_png.with_suffix(".prompt.txt")
    prompt_file.write_text(prompt, encoding="utf-8")

    cmd = [
        sys.executable, str(GEN_SCRIPT),
        "--prompt-file", str(prompt_file),
        "--out", str(out_png),
        "--height-scale", f"{height_scale:.3f}",
        "--seed", str(seed),
    ]
    if color:
        cmd.append("--color")
    cmd.extend(extra_args)

    print(f"  [gen] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [gen] FAILED (exit {result.returncode})", flush=True)
        return False, None
    if not out_png.exists():
        print(f"  [gen] FAILED: output not written ({out_png})", flush=True)
        return False, None

    color_png = out_png.with_name(out_png.stem + "_color.png")
    return True, (str(color_png) if color and color_png.exists() else "")


# --------------------------------------------------------------------------- #
# Claude analysis
# --------------------------------------------------------------------------- #

def load_image_b64(png_path, max_dim=1568):
    """Read a PNG and return (media_type, base64). Downscale if Pillow is present."""
    data = Path(png_path).read_bytes()
    media_type = "image/png"
    try:
        from PIL import Image  # optional
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_dim / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        data = buf.getvalue()
        media_type = "image/jpeg"
    except Exception:
        pass  # send the raw PNG
    return media_type, base64.b64encode(data).decode("ascii")


def build_user_prompt(prompt, height_scale):
    rubric_lines = "\n".join(f"  - {k}: {v}" for k, v in RUBRIC.items())
    return (
        f"Terrain prompt: \"{prompt}\"\n"
        f"Current height-scale: {height_scale:.3f} (x250 world units)\n\n"
        f"Rubric (each criterion is pass/fail):\n{rubric_lines}\n\n"
        "Analyze the screenshot and return STRICT JSON with EXACTLY these keys:\n"
        "{\n"
        '  "pass": <true|false>,   // true only if ALL rubric criteria pass\n'
        '  "scores": {\n'
        '     "not_degenerate": <true|false>,\n'
        '     "plausible_landforms": <true|false>,\n'
        '     "matches_prompt": <true|false>,\n'
        '     "no_artifacts": <true|false>\n'
        "  },\n"
        '  "issues": [<short strings naming concrete problems, [] if none>],\n'
        '  "analysis": "<2-4 sentence explanation of what you see>",\n'
        '  "suggested_prompt": "<a revised generation prompt likely to fix the '
        'issues; repeat the original if it already passes>",\n'
        '  "suggested_height_scale": <float in 0.05..1.0>\n'
        "}\n"
    )


def analyze_frame(api_key, model, png_path, prompt, height_scale):
    media_type, b64 = load_image_b64(png_path)
    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": build_user_prompt(prompt, height_scale)},
            ],
        }],
    }
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"Anthropic API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Anthropic API: {e.reason}")

    text = "".join(
        blk.get("text", "") for blk in body.get("content", [])
        if blk.get("type") == "text"
    ).strip()
    return parse_verdict(text)


def parse_verdict(text):
    """Extract the JSON object from the model reply, tolerant of code fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"No JSON object in model reply:\n{text}")
    verdict = json.loads(s[start:end + 1])
    verdict.setdefault("pass", False)
    verdict.setdefault("issues", [])
    verdict.setdefault("analysis", "")
    verdict.setdefault("suggested_prompt", "")
    verdict.setdefault("suggested_height_scale", None)
    return verdict


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Terrain generation verify loop")
    ap.add_argument("--prompt", required=True, help="initial terrain description")
    ap.add_argument("--max-iters", type=int, default=4,
                    help="max generate/verify iterations (default 4)")
    ap.add_argument("--height-scale", type=float, default=0.25)
    ap.add_argument("--color", action="store_true",
                    help="also generate the albedo/color map")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"analysis model (default {DEFAULT_MODEL})")
    ap.add_argument("--engine-exe", default=DEFAULT_ENGINE_EXE,
                    help="path to RealWorld.exe (default ./RealWorld.exe)")
    ap.add_argument("--delay", type=float, default=10.0,
                    help="seconds the engine waits after apply before dumping")
    ap.add_argument("--engine-timeout", type=float, default=180.0,
                    help="seconds to wait for the engine to dump and exit")
    ap.add_argument("--out-dir", default="content/terrain/.verify_runs",
                    help="where heightmaps + captures for this run are written")
    ap.add_argument("--report", default=None,
                    help="path for the JSON run report (default under out-dir)")
    ap.add_argument("--seed", type=int, default=-1,
                    help="starting seed (-1 = random each iteration)")
    ap.add_argument("--analyze-only", metavar="PNG", default=None,
                    help="skip gen+engine; just analyze this PNG once and exit")
    ap.add_argument("--gen-args", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to terrain_from_text.py")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY is not set in the environment.")

    # --- analyze-only fast path -------------------------------------------
    if args.analyze_only:
        verdict = analyze_frame(api_key, args.model, args.analyze_only,
                                args.prompt, args.height_scale)
        print(json.dumps(verdict, indent=2))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else out_dir / "verify_report.json"

    prompt = args.prompt
    height_scale = args.height_scale
    run = {"initial_prompt": args.prompt, "model": args.model, "iterations": []}

    passed = False
    for i in range(1, args.max_iters + 1):
        seed = args.seed if args.seed >= 0 else random.randint(1, 2_000_000_000)
        print(f"\n=== iteration {i}/{args.max_iters} "
              f"(seed={seed}, height_scale={height_scale:.3f}) ===", flush=True)
        print(f"  prompt: {prompt}", flush=True)

        height_png = out_dir / f"iter{i:02d}.png"
        capture_png = Path("screenshots") / f"terrain_verify_iter{i:02d}.png"

        ok, color_png = generate_terrain(
            prompt, height_png, height_scale, seed, args.color, args.gen_args)
        if not ok:
            run["iterations"].append({"iter": i, "error": "generation_failed"})
            break

        capture = run_engine_oneshot(
            args.engine_exe, height_png, color_png, capture_png,
            args.delay, args.engine_timeout)
        if not capture:
            print("  [engine] no capture produced (launch/timeout/dump "
                  "failed).", flush=True)
            run["iterations"].append({"iter": i, "error": "engine_no_capture"})
            break

        print(f"  [analyze] {capture} via {args.model}...", flush=True)
        try:
            verdict = analyze_frame(api_key, args.model, capture,
                                    prompt, height_scale)
        except RuntimeError as e:
            print(f"  [analyze] {e}", flush=True)
            run["iterations"].append({"iter": i, "error": str(e)})
            break

        verdict_pass = bool(verdict.get("pass"))
        print(f"  [verdict] pass={verdict_pass} "
              f"scores={verdict.get('scores')}", flush=True)
        if verdict.get("issues"):
            print(f"  [verdict] issues: {verdict['issues']}", flush=True)
        print(f"  [verdict] {verdict.get('analysis', '')}", flush=True)

        run["iterations"].append({
            "iter": i, "seed": seed, "prompt": prompt,
            "height_scale": height_scale, "height_png": str(height_png),
            "capture": capture, "verdict": verdict,
        })

        if verdict_pass:
            passed = True
            print(f"\n[OK] terrain accepted on iteration {i}.", flush=True)
            break

        # Claude adjusts prompt / params for the next attempt.
        new_prompt = (verdict.get("suggested_prompt") or "").strip()
        if new_prompt:
            prompt = new_prompt
        new_hs = verdict.get("suggested_height_scale")
        if isinstance(new_hs, (int, float)) and 0.05 <= float(new_hs) <= 1.0:
            height_scale = float(new_hs)
        print(f"  [adjust] next prompt: {prompt}", flush=True)
        print(f"  [adjust] next height_scale: {height_scale:.3f}", flush=True)

    run["passed"] = passed
    report_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nReport written -> {report_path}", flush=True)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
