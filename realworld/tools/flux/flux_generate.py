#!/usr/bin/env python3
"""
flux_generate.py -- single-image text-to-image generator (4-bit FLUX.2).

Launched by the editor's Content Browser (Menu::launchImageGen). It can be
started by ANY python (the editor under Visual Studio spawns a plain
`python`): _reexec_in_venv() re-launches it under tools/flux/.venv so the
heavy deps (diffusers/torch/...) are importable.

Output layout (keeps the content folder clean):
  <folder>/<name>.png                      final image (shows in the grid)
  <folder>/.flux_tmp/<name>.png.seed       actual integer seed used (early)
  <folder>/.flux_tmp/<name>.png.progress   float 0..1, updated per step
  <folder>/.flux_tmp/<name>.png.err        traceback on failure
"""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

DEFAULT_MODEL = "diffusers/FLUX.2-dev-bnb-4bit"

_PIPE = None
_PIPE_KEY = None


def _venv_python() -> Path:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return Path(__file__).resolve().parent / ".venv" / sub


def _reexec_in_venv():
    # If we were started by some other python (e.g. the editor spawns a plain
    # `python` with no diffusers), re-run this exact script + args under the
    # tools/flux/.venv interpreter. No-op when already in the venv, or if the
    # venv has not been created yet (then the normal import error guides setup).
    if os.environ.get("_FLUX_REEXEC") == "1":
        return
    venv_py = _venv_python()
    if not venv_py.exists():
        return
    try:
        if Path(sys.executable).resolve() == venv_py.resolve():
            return
    except Exception:
        pass
    env = dict(os.environ, _FLUX_REEXEC="1")
    print("[flux] re-exec under venv python: " + str(venv_py), flush=True)
    rc = subprocess.run(
        [str(venv_py), os.path.abspath(__file__), *sys.argv[1:]], env=env).returncode
    sys.exit(rc)


def _round16(v: int) -> int:
    """FLUX latents need dimensions divisible by 16."""
    return max(16, (int(v) // 16) * 16)


def _tmp_dir(out: Path) -> Path:
    d = out.parent / ".flux_tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sidecar(out: Path, suffix: str) -> Path:
    """Path of a sidecar file inside the <folder>/.flux_tmp temp dir."""
    return _tmp_dir(out) / (out.name + suffix)


def _write_text_atomic(path: Path, text: str):
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _from_pretrained(model: str, **extra):
    """Load the pipeline, tolerant of the dtype/torch_dtype kwarg rename."""
    import torch
    from diffusers import DiffusionPipeline
    try:
        return DiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.bfloat16, **extra)
    except TypeError:
        return DiffusionPipeline.from_pretrained(
            model, dtype=torch.bfloat16, **extra)


def load_pipeline(model: str, offload: bool):
    global _PIPE, _PIPE_KEY
    key = (model, offload)
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE

    import torch

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not offload:
        print("[flux] no CUDA device -> forcing --offload (CPU, slow)", flush=True)
        offload = True

    print("[flux] loading %s (4-bit NF4)%s ..."
          % (model, " [offload]" if offload else ""), flush=True)

    if offload:
        pipe = _from_pretrained(model)
        if has_cuda:
            pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        except Exception:
            pass
    else:
        pipe = _from_pretrained(model, device_map="cuda")

    if has_cuda:
        free, total = torch.cuda.mem_get_info()
        print("[flux] CUDA VRAM: %.1f GB free / %.1f GB total"
              % (free / 1e9, total / 1e9), flush=True)

    _PIPE, _PIPE_KEY = pipe, key
    return pipe


def generate(args) -> None:
    import torch

    out = Path(args.out)
    progress_path = sidecar(out, ".progress")
    seed_path = sidecar(out, ".seed")

    if args.seed is None or args.seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    else:
        seed = int(args.seed)
    _write_text_atomic(seed_path, str(seed))
    _write_text_atomic(progress_path, "0.0000")

    pipe = load_pipeline(args.model, args.offload)

    out_w, out_h = max(8, int(args.width)), max(8, int(args.height))
    rmin = max(0, int(getattr(args, "render_min", 512)))
    w, h = _round16(max(out_w, rmin)), _round16(max(out_h, rmin))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    print("[flux] prompt: %r" % args.prompt, flush=True)
    print("[flux] %dx%d steps=%d guidance=%s seed=%d"
          % (w, h, args.steps, args.guidance, seed), flush=True)

    def _cb(pipe_, step, t, cbk):
        frac = (step + 1) / float(args.steps)
        _write_text_atomic(progress_path, "%.4f" % max(0.0, min(1.0, frac)))
        print("[flux] step %d/%d" % (step + 1, args.steps), flush=True)
        return cbk

    kwargs = dict(
        prompt=args.prompt,
        width=w,
        height=h,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    try:
        image = pipe(callback_on_step_end=_cb, **kwargs).images[0]
    except TypeError:
        image = pipe(**kwargs).images[0]

    out.parent.mkdir(parents=True, exist_ok=True)
    # Scratch keeps the OUTPUT extension so PIL infers the format (a ".tmp"
    # suffix raised "unknown file extension"); lives in .flux_tmp, then is
    # atomically renamed to the final <name>.
    # Downscale to the requested (icon) size if we rendered larger.
    if (w, h) != (out_w, out_h):
        from PIL import Image as _PILImage
        try:
            _resample = _PILImage.Resampling.LANCZOS
        except AttributeError:
            _resample = _PILImage.LANCZOS
        image = image.resize((out_w, out_h), _resample)
    tmp = _tmp_dir(out) / (out.stem + ".partial" + out.suffix)
    image.save(tmp)
    os.replace(tmp, out)                  # atomic publish into the content folder
    try:
        progress_path.unlink()            # PNG existing is the "done" signal
    except FileNotFoundError:
        pass
    print("[flux] wrote %s (seed=%d)" % (out, seed), flush=True)


def main():
    _reexec_in_venv()

    ap = argparse.ArgumentParser(description="4-bit FLUX.2 text-to-image generator.")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="integer seed; omit or pass <0 for a random seed.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--render-min", type=int, default=512,
                    help="diffuse at least NxN, then downscale to "
                         "--width/--height. Keeps tiny icon sizes "
                         "(32-256) crisp instead of latent-res noise.")
    # CPU offload is the DEFAULT: FLUX.2's Mistral-3 text encoder is large, so
    # device_map="cuda" fills even a 32-34GB card (0 GB free -> OOM). Offload
    # streams components GPU<->CPU. Pass --no-offload only on huge-VRAM setups.
    ap.add_argument("--offload", dest="offload", action="store_true", default=True,
                    help="CPU offload (DEFAULT; needed for FLUX.2).")
    ap.add_argument("--no-offload", dest="offload", action="store_false",
                    help="keep all components on GPU (device_map=cuda); needs huge VRAM.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.selftest and not args.prompt:
        args.prompt = "a small red cube on a white table, studio lighting"
    if not args.prompt:
        sys.exit("error: one of --prompt / --prompt-file is required")

    out = Path(args.out)
    err_path = sidecar(out, ".err")
    progress_path = sidecar(out, ".progress")
    seed_path = sidecar(out, ".seed")
    for p in (err_path, progress_path, seed_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    try:
        generate(args)
    except Exception as e:  # noqa: BLE001 -- surface everything to the editor
        tb = traceback.format_exc()
        msg = str(e).lower()
        hint = ""
        if "out of memory" in msg or "cuda out of memory" in msg:
            hint = ("\n\nHINT: out of VRAM. Re-run with --offload or pick a smaller "
                    "size. 4-bit FLUX.2 needs ~18-20 GB without offload.")
        elif any(k in msg for k in ("gated", "401", "403", "restricted")):
            hint = ("\n\nHINT: FLUX.2-dev weights are gated. Accept the licence and run "
                    "huggingface-cli login (or set HF_TOKEN), then re-run setup_flux.py.")
        elif "no module named" in msg:
            hint = ("\n\nHINT: deps missing. The venv at tools/flux/.venv may not be "
                    "set up -- run: python realworld/tools/flux/setup_flux.py")
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(tb + hint, encoding="utf-8")
        print(tb + hint, file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
