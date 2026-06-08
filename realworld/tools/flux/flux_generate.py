#!/usr/bin/env python3
"""
flux_generate.py - single-shot FLUX.2-klein-4B text-to-image generator.

Invoked per-image by the editor (Content Browser -> right-click -> Generate).
Designed to SHARE the GPU with the running game engine, so it keeps the memory
footprint small:

  * Text encoder (Qwen3) runs on the CPU - it only encodes the prompt once per
    image, so the latency cost is small and it frees its VRAM for the engine.
  * Diffusion transformer runs on the GPU in bf16 (~10 GB peak). int8 quant is
    available but produces noise on this DiT, so the default is bf16; pass
    --transformer-quant 4bit (NF4) if you need a smaller footprint.

On failure it writes "<out>.err" with the traceback so the C++ side can detect
and report the error without parsing stdout.

Usage:
    python flux_generate.py --prompt-file prompt.txt --out img.png \
        --width 1024 --height 1024 --steps 28 --guidance 4.0 --seed -1

  Reference-image conditioning (FLUX.2's native ControlNet-style capability,
  no extra weights):
    python flux_generate.py --prompt-file p.txt --out img.png \
        --ref-image character.png --ref-image pose.png
  The prompt describes how to use the references (edit / restyle / keep the
  same subject in a new pose / match a layout); up to 4 are accepted.

Notes:
  * Needs diffusers >=0.37.0.dev0 (Flux2KleinPipeline / Qwen3) - install from
    source: pip install -U "git+https://github.com/huggingface/diffusers".
  * Weights are gated: accept the licence at
    https://huggingface.co/black-forest-labs/FLUX.2-klein-4B and log in
    (huggingface-cli login or set HF_TOKEN) before first use.
"""
import argparse
import os
import sys
import traceback

# Silence the Windows symlink cache warning from huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _fail(out_path: str, msg: str) -> None:
    """Record the error next to the requested output and exit non-zero."""
    try:
        with open(out_path + ".err", "w", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass
    sys.stderr.write(msg + "\n")
    sys.exit(1)


def _round16(v: int) -> int:
    """FLUX wants spatial dims that are multiples of 16 (vae_scale_factor*2)."""
    v = max(256, min(2048, int(v)))
    return (v // 16) * 16


def _move(obj, device):
    """Recursively move tensors (in tuples/lists/ModelOutput/dict) to device."""
    import torch
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move(x, device) for x in obj)
    if hasattr(obj, "keys"):
        for k in list(obj.keys()):
            obj[k] = _move(obj[k], device)
        return obj
    return obj


def pin_encoder_to_cpu(pipe, out_device):
    """Keep pipe.text_encoder on CPU but transparently accept/return GPU tensors,
    by patching its forward() in place (keeps the real class so diffusers'
    isinstance-based encode path still works)."""
    import torch
    enc = pipe.text_encoder
    orig_forward = enc.forward

    def patched(*args, **kwargs):
        to_cpu = lambda x: x.to("cpu") if torch.is_tensor(x) else x
        args = tuple(to_cpu(a) for a in args)
        kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
        return _move(orig_forward(*args, **kwargs), out_device)

    enc.forward = patched


def force_cuda_execution_device(pipe):
    """Make pipe._execution_device always report CUDA so the denoise loop runs on
    the GPU even though the text encoder lives on the CPU."""
    import torch
    base = type(pipe)
    pipe.__class__ = type(
        base.__name__ + "_CudaExec", (base,),
        {"_execution_device": property(lambda self: torch.device("cuda"))},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None,
                    help="read the prompt from this UTF-8 file (avoids shell "
                         "escaping)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--ref-image", action="append", default=[],
                    help="reference image path for FLUX.2's native image "
                         "conditioning (repeatable, up to 4).  The prompt "
                         "describes how to use the reference(s): edits, "
                         "style transfer, layout guidance — klein's "
                         "built-in ControlNet-style capability, no extra "
                         "weights needed.")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random")
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--transformer-quant", choices=["none", "fp8", "int8", "4bit"],
                    default="fp8", help="GPU transformer quant. Default fp8 "
                    "(quanto float8, ~half VRAM, clean). none=bf16 (~10GB); "
                    "int8 produces noise on this DiT; 4bit=NF4 (smallest).")
    ap.add_argument("--encoder-quant", choices=["none", "int8", "int4"],
                    default="none", help="CPU encoder quant (optimum-quanto)")
    args = ap.parse_args()

    out = args.out
    try:
        if os.path.exists(out + ".err"):
            os.remove(out + ".err")
    except Exception:
        pass

    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except Exception as e:
            _fail(out, f"failed to read prompt file: {e}")
    if not prompt:
        _fail(out, "empty prompt")

    w, h = _round16(args.width), _round16(args.height)

    try:
        import torch
        if not torch.cuda.is_available():
            _fail(out, "CUDA GPU not available - FLUX.2-klein-4B needs an NVIDIA GPU.")

        from diffusers import DiffusionPipeline

        # ── transformer: int8/4bit on the GPU (bitsandbytes, load-time) ─────
        # (fp8 is quanto and is applied post-load, after the device split.)
        pre_transformer = None
        if args.transformer_quant in ("int8", "4bit"):
            from diffusers import BitsAndBytesConfig
            try:
                from diffusers import Flux2Transformer2DModel
            except ImportError:
                from diffusers.models import Flux2Transformer2DModel
            if args.transformer_quant == "int8":
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            else:
                bnb = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)
            print(f"[flux] loading transformer in {args.transformer_quant} "
                  f"(bitsandbytes, GPU)...", flush=True)
            pre_transformer = Flux2Transformer2DModel.from_pretrained(
                args.model, subfolder="transformer",
                quantization_config=bnb, torch_dtype=torch.bfloat16)

        # ── pipeline via auto-loader (picks Flux2KleinPipeline for klein) ────
        print(f"[flux] loading {args.model} ...", flush=True)
        load_kwargs = {"torch_dtype": torch.bfloat16}
        if pre_transformer is not None:
            load_kwargs["transformer"] = pre_transformer
        pipe = DiffusionPipeline.from_pretrained(args.model, **load_kwargs)

        # ── device split: encoder->CPU, transformer+VAE->GPU ────────────────
        pipe.vae = pipe.vae.to("cuda")
        pipe.text_encoder = pipe.text_encoder.to("cpu")
        if pre_transformer is None:
            pipe.transformer = pipe.transformer.to("cuda")

        # fp8 (quanto): quantize the on-GPU bf16 transformer in place. float8
        # keeps an exponent+mantissa, so unlike int8 it stays clean. ~half VRAM.
        if args.transformer_quant == "fp8":
            from optimum.quanto import quantize, freeze, qfloat8
            print("[flux] quantizing transformer to fp8 (quanto)...", flush=True)
            quantize(pipe.transformer, weights=qfloat8)
            freeze(pipe.transformer)

        # Optional CPU encoder quant (klein's Qwen3 encoder is small, so the
        # default of bf16-on-CPU is fine; quantize only if explicitly asked).
        if args.encoder_quant != "none":
            from optimum.quanto import quantize, freeze, qint8, qint4
            quantize(pipe.text_encoder,
                     weights=qint8 if args.encoder_quant == "int8" else qint4)
            freeze(pipe.text_encoder)

        pin_encoder_to_cpu(pipe, torch.device("cuda"))
        force_cuda_execution_device(pipe)

        gen = None
        if args.seed >= 0:
            gen = torch.Generator(device="cuda").manual_seed(int(args.seed))

        # ── Reference images (FLUX.2 native image conditioning) ─────────────
        # Loaded as PIL and handed to the pipeline's `image` argument; the
        # reference tokens join the denoise pass so the prompt can steer
        # edits / style / structure relative to them.
        ref_images = None
        if args.ref_image:
            from PIL import Image as PILImage
            ref_images = []
            for rp in args.ref_image[:4]:
                if not os.path.exists(rp):
                    _fail(out, f"reference image not found: {rp}")
                img = PILImage.open(rp).convert("RGB")
                # Keep references modest — they ride through the VAE and
                # attention; gigantic inputs only cost VRAM/time.
                img.thumbnail((1024, 1024), PILImage.LANCZOS)
                ref_images.append(img)
            print(f"[flux] conditioning on {len(ref_images)} reference "
                  f"image(s)", flush=True)

        print(f"[flux] generating {w}x{h}, {args.steps} steps ...", flush=True)
        call_kwargs = dict(
            prompt=prompt,
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.guidance),
            height=h,
            width=w,
            generator=gen,
        )
        if ref_images is not None:
            call_kwargs["image"] = ref_images
        image = pipe(**call_kwargs).images[0]

        # Atomic write so the editor's folder watcher never sees a partial PNG.
        tmp = out + ".tmp.png"
        image.save(tmp)
        os.replace(tmp, out)
        print("OK", out, flush=True)
    except SystemExit:
        raise
    except Exception:
        _fail(out, "".join(traceback.format_exc()))


if __name__ == "__main__":
    main()
