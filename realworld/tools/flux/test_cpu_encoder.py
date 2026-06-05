#!/usr/bin/env python3
"""
test_cpu_encoder.py — proof-of-concept: run the (large) text encoder on the CPU
while the diffusion transformer + VAE run on the GPU.

Why: FLUX.2's text encoder is a ~24B Mistral model (~48GB bf16) that won't fit
alongside everything else on a 24GB card.  But the encoder only runs ONCE per
image to turn the prompt into embeddings, so we can keep it permanently in
system RAM (CPU) and let only the small 4B transformer + VAE use VRAM.

How it works without depending on FLUX.2's exact encode_prompt() API:
  * We physically place text_encoder on CPU, transformer + VAE on CUDA.
  * We wrap text_encoder in a thin "device bridge" so the pipeline can hand it
    CUDA tensors as usual — the bridge moves inputs to CPU, runs the encoder,
    and moves the resulting embeddings back to CUDA.
  * We force the pipeline's _execution_device to CUDA so the denoise loop runs
    on the GPU regardless of where individual modules live.

Requirements: ~48GB system RAM for the bf16 encoder (less if you later quantize
it for CPU), an NVIDIA GPU, and an accepted/authenticated FLUX.2 licence.

Usage:
    python test_cpu_encoder.py --prompt "a cozy fantasy tavern, warm light" \
        --out test.png --steps 28 --width 1024 --height 1024
"""
import argparse
import time
import sys

import torch


# ── recursive device move (tensors inside tuples / lists / ModelOutput / dict) ──
def _move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move(x, device) for x in obj)
    if hasattr(obj, "keys"):                       # dict / transformers ModelOutput
        for k in list(obj.keys()):
            obj[k] = _move(obj[k], device)
        return obj
    return obj


def pin_encoder_to_cpu(pipe, out_device):
    """Keep pipe.text_encoder physically on CPU but make it transparently accept
    GPU inputs and return GPU outputs, by patching its forward() in place.

    We patch forward (rather than wrapping the module in a new class) so the
    object stays a real text-encoder instance — newer diffusers selects the
    encode path with isinstance(), which a wrapper class would break.

    Returns a stats dict {seconds, calls} for the time spent encoding.
    """
    enc = pipe.text_encoder
    orig_forward = enc.forward
    stats = {"seconds": 0.0, "calls": 0}

    def patched(*args, **kwargs):
        to_cpu = lambda x: x.to("cpu") if torch.is_tensor(x) else x
        args = tuple(to_cpu(a) for a in args)
        kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
        t0 = time.perf_counter()
        out = orig_forward(*args, **kwargs)
        stats["seconds"] += time.perf_counter() - t0
        stats["calls"] += 1
        return _move(out, out_device)

    enc.forward = patched     # instance-level override; nn.Module.__call__ uses it
    return stats


def force_cuda_execution_device(pipe):
    """Make pipe._execution_device always report CUDA (denoise runs on GPU)."""
    base = type(pipe)
    pipe.__class__ = type(
        base.__name__ + "_CudaExec", (base,),
        {"_execution_device": property(lambda self: torch.device("cuda"))},
    )


def gb(x):  # bytes -> GiB
    return x / (1024.0 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a cozy fantasy tavern interior, warm "
                                         "candlelight, highly detailed")
    ap.add_argument("--out", default="test_cpu_encoder.png")
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B",
                    help="HF repo id (FLUX.2-klein-4B / -9B)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder-quant", choices=["none", "int8", "int4"],
                    default="int8",
                    help="CPU quantization for the text encoder. int8 ~24GB RAM "
                         "(good for 48GB machines), int4 ~12GB, none = bf16 ~48GB.")
    ap.add_argument("--transformer-quant", choices=["none", "int8", "4bit", "fp8"],
                    default="none",
                    help="GPU quantization for the diffusion transformer. "
                         "fp8 (quanto, ~half VRAM, float8 so usually clean), "
                         "int8/4bit (bitsandbytes; int8 tends to produce noise).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[test] no CUDA GPU available.", file=sys.stderr)
        sys.exit(1)

    print(f"[test] torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
    import diffusers
    from diffusers import DiffusionPipeline
    print(f"[test] diffusers {diffusers.__version__}")

    # ── Load via the AUTO loader so the right class is chosen ────────────────
    # klein declares _class_name="Flux2KleinPipeline" (Qwen3 encoder); dev uses
    # Flux2Pipeline (Mistral encoder). DiffusionPipeline reads model_index.json
    # and instantiates the correct one. Hardcoding Flux2Pipeline forces the
    # Mistral chat-template path and crashes on klein. Needs diffusers >=0.37.dev.
    print(f"[test] loading {args.model} (bf16)…")
    t_load = time.perf_counter()

    # Optionally pre-load the diffusion transformer quantized straight onto the
    # GPU. int8/4bit use bitsandbytes (load-time); fp8 uses quanto (post-load,
    # handled after the device split below).
    pre_transformer = None
    if args.transformer_quant in ("int8", "4bit"):
        try:
            from diffusers import BitsAndBytesConfig
            try:
                from diffusers import Flux2Transformer2DModel
            except ImportError:
                from diffusers.models import Flux2Transformer2DModel
        except Exception as e:
            print(f"[test] transformer quant needs diffusers+bitsandbytes: {e}\n"
                  f"       pip install -U bitsandbytes", file=sys.stderr)
            sys.exit(1)
        if args.transformer_quant == "int8":
            bnb = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16)
        print(f"[test] loading transformer in {args.transformer_quant} "
              f"(bitsandbytes, GPU)…")
        pre_transformer = Flux2Transformer2DModel.from_pretrained(
            args.model, subfolder="transformer",
            quantization_config=bnb, torch_dtype=torch.bfloat16)

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if pre_transformer is not None:
        load_kwargs["transformer"] = pre_transformer
    try:
        pipe = DiffusionPipeline.from_pretrained(args.model, **load_kwargs)
    except Exception as e:
        print(f"[test] load failed: {e}\n"
              f"       klein needs diffusers >=0.37.0.dev0 (with Flux2KleinPipeline):\n"
              f"         python -m pip uninstall diffusers -y\n"
              f"         python -m pip install -U "
              f"\"git+https://github.com/huggingface/diffusers\"", file=sys.stderr)
        sys.exit(1)
    print(f"[test] loaded in {time.perf_counter() - t_load:.1f}s "
          f"(pipeline: {type(pipe).__name__})")

    # ── Device split: encoder→CPU, transformer+VAE→GPU ──────────────────────
    pipe.vae = pipe.vae.to("cuda")
    pipe.text_encoder = pipe.text_encoder.to("cpu")
    if pre_transformer is None:
        # bf16 transformer: move it to the GPU. (A bitsandbytes-quantized one is
        # already on the GPU and must NOT be .to()'d.)
        pipe.transformer = pipe.transformer.to("cuda")

    # fp8 path: quantize the (now on-GPU, bf16) transformer in place with quanto.
    # float8 keeps an exponent+mantissa, so unlike int8 it usually stays clean.
    if args.transformer_quant == "fp8":
        try:
            from optimum.quanto import quantize, freeze, qfloat8
        except Exception:
            print("[test] fp8 needs optimum-quanto — run:\n"
                  "         pip install optimum-quanto", file=sys.stderr)
            sys.exit(1)
        print("[test] quantizing transformer to fp8 (quanto, GPU)…")
        t_fp8 = time.perf_counter()
        quantize(pipe.transformer, weights=qfloat8)
        freeze(pipe.transformer)
        print(f"[test] transformer fp8 in {time.perf_counter() - t_fp8:.1f}s")

    # ── Quantize the big text encoder on CPU (int8 ≈ 24GB, int4 ≈ 12GB) ──────
    # optimum-quanto runs on CPU (bitsandbytes 8/4-bit is CUDA-only, so it can't
    # be used here). The first load may briefly peak/swap while bf16 + quantized
    # copies coexist; afterwards the resident encoder is small.
    if args.encoder_quant != "none":
        try:
            from optimum.quanto import quantize, freeze, qint8, qint4
        except Exception:
            print("[test] optimum-quanto not installed — run:\n"
                  "         pip install optimum-quanto", file=sys.stderr)
            sys.exit(1)
        w = qint8 if args.encoder_quant == "int8" else qint4
        print(f"[test] quantizing text encoder to {args.encoder_quant} on CPU "
              f"(one-time; may swap briefly)…")
        t_q = time.perf_counter()
        quantize(pipe.text_encoder, weights=w)
        freeze(pipe.text_encoder)
        print(f"[test] encoder quantized in {time.perf_counter() - t_q:.1f}s")

    enc_stats = pin_encoder_to_cpu(pipe, torch.device("cuda"))
    force_cuda_execution_device(pipe)

    print(f"[test] text_encoder on CPU ({args.encoder_quant}), "
          f"transformer ({args.transformer_quant}) + VAE on CUDA")

    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    # ── Generate ────────────────────────────────────────────────────────────
    print(f"[test] generating {args.width}x{args.height}, {args.steps} steps…")
    t_gen = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=gen,
        ).images[0]
    total = time.perf_counter() - t_gen

    image.save(args.out)

    # ── Report ──────────────────────────────────────────────────────────────
    enc_s = enc_stats["seconds"]
    print("\n-------- results --------")
    print(f"  saved:              {args.out}")
    print(f"  total generate:     {total:6.1f} s")
    print(f"  CPU text-encode:    {enc_s:6.1f} s "
          f"({enc_stats['calls']} call(s))")
    print(f"  GPU denoise+vae:    {total - enc_s:6.1f} s")
    print(f"  peak VRAM (torch):  {gb(torch.cuda.max_memory_allocated()):6.2f} GiB "
          f"allocated / {gb(torch.cuda.max_memory_reserved()):.2f} GiB reserved")
    print("-------------------------")
    print("If peak VRAM is only a few GiB, the split worked: the text encoder "
          "never touched the GPU.")


if __name__ == "__main__":
    main()
