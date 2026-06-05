#!/usr/bin/env python3
"""
test_klein_gpu.py - minimal FLUX.2-klein-4B test, everything on the GPU.

klein-4B is small (4B transformer + a small Qwen3 text encoder, ~16GB on disk),
so it fits comfortably on a 24GB card with no CPU offload and no quantization.

IMPORTANT: klein-4B uses a Qwen3 text encoder, which older diffusers releases do
not support (they assume dev's Mistral encoder and crash in the chat template).
Install the latest diffusers from source first:
    pip install -U "git+https://github.com/huggingface/diffusers.git"

Usage:
    python test_klein_gpu.py --prompt "a cozy fantasy tavern, warm light" --out klein.png
"""
import argparse
import sys
import time

import torch


def gb(x):
    return x / (1024.0 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a cozy fantasy tavern, warm candlelight")
    ap.add_argument("--out", default="klein_test.png")
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-offload", action="store_true",
                    help="stream components CPU<->GPU if VRAM is ever tight")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[klein] no CUDA GPU available.", file=sys.stderr)
        sys.exit(1)

    print(f"[klein] torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
    try:
        from diffusers import Flux2Pipeline
    except Exception as e:
        print(f"[klein] cannot import Flux2Pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[klein] loading {args.model} (bf16)...")
    t = time.perf_counter()
    pipe = Flux2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    print(f"[klein] ready in {time.perf_counter() - t:.1f}s")

    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"[klein] generating {args.width}x{args.height}, {args.steps} steps...")
    t = time.perf_counter()
    with torch.inference_mode():
        img = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=gen,
        ).images[0]
    dt = time.perf_counter() - t

    img.save(args.out)
    print(f"\n[klein] saved {args.out} in {dt:.1f}s")
    print(f"[klein] peak VRAM: {gb(torch.cuda.max_memory_allocated()):.2f} GiB "
          f"allocated / {gb(torch.cuda.max_memory_reserved()):.2f} GiB reserved")


if __name__ == "__main__":
    main()
