#!/usr/bin/env python3
"""
audiogen_generate.py - single-shot text-to-audio generator (music + SFX).

Invoked per-clip by the editor (Content Browser -> right-click -> Generate
Audio...).  Uses Stability AI's Stable Audio Open 1.0 through diffusers: one
local model that handles both music and sound effects, up to 47 seconds of
44.1 kHz stereo, ~5 GB VRAM in fp16 so it can share the GPU with the engine.

On failure it writes "<out>.err" with the traceback so the C++ side can detect
and report the error without parsing stdout (same contract as tools/flux).

Usage:
    python audiogen_generate.py --prompt-file p.txt --out clip.wav \
        --type music --duration 12 --steps 100 --seed -1

Notes:
  * Weights are gated: accept the licence at
    https://huggingface.co/stabilityai/stable-audio-open-1.0 and log in
    (huggingface-cli login or set HF_TOKEN) before first use - the same
    routine as the FLUX image generator.
  * Dependencies: torch (CUDA), diffusers, transformers, soundfile.
    Run setup_audiogen.py once (or Setup.bat -audiogen if wired).
"""
import argparse
import os
import sys
import traceback

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None,
                    help="read the prompt from this UTF-8 file (avoids shell "
                         "escaping)")
    ap.add_argument("--out", required=True, help="output WAV path")
    ap.add_argument("--type", choices=["music", "sfx"], default="music",
                    help="tunes defaults (negative prompt / steps); the same "
                         "model serves both")
    ap.add_argument("--duration", type=float, default=12.0,
                    help="clip length in seconds (max 47)")
    ap.add_argument("--steps", type=int, default=0,
                    help="diffusion steps; 0 = per-type default")
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random")
    ap.add_argument("--model", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--quant", choices=["none", "fp8"], default="fp8",
                    help="DiT weight quantization (optimum-quanto qfloat8, "
                         "same scheme as tools/flux). fp8 halves transformer "
                         "VRAM with near-inaudible quality cost; none = fp16.")
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

    duration = max(1.0, min(47.0, float(args.duration)))
    steps = int(args.steps) if args.steps > 0 else \
        (100 if args.type == "music" else 70)
    # Generic quality guards; SFX additionally pushes away musicality so a
    # "creaky wooden door" doesn't come back with a melody underneath it.
    negative = "low quality, distortion, clipping"
    if args.type == "sfx":
        negative += ", music, melody, singing"

    try:
        import torch
        if not torch.cuda.is_available():
            _fail(out, "CUDA GPU not available - Stable Audio Open needs an "
                       "NVIDIA GPU.")

        import soundfile as sf
        from diffusers import StableAudioPipeline

        print(f"[audiogen] loading {args.model} ...", flush=True)
        pipe = StableAudioPipeline.from_pretrained(
            args.model, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        # fp8 (quanto): quantize the on-GPU fp16 DiT weights in place —
        # float8 keeps exponent+mantissa so it stays clean where int8
        # wouldn't.  Mirrors the FLUX transformer treatment.  The small T5
        # text encoder and the VAE stay fp16 (they're cheap, and the VAE is
        # the most quality-sensitive stage).
        if args.quant == "fp8":
            from optimum.quanto import quantize, freeze, qfloat8
            print("[audiogen] quantizing DiT to fp8 (quanto)...", flush=True)
            quantize(pipe.transformer, weights=qfloat8)
            freeze(pipe.transformer)

        gen = None
        if args.seed >= 0:
            gen = torch.Generator(device="cuda").manual_seed(int(args.seed))

        print(f"[audiogen] generating {duration:.1f}s {args.type}, "
              f"{steps} steps ...", flush=True)
        audio = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            audio_end_in_s=duration,
            num_waveforms_per_prompt=1,
            generator=gen,
        ).audios[0]

        # [channels, samples] -> [samples, channels] float32 for soundfile.
        data = audio.T.float().cpu().numpy()
        sr = int(pipe.vae.sampling_rate)

        # Atomic write so the editor's poll never sees a partial WAV.
        tmp = out + ".tmp.wav"
        sf.write(tmp, data, sr, subtype="PCM_16")
        os.replace(tmp, out)
        print("OK", out, flush=True)
    except SystemExit:
        raise
    except Exception:
        _fail(out, "".join(traceback.format_exc()))


if __name__ == "__main__":
    main()
