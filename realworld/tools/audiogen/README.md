# audiogen — local text-to-audio generation (music + SFX)

Generates music and sound-effect clips from text prompts, locally on the GPU,
for the editor's Content Browser (right-click → **Generate Audio...**).

- Model: [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
  via `diffusers` — one model for both music and SFX, up to 47 s of
  44.1 kHz stereo, ~5 GB VRAM in fp16.
- Output: `.wav` written into the Content Browser folder you invoked it from;
  on failure a `<name>.wav.err` sidecar carries the traceback (same contract
  as `tools/flux`).

## One-time setup

1. Accept the model licence:
   <https://huggingface.co/stabilityai/stable-audio-open-1.0>
2. Log in: `huggingface-cli login` (or set `HF_TOKEN`).
3. Install deps + prefetch weights:

   ```
   python tools/audiogen/setup_audiogen.py
   ```

   (or build the `audiogen_setup` CMake target / project in Visual Studio)

## CLI

```
python tools/audiogen/audiogen_generate.py \
    --prompt-file p.txt --out clip.wav --type music --duration 12
```

| flag           | default | notes                                    |
|----------------|---------|------------------------------------------|
| `--type`       | music   | `music` or `sfx` (tunes steps/negatives) |
| `--duration`   | 12      | seconds, max 47                          |
| `--steps`      | 0       | 0 = per-type default (100 music, 70 sfx) |
| `--seed`       | -1      | -1 = random                              |
| `--quant`      | fp8     | quanto qfloat8 on the DiT (same as flux);
                             `none` = plain fp16                       |
