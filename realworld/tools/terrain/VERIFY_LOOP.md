# Terrain generation verify loop

A closed loop that judges whether generated terrain "makes sense" and, when it
doesn't, revises the prompt/parameters and regenerates.

```
generate heightmap ──► engine applies it, waits, dumps a clean viewport frame, quits
                   ──► the frame is scored against a rubric
                   ──► pass → done
                       fail → adjust prompt / height-scale → regenerate
```

The dump is the engine's own render-buffer readback written to a PNG — no OS
screenshots involved. Analysis can be done two ways: by the `verify_loop.py`
harness via the Claude API (fully automated), or by reading the dumped PNG
directly (e.g. Claude in Cowork looks at the file).

## One-shot verify-dump mode (engine CLI)

The engine has a headless mode that does exactly one capture and exits:

```
RealWorld.exe --verify-apply <heightmap.png> [--verify-color <albedo.png>] \
              --verify-dump <out.png> --verify-delay 10
```

Flags:

| Flag | Meaning |
|------|---------|
| `--verify-dump[=<path>]` | Enable the mode. After a terrain is applied, wait, dump a clean frame to `<path>` (default `screenshots/terrain_verify_latest.png`), then quit. |
| `--verify-apply <png>` | Apply this heightmap on startup — no manual "Generate" click needed (fully scripted). |
| `--verify-color <png>` | Optional albedo to pair with `--verify-apply`. |
| `--verify-delay <sec>` | Seconds to wait after apply before the dump (default 10). |

Flow: apply the heightmap via `createTerrainFromMaps` → wait `--verify-delay`
seconds for streaming → render ONE frame with the editor UI hidden
(`Menu::draw(hide_ui=true)`) → read that swapchain image back to `<out.png>`
(also refreshes `screenshots/frame_latest.png`) → write
`screenshots/terrain_verify_done.txt` → `glfwSetWindowShouldClose`.

Without `--verify-apply`, the timer arms instead on the next manual/AI terrain
apply — so you can launch with just `--verify-dump`, click **Generate**, and the
app will dump-and-close ~10 s after the terrain lands.

`terrain_verify_done.txt` lets a driver know the capture is ready:

```
capture=<out.png>
width=<w>
height=<h>
```

## Automated harness (`verify_loop.py`)

Runs the whole loop: generate → launch the engine one-shot → analyze the dump
via the Claude API → adjust on failure → repeat.

Prerequisites: a built `RealWorld.exe`, `ANTHROPIC_API_KEY` in the environment,
run from `realworld/`. Optional `Pillow` (the harness downscales the frame
before upload if present).

```bash
# from realworld/
set ANTHROPIC_API_KEY=sk-ant-...

python tools/terrain/verify_loop.py --prompt "volcanic island with a caldera lake"
python tools/terrain/verify_loop.py --prompt "desert canyon" --max-iters 5 --color
```

Analyze an existing frame once (no gen, no engine) — handy for tuning the rubric:

```bash
python tools/terrain/verify_loop.py --analyze-only screenshots/frame_latest.png \
       --prompt "snowy alpine range"
```

Key flags: `--max-iters` (default 4), `--height-scale`, `--color`,
`--model` (default `claude-sonnet-5`), `--delay`, `--engine-exe`
(default `RealWorld.exe`), `--seed`, `--out-dir`, `--report`. Extra generator
flags go after `--gen-args`. Exit code is `0` if the terrain passed, `2`
otherwise; a JSON report of every iteration is written under `--out-dir`.

## Manual / Cowork analysis

If you want a human or Claude-in-Cowork to judge instead of the API:

1. Generate a heightmap: `python tools/terrain/terrain_from_text.py --prompt "..." --out content/terrain/x.png`
2. `RealWorld.exe --verify-apply content/terrain/x.png --verify-dump screenshots/x.png`
3. Open `screenshots/x.png` — it's a clean, UI-free render of the terrain — and
   judge it against the rubric below. Adjust the prompt and repeat.

## Rubric

Each criterion is pass/fail; terrain passes only if all four pass:

- **not_degenerate** — real terrain, not flat/blank/black, a failed load, NaN
  speckle, or corruption.
- **plausible_landforms** — natural elevation; no impossible spikes, hard
  stair-stepping, or repeating stamped tiles.
- **matches_prompt** — reflects the requested biome / dominant landform /
  notable features.
- **no_artifacts** — no seams, tiling, holes, z-fighting, or texture corruption.

## Notes / limits

- The capture reflects the current editor camera. Terrain-apply already snaps the
  camera toward the new heightmap; for a fixed overview per run, add a camera
  snap before the capture.
- `--verify-delay` is a fixed wall-clock wait, not a "streaming finished" signal.
  Increase it for very large heightmaps if captures look mid-stream.
- A file-based live channel (`content/terrain/.verify/request.txt` →
  `serviceTerrainVerify`) also exists for driving apply+capture without
  relaunching; the one-shot CLI mode above is the simpler default.
```
