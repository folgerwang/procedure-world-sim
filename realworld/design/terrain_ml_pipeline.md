# Terrain Generation — ML Redesign

## Why

The current pipeline generates each modality separately and hopes they agree:

```
prompt ──FLUX──► albedo ──MiDaS/heuristic──► heightmap
                   │
                   └──colour heuristics──► masks ──► PCG proxies
```

Every consistency bug we hit came from the same root: **the masks are
reverse-engineered from pixels with hand-tuned rules**, so buildings/trees
never quite match the painted albedo (rivers classified as roofs, brown
scrub as towns, antique-style layouts quantizing to 0% town, etc.). The
layout-first experiment showed the second failure mode: FLUX's reference
conditioning is *soft*, so even a correct layout drifts against the albedo
that's rendered from it.

Design rule going forward: **exactly one modality is generated freely; every
other modality is *derived* from it by a model that looks at it directly.**

---

## Phase 1 — Learned perception on the albedo (days, no training)

Keep colour-first generation (FLUX orthophoto stays the source of truth —
it's the thing the player actually sees). Replace `terrain_pcg.segment()`
heuristics with off-the-shelf perception models. Placement then matches the
albedo *by definition*, because it is detected from it.

### 1a. Land-cover segmentation
- Model: **SegFormer-B2 / U-Net fine-tuned on OpenEarthMap** (0.25–0.5 m GSD,
  8 classes: bareland, rangeland, developed, road, tree, water, agriculture,
  building — almost exactly our class set), or LoveDA-trained equivalents.
  Several checkpoints are on HuggingFace; ONNX-export and run through the
  already-shipped `onnxruntime`.
- Inference: tile 8192² into 1024² tiles with 128 px overlap, feathered
  blending, fp16. Seconds on the 5090.
- Interface: unchanged — returns the same `{water, roof, road, veg}` mask
  dict `segment()` returns today (the code was designed for this swap).

### 1b. Building instances
- The land-cover "building" class + the existing PCA/oriented-box +
  merged-block subdivision code (already written and tuned) turns the mask
  into per-house boxes.
- Optional upgrade: **SAM2** prompted with seed points inside the building
  mask → per-instance masks → min-area rotated rect per instance. Gives
  clean per-house separation in dense quarters without the 16 m grid
  approximation.

### 1c. Tree instances
- **DeepForest** (RGB tree-crown detector, RetinaNet, pip-installable) on
  the albedo: individual crown boxes → cone position + radius from box size.
  Trees then stand exactly on the painted canopy instead of a jittered grid.
- Fallback: keep the jittered grid inside the tree-class mask.

### 1d. Height from albedo
- Replace MiDaS-small (indoor/outdoor photos; wrong domain) with an
  **aerial monocular elevation model**: fine-tune Depth-Anything-V2-Small on
  aligned NAIP orthophoto ↔ USGS 3DEP DEM tiles (public, effectively
  unlimited pairs; a weekend of 5090 time), or use published IM2ELEVATION
  weights. Then keep the existing erosion/refine stack, and drive
  `semantic_height_pass`/water carving from the *learned* masks
  (roads/towns flat, water below banks — already implemented, just feed it
  better masks).

**Deliverable:** same pipeline shape, three model calls replace ~150 lines
of colour rules. Buildings/trees/water match the albedo the player sees.

---

## Phase 2 — Consistency by construction (weeks, light training)

Fixes the remaining gap: FLUX freestyles layout *within* town/forest
regions, so even perfect perception only recovers what FLUX chose to paint.
Make generation itself controllable:

### 2a. Layout LoRA for FLUX
- Fine-tune a LoRA on FLUX-klein: caption → **land-cover raster in the
  exact palette** (flat fills, no antique-map style drift).
- Training data is free: rasterize **OSM** (buildings/roads/water) over
  **ESA WorldCover / OpenEarthMap** land cover into palette PNGs; captions
  from templated descriptions of the tile ("four towns along a river
  crossing, mountains north", derivable from the raster itself).
  A few thousand 1024² tiles is enough for style lock-in.

### 2b. Seg→albedo with hard conditioning
- FLUX ref-image conditioning is soft — replace with either:
  - an **img2img init**: colorize the layout with per-class base colours +
    noise, then denoise at ~0.5–0.6 strength. Cheap, no training, keeps
    macro layout pixel-aligned (this alone fixes most drift), or
  - a proper **ControlNet trained for seg→aerial** on the same OSM/OEM
    pairs (strong alignment, more training work).
- Because the albedo is now pixel-aligned to the layout, PCG can place from
  the layout masks directly and still match the texture. The Phase-1
  perception models stay as a verification pass (they should agree; log a
  mismatch score).

### 2c. Height conditioned on layout
- Keep the class-driven synthesis (base elevation + class-scaled relief +
  erosion) already implemented for layout mode — it's aligned by
  construction and erosion makes it read naturally. The Phase-1 elevation
  model can add realism as a residual on top of the class base.

---

## Phase 3 — One-pass multi-modal generation (research)

Single diffusion model emits **aligned channels in one sampling pass**:
RGB + height + semantics (5–6 channels). Marigold showed a pretrained
image diffusion model fine-tunes into a dense-prediction generator with
tiny data; the same recipe applies to (albedo, DEM, land-cover) triplets
from NAIP + 3DEP + OSM. All modalities agree because they're jointly
denoised. This subsumes Phases 1–2; the pipeline keeps the same file
contract (`<map>.png`, `_color.png`, `_seg.png`, `_pcg.glb`) so the engine
never notices.

---

## Runtime integration notes

- All models ONNX where possible (`onnxruntime` ships with the engine);
  torch fallback is fine for tooling since the pipeline runs out-of-process.
- The detail-tile worker can reuse the Phase-1 segmentation to refine the
  1 m tiles (crisp water edges, road flattening at detail scale) and to
  spawn *streamed* micro-PCG (rocks, bushes) per tile from the same masks.
- Keep `<map>_seg.png` (indexed palette) as the canonical mask artifact:
  gameplay systems (navigation, spawning, walkability, collision classes)
  should read it rather than re-deriving anything from pixels.
- Determinism: seed everything from the map's numeric id (already the
  convention for `_det_rand`).

## Suggested order of work

1. SegFormer/OpenEarthMap ONNX segmentation behind `segment()`  ← biggest
   payoff per effort; directly fixes "placement doesn't match albedo".
2. DeepForest tree crowns (drop-in, replaces grid trees).
3. Depth-Anything-V2 aerial fine-tune for height (replaces MiDaS).
4. img2img hard-init for the layout→albedo pass (kills drift cheaply).
5. Layout LoRA; then evaluate whether Phase 3 is worth the training run.
