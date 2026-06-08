# Text-to-voice voices

In-game dialog speech runs **sherpa-onnx** on the CPU with a Piper-style VITS
voice.  Each voice is a folder under this directory containing a `.onnx`
model + `tokens.txt` (and usually `espeak-ng-data/`).  Voices are NOT
committed to git (Setup.bat downloads them).

> Location note: voices live under `assets/ml_models/tts/` (moved from the
> old `assets/models/tts/`).

## Which voices

By default Setup.bat downloads the **English Piper** voices only, the
**smallest (lowest-RAM, int8) model per voice**.  Examples:

- **Female:** `vits-piper-en_US-amy-low-int8`
- **Male:** `vits-piper-en_US-ryan-low-int8`
- **Male (alt):** `vits-piper-en_US-joe-medium-int8`

The in-engine picker (**Audio → Voice**) lists only **VITS / Piper** voices —
the reliable family (espeak G2P, every language).  Other families in the
release (Kokoro, Kitten, Matcha) and CJK voices have per-model quirks that
make sherpa-onnx abort on load/generate, so they're hidden.  To experiment:

```
set RW_TTS_EXPERIMENTAL=1
RealWorld.exe
```

## Picking / changing the voice at runtime

When more than one voice is installed, the engine picks one **at random each
launch**.  Controls:

```
set RW_TTS_VOICE=ryan     :: force a specific voice (substring, case-insensitive)
set RW_TTS_RANDOM=0       :: disable random; first folder alphabetically
RealWorld.exe
```

You can also switch live in the editor (**Audio → Voice**) — clicking a voice
auditions it (re-reads the last dialog line) without closing the menu.

## Bulk download

`tools/tts/download_voices.py` fetches voices from the sherpa-onnx
**tts-models** release into this folder (skips ones already present,
resumable, retries transient failures):

```
python realworld/tools/tts/download_voices.py --list                      # browse
python realworld/tools/tts/download_voices.py --filter vits-piper-en_      # English Piper (default)
python realworld/tools/tts/download_voices.py --all                        # every language
python realworld/tools/tts/download_voices.py --filter amy --filter ryan
```

By default only the **smallest model per voice** is kept (lowest quality +
most-quantized precision) to save RAM and disk.  Pass `--all-qualities` to
keep every variant.  Set `GITHUB_TOKEN` first if you hit the API rate limit.

Via Setup.bat: `-tts-voice=all:vits-piper-en_` (English, default),
`-tts-voice=all` (every language), or an explicit comma list.

## Custom / trained voices

Train a Piper voice (or convert an existing Piper checkpoint) to the
sherpa-onnx layout (`model.onnx` + `tokens.txt`), drop the folder here, and
point `RW_TTS_VOICE` at it — no C++ changes.
