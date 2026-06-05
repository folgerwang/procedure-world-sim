# FLUX.2 (4-bit) text-to-image for the Content Browser

This folder hosts the **4-bit / NF4** FLUX.2 image generator the editor's
Content Browser uses (right-click a folder → *Generate image*).

| File | Role |
|------|------|
| `setup_flux.py` | One-shot installer: builds `.venv`, installs CUDA torch + diffusers stack, handles the gated Hugging Face login, downloads `diffusers/FLUX.2-dev-bnb-4bit`. |
| `flux_generate.py` | Per-image worker the editor launches. Reads a prompt, renders one PNG, writes `<out>.err` on failure. |
| `.venv/` | Created by `setup_flux.py`. The editor runs `.venv/python flux_generate.py`. |

Why 4-bit: the NF4 build quantizes the DiT + text encoders to 4-bit (VAE stays
full precision), so generation fits in **~18–20 GB VRAM** instead of the ~32 GB
the FP8 build needs. Add `--offload` to go lower (slower).

## Install

FLUX.2-dev is gated under the FLUX dev non-commercial licence. **Accept it and
log in first**, then run setup:

```
# 1. accept licence (both pages, once):
#    https://huggingface.co/black-forest-labs/FLUX.2-dev
#    https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit
# 2. authenticate:
huggingface-cli login            # or: set HF_TOKEN=hf_xxx
# 3. install:
python realworld\tools\flux\setup_flux.py            # add --cuda cu121 etc. to match your GPU
```

Re-running is safe/idempotent. `--selftest` renders a tiny image to verify.

## Manual generation (sanity check)

```
realworld\tools\flux\.venv\Scripts\python.exe ^
    realworld\tools\flux\flux_generate.py ^
    --prompt "a red sports car at dusk" --out out.png --width 1024 --height 1024
```

## Setup.bat block

The installer is meant to be invoked from `Setup.bat`. If it isn't already
wired (it's gated behind the existing `-skip-flux` flag / `SKIP_FLUX`), add this
after the Ollama section:

```bat
rem -- FLUX.2 (4-bit) image generator --------------------------------
if "%SKIP_FLUX%"=="1" (
    echo [INFO]  -skip-flux given: skipping FLUX.2 generator setup.
    goto :flux_done
)
echo.
echo -- FLUX.2 image generator ---------------------------------
echo [INFO]  Installing deps + downloading FLUX.2-dev (4-bit NF4) weights (gated).
echo [INFO]  Accept the licence and log in FIRST, otherwise the download fails:
echo            https://huggingface.co/black-forest-labs/FLUX.2-dev
echo            huggingface-cli login        (or set the HF_TOKEN env var)
python "%~dp0realworld\tools\flux\setup_flux.py" --cuda %CUDA_WHEEL%
if errorlevel 1 (
    echo [WARN]  FLUX.2 setup failed -- see messages above.  Re-run after
    echo         accepting the licence / logging in:
    echo            python realworld\tools\flux\setup_flux.py
)
echo -----------------------------------------------------------
:flux_done
```

## Editor glue contract (`Menu::launchImageGen`)

`menu.h` declares `launchImageGen(folder, prompt, width, height)` with this
behaviour: write the prompt to a temp file, spawn the worker **detached**, then
poll for the output PNG (success) or `<out>.err` (failure). The matching command:

```
<repo>\realworld\tools\flux\.venv\Scripts\python.exe
<repo>\realworld\tools\flux\flux_generate.py
    --prompt-file <tmp_prompt.txt>
    --out        <folder>\flux_<timestamp>.png
    --width <w> --height <h>
```

`flux_generate.py` writes the PNG atomically (temp + rename), so the poll never
sees a half-written file. On any error it writes the traceback to `<out>.err`
with a hint (OOM → suggest `--offload`; gated → suggest login; old diffusers →
suggest re-running setup).

> Note: `launchImageGen` is currently only *declared* in `menu.h` — the C++
> implementation still needs to be written to actually spawn this command and
> poll. Ask and it can be added.
