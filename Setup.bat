@echo off
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo  RealWorld -- Environment Setup
echo ============================================================
echo.
echo  This script prepares the build environment:
echo    * Python dependencies (ml_training\requirements.txt)
echo    * Auto-rig ML model export
echo    * LibTorch download (if not already present)
echo    * Ollama install + qwen3.5:2b pull (Mesh Category classifier)
echo    * FLUX.2-klein-4B image generator install + weight download (Content Browser)
echo    * Stable Audio Open text-to-audio generator install + weights (Generate Audio)
echo    * Text-to-voice voices download -- English Piper, lowest-RAM (-skip-tts to skip)
echo    * Qwen3.5-2B Jin Yong chatbot fine-tune (default ON, idempotent; -skip-jinyong)
echo.
echo  Run GenerateProjectFiles.bat afterwards to produce the VS
echo  solution. GenerateProjectFiles.bat also invokes this script
echo  automatically unless you pass -skip-setup.
echo.

rem ── Parse arguments ──────────────────────────────────────────────────────────
rem   -libtorch=<path>    Use existing LibTorch at this path (skip download)
rem   -skip-deps          Skip Python dependency install
rem   -skip-model         Skip ML model export
rem   -skip-libtorch      Skip LibTorch download
rem   -skip-bc7enc        Skip bc7enc clone (BC7 albedo encode for VT)
rem   -skip-ollama        Skip Ollama install + model pull (Mesh Category)
rem   -ollama-model=<tag> Ollama model tag to pull (default: qwen3.5:2b).
rem                         Must match OLLAMA_MODEL at runtime; the
rem                         classifier reads that env var.
rem   -skip-cuda          Skip CUDA torch install (use CPU wheel from requirements.txt)
rem   -skip-flux          Skip the FLUX.2-klein-4B image generator install +
rem                         weight download (diffusers-from-source + bitsandbytes
rem                         + the gated FLUX.2-klein-4B weights, ~16GB). ON by
rem                         default; reuses the CUDA torch installed above. Runs
rem                         the Qwen3 encoder on CPU + an fp8 transformer on the
rem                         GPU (~5-6GB VRAM). Needs a Hugging Face login.
rem   -skip-audiogen      Skip the Stable Audio Open text-to-audio generator
rem                         install + weight prefetch (gated, ~5GB; powers the
rem                         Content Browser's "Generate Audio..."). ON by
rem                         default; reuses the same CUDA torch + HF login as
rem                         FLUX. fp8-quantized DiT at generation time.
rem   -skip-tts           Skip the text-to-voice VOICE MODEL download
rem                         (vits-piper-en_US-amy-medium, ~65MB, ungated) into
rem                         realworld\assets\ml_models\tts.  The sherpa-onnx SDK
rem                         itself is fetched by CMake at configure time.
rem   -tts-voice=<list>   sherpa-onnx tts-models voices to download.
rem                         DEFAULT: all:vits-piper-en_  (English Piper only,
rem                         smallest int8 model per voice).  Alternatives:
rem                           all           every voice/language (~several GB)
rem                           all:<substr>  filtered subset
rem                           <a>,<b>,...    explicit COMMA-SEPARATED list
rem                         Pick one at runtime with the RW_TTS_VOICE env var
rem                         (substring match); multiple installed = random
rem                         per launch.  Skip the download with -skip-tts.
rem   -cuda=<wheel>       CUDA wheel tag (default: cu128). Examples:
rem                         cu128  -- works with all current NVIDIA GPUs incl.
rem                                   Blackwell/RTX 50-series (sm_120). DEFAULT.
rem                         cu121  -- older; no sm_120 kernels.
rem                         cu118  -- older still, Turing/Ampere-friendly.
rem                         nightly/cu128  -- bleeding edge if stable lacks
rem                                           support for your card yet.
rem                         cpu    -- force CPU torch (same as -skip-cuda).
rem   Unknown args are ignored (allows pass-through from GenerateProjectFiles.bat)
set "LIBTORCH_DIR_ARG="
set "SKIP_DEPS=0"
set "SKIP_MODEL=0"
set "SKIP_LIBTORCH=0"
set "SKIP_BC7ENC=0"
set "SKIP_OLLAMA=0"
set "OLLAMA_MODEL_ARG="
set "SKIP_CUDA=0"
set "CUDA_WHEEL=cu128"
set "SKIP_FLUX=0"
set "SKIP_AUDIOGEN=0"
set "SKIP_TTS=0"
rem Default downloads the ENGLISH Piper voices only, smallest (lowest-RAM,
rem int8) model per voice.  Override: -tts-voice=all (every language),
rem -tts-voice=all:<substr> (filtered), an explicit comma list, or -skip-tts.
set "TTS_VOICE=all:vits-piper-en_"
rem Qwen3.5-2B Jin Yong chatbot fine-tune. ON by default; delegates to
rem jinyong_finetune\setup.bat (full fp16 fine-tune on realworld\assets\txt;
rem dataset -> ml_training\qwen3.5-2b-jinyong\data). It is IDEMPOTENT: once a
rem trained model exists it skips, and it auto-skips when no NVIDIA GPU is found.
rem Disable with -skip-jinyong.
set "SKIP_JINYONG=0"

:arg_loop
if "%~1"=="" goto :arg_done
set "_a=%~1"
if /i "!_a:~0,10!"=="-libtorch=" set "LIBTORCH_DIR_ARG=!_a:~10!"
if /i "!_a!"=="-skip-deps" set "SKIP_DEPS=1"
if /i "!_a!"=="-skip-model" set "SKIP_MODEL=1"
if /i "!_a!"=="-skip-libtorch" set "SKIP_LIBTORCH=1"
if /i "!_a!"=="-skip-bc7enc" set "SKIP_BC7ENC=1"
if /i "!_a!"=="-skip-ollama" set "SKIP_OLLAMA=1"
if /i "!_a:~0,14!"=="-ollama-model=" set "OLLAMA_MODEL_ARG=!_a:~14!"
if /i "!_a!"=="-skip-cuda" set "SKIP_CUDA=1"
if /i "!_a:~0,6!"=="-cuda=" set "CUDA_WHEEL=!_a:~6!"
if /i "!_a!"=="-skip-flux" set "SKIP_FLUX=1"
if /i "!_a!"=="-skip-audiogen" set "SKIP_AUDIOGEN=1"
if /i "!_a!"=="-skip-tts" set "SKIP_TTS=1"
if /i "!_a:~0,11!"=="-tts-voice=" set "TTS_VOICE=!_a:~11!"
if /i "!_a!"=="-skip-jinyong" set "SKIP_JINYONG=1"
if /i "!_a!"=="-jinyong" set "SKIP_JINYONG=0"
shift
goto :arg_loop
:arg_done

rem -cuda=cpu is just a friendlier spelling of -skip-cuda
if /i "!CUDA_WHEEL!"=="cpu" set "SKIP_CUDA=1"

rem ── Python dependencies for ML training ──────────────────────────────────────
rem   The Auto-Rig "Train Model" button and the model export step below both
rem   invoke Python scripts in ml_training\. pip is fast when everything is
rem   already satisfied. Use -skip-deps to bypass.
set "ML_REQS=ml_training\requirements.txt"

if "!SKIP_DEPS!"=="1" (
    echo [INFO]  Skipping Python dependency install ^(-skip-deps^)
    goto :deps_done
)

if not exist "%ML_REQS%" (
    echo [INFO]  %ML_REQS% not found -- skipping dependency install
    goto :deps_done
)

where python >nul 2>&1
if errorlevel 1 (
    echo [WARN]  Python not found on PATH -- skipping dependency install.
    echo         The Auto-Rig Train Model button will fail until you:
    echo           1^) install Python 3.8+
    echo           2^) run:  python -m pip install -r %ML_REQS%
    goto :deps_done
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
    echo [INFO]  Python %%v
)

rem ── CUDA torch ────────────────────────────────────────────────────────────────
rem   requirements.txt pins torch>=2.0 but doesn't specify an index, so pip
rem   picks the CPU wheel on Windows. If an NVIDIA GPU is present we install
rem   the CUDA wheel FIRST -- then the requirements.txt install below sees the
rem   torch requirement as already satisfied and won't downgrade it.
rem
rem   Default wheel: cu128 (works with RTX 50-series Blackwell cards that need
rem   sm_120 kernels, and backward compatible with older NVIDIA cards).
rem   Override with -cuda=<tag>; skip with -skip-cuda or -cuda=cpu.
if "!SKIP_CUDA!"=="1" (
    echo [INFO]  Skipping CUDA torch install ^(-skip-cuda / -cuda=cpu^)
    goto :cuda_torch_done
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO]  nvidia-smi not found -- skipping CUDA torch install
    echo         ^(training will run on CPU. Pass -skip-cuda to silence this.^)
    goto :cuda_torch_done
)

rem Is a CUDA-enabled torch already installed? If so, skip the slow reinstall.
python -c "import sys, torch; sys.exit(0 if torch.backends.cuda.is_built() else 1)" >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%t in ('python -c "import torch; print(torch.__version__)"') do (
        echo [INFO]  CUDA torch already installed: %%t -- skipping reinstall
    )
    goto :cuda_torch_done
)

set "CUDA_INDEX=https://download.pytorch.org/whl/!CUDA_WHEEL!"
echo.
echo -- CUDA torch install -------------------------------------
echo [INFO]  NVIDIA GPU detected -- installing CUDA torch wheel
echo [INFO]  Wheel tag:  !CUDA_WHEEL!
echo [INFO]  Index:      !CUDA_INDEX!
echo [INFO]  One-time ~2.5 GB download.
python -m pip uninstall -y torch torchvision >nul 2>&1
python -m pip install --disable-pip-version-check torch torchvision --index-url "!CUDA_INDEX!"
if errorlevel 1 (
    echo.
    echo [WARN]  CUDA torch install failed ^(index: !CUDA_INDEX!^).
    echo         For RTX 50-series ^(Blackwell^) try nightly:
    echo           Setup.bat -cuda=nightly/cu128
    echo         Or fall back to CPU:
    echo           Setup.bat -skip-cuda
    echo         Continuing -- requirements.txt will install a CPU wheel instead.
) else (
    echo [INFO]  CUDA torch installed. Verify with:
    echo           python ml_training\check_device.py
)
echo -----------------------------------------------------------
echo.

:cuda_torch_done

echo.
echo -- Python dependencies ------------------------------------
echo [INFO]  Installing from %ML_REQS% ^(use -skip-deps to bypass^)
python -m pip install --disable-pip-version-check -r "%ML_REQS%"
if errorlevel 1 (
    echo.
    echo [WARN]  pip install failed. The Auto-Rig Train Model button
    echo         and ML model export will not work until this is fixed.
    echo         Try manually:  python -m pip install -r %ML_REQS%
    echo         For CUDA torch, install it separately first from
    echo         https://pytorch.org/get-started/locally/
) else (
    echo [INFO]  Python dependencies OK
)
echo -----------------------------------------------------------
echo.

:deps_done

rem ── Auto-rig ML model ────────────────────────────────────────────────────────
rem   If the TorchScript model doesn't exist, run the Python export.
rem   Use -skip-model to bypass this step.
set "MODEL_PATH=realworld\assets\models\rig_diffusion.pt"

if "!SKIP_MODEL!"=="1" (
    echo [INFO]  Skipping ML model export ^(-skip-model^)
    goto :model_done
)

if exist "%MODEL_PATH%" (
    echo [INFO]  ML model found: %MODEL_PATH%
    goto :model_done
)

echo [INFO]  ML model not found at %MODEL_PATH%
echo [INFO]  Attempting to export auto-rig model...

where python >nul 2>&1
if errorlevel 1 (
    echo [WARN]  Python not found -- skipping ML model export.
    echo         The auto-rig plugin will run in heuristic mode.
    echo         To enable ML mode, install Python 3.8+ and re-run, or run:
    echo           python ml_training\scripts\download_and_export_model.py --method distill
    goto :model_done
)

rem Make sure output directory exists
if not exist "realworld\assets\models" mkdir "realworld\assets\models"

rem Try distillation first (downloads pretrained weights from torchvision).
rem If that fails (no internet / missing deps), fall back to skeleton export.
echo.
echo -- ML Model Export ----------------------------------------
python ml_training\scripts\download_and_export_model.py ^
    --method distill ^
    --output "%MODEL_PATH%" ^
    --resolution 256

if errorlevel 1 (
    echo [WARN]  Distillation failed ^(likely no internet or missing torch^).
    echo [INFO]  Falling back to skeleton export ^(random weights^)...
    python ml_training\scripts\download_and_export_model.py ^
        --method skeleton ^
        --output "%MODEL_PATH%" ^
        --resolution 256
    if errorlevel 1 (
        echo [WARN]  Skeleton export also failed. Skipping ML model.
        echo         Install PyTorch: pip install torch torchvision
    ) else (
        echo [INFO]  Skeleton model exported. For better results, run distillation:
        echo           python ml_training\scripts\download_and_export_model.py --method distill
    )
) else (
    echo [INFO]  Distilled model exported successfully.
)
echo -----------------------------------------------------------
echo.

:model_done

rem ── bc7enc (Rich Geldreich) ──────────────────────────────────────────────────
rem   Production-grade BC7 encoder used by the Runtime Virtual Texture's
rem   albedo layer. Replaces the in-tree minimal Mode-6 encoder that was
rem   producing visibly wrong colours on the GPU (the streamer + page table
rem   were verified independently with an RGBA8 pool; only the BC7 path
rem   was at fault). bc7enc is a small two-file repo (~3000 LoC, MIT
rem   licensed, no dependencies); we only need bc7enc.h + bc7enc.cpp.
rem   Use -skip-bc7enc to bypass (build will fail to compile if the files
rem   aren't already present from a previous run).
set "BC7ENC_DIR=realworld\src\sim_engine\third_parties\bc7enc"
set "BC7ENC_REPO=https://github.com/richgel999/bc7enc.git"

if "!SKIP_BC7ENC!"=="1" (
    echo [INFO]  Skipping bc7enc clone ^(-skip-bc7enc^)
    goto :bc7enc_done
)

rem Note: the upstream source file is bc7enc.c (C, not C++) — the header
rem uses `extern "C"` so we treat it as a C compilation unit and let the
rem CMake build pick it up alongside our C++ wrapper bc7_encoder.cpp.
if exist "!BC7ENC_DIR!\bc7enc.c" if exist "!BC7ENC_DIR!\bc7enc.h" (
    echo [INFO]  bc7enc already present: !BC7ENC_DIR!
    goto :bc7enc_done
)

if not exist "!BC7ENC_DIR!" mkdir "!BC7ENC_DIR!"

echo.
echo -- bc7enc clone -------------------------------------------
where git >nul 2>&1
if not errorlevel 1 (
    rem Shallow clone into a temp dir, then move only the two files we
    rem need into !BC7ENC_DIR!. We avoid `git clone` directly into
    rem !BC7ENC_DIR! so we don't drop a .git folder + ~3 MB of unrelated
    rem files (rgbcx, ert, test images) into our third_parties tree.
    set "BC7ENC_TMP=%TEMP%\bc7enc-clone-%RANDOM%"
    echo [INFO]  Cloning !BC7ENC_REPO!
    git clone --depth 1 --quiet "!BC7ENC_REPO!" "!BC7ENC_TMP!"
    if errorlevel 1 (
        echo [WARN]  git clone failed. Trying direct file download...
        goto :bc7enc_curl
    )
    if exist "!BC7ENC_TMP!\bc7enc.c" (
        copy /Y "!BC7ENC_TMP!\bc7enc.h" "!BC7ENC_DIR!\bc7enc.h" >nul
        copy /Y "!BC7ENC_TMP!\bc7enc.c" "!BC7ENC_DIR!\bc7enc.c" >nul
        rmdir /S /Q "!BC7ENC_TMP!" 2>nul
        echo [INFO]  bc7enc installed: !BC7ENC_DIR!
        goto :bc7enc_done
    )
    rmdir /S /Q "!BC7ENC_TMP!" 2>nul
    echo [WARN]  git clone produced no bc7enc.c. Trying direct download...
)

:bc7enc_curl
set "BC7ENC_RAW_H=https://raw.githubusercontent.com/richgel999/bc7enc/master/bc7enc.h"
set "BC7ENC_RAW_C=https://raw.githubusercontent.com/richgel999/bc7enc/master/bc7enc.c"
where curl >nul 2>&1
if not errorlevel 1 (
    echo [INFO]  Downloading bc7enc.h via curl...
    curl -L --silent --fail -o "!BC7ENC_DIR!\bc7enc.h" "!BC7ENC_RAW_H!"
    echo [INFO]  Downloading bc7enc.c via curl...
    curl -L --silent --fail -o "!BC7ENC_DIR!\bc7enc.c" "!BC7ENC_RAW_C!"
) else (
    echo [INFO]  curl not found, trying PowerShell...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!BC7ENC_RAW_H!' -OutFile '!BC7ENC_DIR!\bc7enc.h'"
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!BC7ENC_RAW_C!' -OutFile '!BC7ENC_DIR!\bc7enc.c'"
)

if exist "!BC7ENC_DIR!\bc7enc.c" if exist "!BC7ENC_DIR!\bc7enc.h" (
    echo [INFO]  bc7enc installed: !BC7ENC_DIR!
) else (
    echo [WARN]  bc7enc download failed. The VT albedo BC7 path will fail
    echo         to compile. Either re-run Setup.bat with internet, or
    echo         manually copy bc7enc.h / bc7enc.c into:
    echo           !BC7ENC_DIR!
    echo         from !BC7ENC_REPO!
)
echo -----------------------------------------------------------
echo.

:bc7enc_done

rem ── LibTorch download ────────────────────────────────────────────────────────
rem   Download LibTorch to third_parties\libtorch if it's not already there.
rem   Path resolution (which LibTorch to actually use for the build) is the
rem   responsibility of GenerateProjectFiles.bat.
set "LIBTORCH_LOCAL=realworld\src\sim_engine\third_parties\libtorch"
set "LIBTORCH_VERSION=2.7.0"
set "LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-!LIBTORCH_VERSION!%%2Bcpu.zip"

if "!SKIP_LIBTORCH!"=="1" (
    echo [INFO]  Skipping LibTorch download ^(-skip-libtorch^)
    goto :libtorch_done
)

rem If user points at their own LibTorch, we don't need to download.
if not "!LIBTORCH_DIR_ARG!"=="" (
    echo [INFO]  LibTorch ^(argument^): !LIBTORCH_DIR_ARG!
    echo [INFO]  Skipping download -- using user-provided LibTorch
    goto :libtorch_done
)

rem Already downloaded to third_parties?
if exist "!LIBTORCH_LOCAL!\share\cmake\Torch\TorchConfig.cmake" (
    echo [INFO]  LibTorch already present: !LIBTORCH_LOCAL!
    goto :libtorch_done
)

rem Common system locations?
for %%d in (
    "C:\LibTorch"
    "%USERPROFILE%\libtorch"
    "%LOCALAPPDATA%\libtorch"
) do (
    if exist "%%~d\share\cmake\Torch\TorchConfig.cmake" (
        echo [INFO]  LibTorch detected at %%~d -- skipping download
        goto :libtorch_done
    )
)

rem Auto-download
echo.
echo [INFO]  LibTorch not found -- downloading v!LIBTORCH_VERSION! ^(CPU^) ...
echo [INFO]  URL: !LIBTORCH_URL!
echo [INFO]  This is a one-time ~200 MB download.
echo.

set "DL_ZIP=%TEMP%\libtorch-download.zip"

where curl >nul 2>&1
if not errorlevel 1 (
    echo [INFO]  Downloading with curl...
    curl -L --progress-bar -o "!DL_ZIP!" "!LIBTORCH_URL!"
) else (
    echo [INFO]  curl not found, trying PowerShell...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!LIBTORCH_URL!' -OutFile '!DL_ZIP!'"
)

if not exist "!DL_ZIP!" (
    echo [WARN]  Download failed. Auto-rig will use heuristic mode.
    echo         Download manually from: https://pytorch.org/get-started/locally/
    echo         Extract to: !LIBTORCH_LOCAL!
    goto :libtorch_done
)

echo [INFO]  Extracting to !LIBTORCH_LOCAL! ...
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "!DL_ZIP!" "realworld\src\sim_engine\third_parties"

del "!DL_ZIP!" 2>nul

if exist "!LIBTORCH_LOCAL!\share\cmake\Torch\TorchConfig.cmake" (
    echo [INFO]  LibTorch installed: !LIBTORCH_LOCAL!
) else (
    echo [WARN]  Extraction failed -- TorchConfig.cmake not found.
    echo         Auto-rig will use heuristic mode.
)

:libtorch_done

rem ── Ollama (local LLM for the Mesh Category classifier) ──────────────────────
rem   The runtime classifier at helper/material_classifier.cpp talks to a local
rem   Ollama daemon to tag each material / object node with one of the
rem   MeshCategory enum values (Floor / Wall / Door / …).  Without Ollama
rem   running the call returns Unknown for everything, the Render Debug →
rem   Mesh Category overlay paints the whole scene grey, and the per-category
rem   physics behaviour falls back to the substring heuristic.
rem
rem   This section:
rem     1) Detects an existing Ollama install (PATH or
rem        %LOCALAPPDATA%\Programs\Ollama\ollama.exe).
rem     2) If absent, downloads OllamaSetup.exe from ollama.com and runs
rem        it silently (per-user install, no admin required).
rem     3) Starts the daemon if it isn't already serving (uses `ollama list`
rem        as a liveness probe; on failure spawns the tray app which in turn
rem        launches `ollama serve`).
rem     4) `ollama pull <model>` so the chosen model is ready locally.
rem
rem   Override the default model with -ollama-model=qwen2.5:7b (or any other
rem   tag).  Must match the OLLAMA_MODEL env var the engine reads at runtime
rem   — set it before launching RealWorld if you're not using the default.
rem
set "OLLAMA_DIR=%LOCALAPPDATA%\Programs\Ollama"
set "OLLAMA_EXE=!OLLAMA_DIR!\ollama.exe"
set "OLLAMA_APP=!OLLAMA_DIR!\ollama app.exe"
set "OLLAMA_INSTALLER_URL=https://ollama.com/download/OllamaSetup.exe"
set "OLLAMA_MODEL=qwen3.5:2b"
if not "!OLLAMA_MODEL_ARG!"=="" set "OLLAMA_MODEL=!OLLAMA_MODEL_ARG!"

if "!SKIP_OLLAMA!"=="1" (
    echo [INFO]  Skipping Ollama setup ^(-skip-ollama^) -- the Mesh
    echo         Category classifier will fall back to substring matching
    echo         until you install Ollama and run:  ollama pull !OLLAMA_MODEL!
    goto :ollama_done
)

rem Locate ollama.exe -- PATH first, then default install dir.
set "OLLAMA_BIN="
where ollama >nul 2>&1
if not errorlevel 1 set "OLLAMA_BIN=ollama"
if "!OLLAMA_BIN!"=="" if exist "!OLLAMA_EXE!" set "OLLAMA_BIN=!OLLAMA_EXE!"

if not "!OLLAMA_BIN!"=="" goto :ollama_have_bin

echo.
echo -- Ollama install -----------------------------------------
echo [INFO]  Ollama not found -- downloading installer
echo [INFO]  URL: !OLLAMA_INSTALLER_URL!
echo [INFO]  One-time ~600 MB download (installer + base runtime).

set "OLLAMA_DL=%TEMP%\OllamaSetup.exe"
where curl >nul 2>&1
if not errorlevel 1 (
    echo [INFO]  Downloading with curl...
    curl -L --progress-bar -o "!OLLAMA_DL!" "!OLLAMA_INSTALLER_URL!"
) else (
    echo [INFO]  curl not found, trying PowerShell...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!OLLAMA_INSTALLER_URL!' -OutFile '!OLLAMA_DL!'"
)

if not exist "!OLLAMA_DL!" (
    echo [WARN]  Ollama installer download failed.  Install manually
    echo         from !OLLAMA_INSTALLER_URL! and re-run Setup.bat, or
    echo         skip with -skip-ollama.
    goto :ollama_done
)

rem /SILENT shows a small progress dialog; /VERYSILENT hides it
rem entirely.  /SILENT is the gentler choice -- the user sees that
rem an install is happening if it's slow.  The Inno Setup installer
rem auto-creates the per-user install at %LOCALAPPDATA%\Programs\Ollama
rem and registers the tray-app autostart entry.
echo [INFO]  Running installer silently...
"!OLLAMA_DL!" /SILENT /CLOSEAPPLICATIONS
del "!OLLAMA_DL!" 2>nul

if not exist "!OLLAMA_EXE!" (
    echo [WARN]  Installation did not land at !OLLAMA_EXE!.
    echo         Run the installer manually from !OLLAMA_INSTALLER_URL!.
    goto :ollama_done
)
set "OLLAMA_BIN=!OLLAMA_EXE!"
echo [INFO]  Ollama installed: !OLLAMA_DIR!
echo -----------------------------------------------------------
echo.

:ollama_have_bin

rem Daemon liveness probe -- `ollama list` queries the local server at
rem 127.0.0.1:11434 and exits non-zero when it can't reach it.  If
rem the probe fails, spawn the tray app (which launches `ollama serve`
rem in the background) and wait a few seconds for the port to bind.
"!OLLAMA_BIN!" list >nul 2>&1
if errorlevel 1 (
    if exist "!OLLAMA_APP!" (
        echo [INFO]  Starting Ollama daemon ^(tray app^)...
        start "" "!OLLAMA_APP!"
        rem ~3 seconds for the server to bind 127.0.0.1:11434.
        ping -n 4 127.0.0.1 >nul
    ) else (
        echo [WARN]  Could not reach the Ollama daemon and no tray app
        echo         binary was found at !OLLAMA_APP!.  Start the daemon
        echo         manually with:  ollama serve
    )
)

echo.
echo -- Pulling Ollama model -----------------------------------
echo [INFO]  Model:    !OLLAMA_MODEL!
echo [INFO]  Override: -ollama-model=^<tag^>   ^(see ollama.com/library^)
echo [INFO]  Pull is idempotent -- a present model is a fast no-op.
"!OLLAMA_BIN!" pull "!OLLAMA_MODEL!"
if errorlevel 1 (
    echo [WARN]  ollama pull failed.  The Mesh Category classifier will
    echo         fall back to substring matching until the model is
    echo         available.  Try manually:
    echo            "!OLLAMA_BIN!" pull !OLLAMA_MODEL!
) else (
    echo [INFO]  Model ready: !OLLAMA_MODEL!
    echo [INFO]  RealWorld reads OLLAMA_MODEL at runtime; if you change
    echo         the tag here, also set the env var:
    echo            set OLLAMA_MODEL=!OLLAMA_MODEL!
)
echo -----------------------------------------------------------
echo.

:ollama_done

rem ── Hugging Face access (gated weights: FLUX.2 + Stable Audio Open) ──────────
rem   Both generators download GATED weights, which needs (a) the licence
rem   accepted on each model page while logged into your HF account, and
rem   (b) a read token stored locally.  This section detects an existing
rem   login; otherwise it opens the model pages + token page in the browser
rem   and prompts for the token.  Leaving the prompt empty skips (the
rem   generator setups below will then fail with instructions).
if "!SKIP_FLUX!"=="1" if "!SKIP_AUDIOGEN!"=="1" goto :hf_done

where python >nul 2>&1
if errorlevel 1 goto :hf_done

rem huggingface_hub may not be installed yet (it normally rides in with
rem diffusers, which the generator setups below install) -- make sure the
rem login/whoami calls in this section can import it.
python -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo [INFO]  Installing huggingface_hub for the login step...
    python -m pip install --disable-pip-version-check -q huggingface_hub
)

rem Existing login (env token or cached ~/.cache/huggingface/token)?
python -c "from huggingface_hub import whoami; whoami()" >nul 2>&1
if not errorlevel 1 (
    echo [INFO]  Hugging Face: already logged in -- skipping token prompt.
    echo [INFO]  If a weight download still reports 403/GatedRepoError, open
    echo         the model page and click "Agree" with THIS account:
    echo            https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
    echo            https://huggingface.co/stabilityai/stable-audio-open-1.0
    goto :hf_done
)

echo.
echo -- Hugging Face access ------------------------------------
echo [INFO]  The image / audio generators use GATED model weights.
echo [INFO]  Opening the model pages -- click "Agree and access repository"
echo         on BOTH while logged into your Hugging Face account:
start "" "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
start "" "https://huggingface.co/stabilityai/stable-audio-open-1.0"
echo [INFO]  Then create a READ token (page also opened) and paste it below:
start "" "https://huggingface.co/settings/tokens"
echo.
set "HF_TOKEN_IN="
set /p "HF_TOKEN_IN=  Paste HF token (hf_...), or press Enter to skip: "
if "!HF_TOKEN_IN!"=="" (
    echo [WARN]  Skipped Hugging Face login -- gated weight downloads will
    echo         fail until you run:  huggingface-cli login
    goto :hf_done
)
python -c "from huggingface_hub import login; login(token=r'!HF_TOKEN_IN!')" 2>nul
if errorlevel 1 (
    echo [WARN]  Login failed -- check the token and run:  huggingface-cli login
) else (
    echo [INFO]  Hugging Face login stored.
)
echo -----------------------------------------------------------
echo.

:hf_done

rem ── FLUX.2-klein-4B text-to-image generator ──────────────────────────────────
rem   Installs diffusers-from-source + bitsandbytes (reusing the CUDA torch
rem   installed above) and downloads the gated FLUX.2-klein-4B weights (~16GB),
rem   so the editor's Content Browser can generate images (right-click a folder
rem   -> Generate).  Runs the Qwen3 text encoder on CPU and an fp8 diffusion
rem   transformer on the GPU (~5-6GB VRAM, sharing the card with the engine).
rem   ON by default -- needs an NVIDIA GPU and a Hugging Face login.
rem   Skip with:  Setup.bat -skip-flux
if "!SKIP_FLUX!"=="1" (
    echo [INFO]  Skipping FLUX.2 image generator ^(-skip-flux^) -- the Content
    echo         Browser's right-click Generate will not work until you run:
    echo         python realworld\tools\flux\setup_flux.py
    goto :flux_done
)

where python >nul 2>&1
if errorlevel 1 (
    echo [WARN]  Python not found -- cannot set up the FLUX.2 generator.
    goto :flux_done
)

echo.
echo -- FLUX.2-klein-4B image generator ------------------------
echo [INFO]  Installing deps + downloading FLUX.2-klein-4B weights ^(gated, ~16GB^).
echo [INFO]  Accept the licence and log in FIRST, otherwise the download fails:
echo            https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
echo            huggingface-cli login        ^(or set the HF_TOKEN env var^)
python realworld\tools\flux\setup_flux.py
if errorlevel 1 (
    echo [WARN]  FLUX.2 setup failed -- see messages above.  Re-run after
    echo         accepting the licence / logging in:  python realworld\tools\flux\setup_flux.py
) else (
    echo [INFO]  FLUX.2 generator ready.
)
echo -----------------------------------------------------------
echo.

:flux_done

rem ── Stable Audio Open text-to-audio generator ────────────────────────────────
rem   Installs diffusers/soundfile/optimum-quanto (reusing the CUDA torch from
rem   above) and prefetches the gated Stable Audio Open 1.0 weights (~5GB), so
rem   the editor's Content Browser can generate music / sound effects
rem   (right-click a folder -> Generate Audio...).  The DiT is fp8-quantized
rem   (quanto) at generation time and shares the GPU with the engine.
rem   ON by default -- needs an NVIDIA GPU and the SAME Hugging Face login as
rem   FLUX (accept the model licence first).
rem   Skip with:  Setup.bat -skip-audiogen
if "!SKIP_AUDIOGEN!"=="1" (
    echo [INFO]  Skipping audio generator ^(-skip-audiogen^) -- the Content
    echo         Browser's "Generate Audio..." will not work until you run:
    echo         python realworld\tools\audiogen\setup_audiogen.py
    goto :audiogen_done
)

where python >nul 2>&1
if errorlevel 1 (
    echo [WARN]  Python not found -- cannot set up the audio generator.
    goto :audiogen_done
)

echo.
echo -- Stable Audio Open text-to-audio generator --------------
echo [INFO]  Installing deps + prefetching Stable Audio Open 1.0 weights ^(gated, ~5GB^).
echo [INFO]  Accept the licence and log in FIRST, otherwise the download fails:
echo            https://huggingface.co/stabilityai/stable-audio-open-1.0
echo            huggingface-cli login        ^(or set the HF_TOKEN env var^)
python realworld\tools\audiogen\setup_audiogen.py
if errorlevel 1 (
    echo [WARN]  Audio generator setup failed -- see messages above.  Re-run
    echo         after accepting the licence / logging in:
    echo            python realworld\tools\audiogen\setup_audiogen.py
) else (
    echo [INFO]  Audio generator ready.
)
echo -----------------------------------------------------------
echo.

:audiogen_done

rem ── Text-to-voice VOICE MODEL(s) (sherpa-onnx Piper voice) ───────────────────
rem   The in-game text-to-voice (dialog lines) runs sherpa-onnx on the CPU
rem   with a Piper-style VITS voice.  The SDK (DLLs + import lib) is fetched
rem   by CMake at configure time; THIS step downloads the ungated ~65MB voice
rem   model(s) into realworld\assets\ml_models\tts\.
rem
rem   -tts-voice accepts a COMMA-SEPARATED list, so you can install several
rem   and switch at runtime with the RW_TTS_VOICE env var (substring match on
rem   the folder name).  Examples (any asset from the sherpa-onnx "tts-models"
rem   release tag):
rem     female (default):  vits-piper-en_US-amy-medium
rem     male:              vits-piper-en_US-ryan-medium
rem     male (alt):        vits-piper-en_US-joe-medium
rem     both at once:      -tts-voice=vits-piper-en_US-amy-medium,vits-piper-en_US-ryan-medium
set "TTS_DIR=realworld\assets\ml_models\tts"

if "!SKIP_TTS!"=="1" (
    echo [INFO]  Skipping TTS voice model ^(-skip-tts^) -- in-game text-to-voice
    echo         stays disabled until a voice is placed in !TTS_DIR!
    goto :tts_done
)

if not exist "!TTS_DIR!" mkdir "!TTS_DIR!"
echo.
echo -- Text-to-voice voice model^(s^) ---------------------------

rem Special value: -tts-voice=all downloads EVERY voice in the release
rem (~130+, several GB) via the Python enumerator.  Also accepts
rem -tts-voice=all:<substr> to bulk-download a filtered subset, e.g.
rem   -tts-voice=all:vits-piper-en   (all English Piper voices)
if /i "!TTS_VOICE:~0,3!"=="all" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [WARN]  Python not found -- cannot bulk-download voices.
        goto :tts_done
    )
    set "TTS_FILTER=!TTS_VOICE:~4!"
    if "!TTS_VOICE!"=="all" set "TTS_FILTER="
    if "!TTS_FILTER!"=="" (
        echo [INFO]  Downloading ALL voices ^(~130+, several GB^)...
        python realworld\tools\tts\download_voices.py --all
    ) else (
        echo [INFO]  Downloading voices matching '!TTS_FILTER!'...
        python realworld\tools\tts\download_voices.py --filter "!TTS_FILTER!"
    )
    echo [INFO]  Default voice = first folder alphabetically; override at
    echo         runtime with RW_TTS_VOICE ^(e.g. set RW_TTS_VOICE=ryan^).
    goto :tts_done
)

rem Loop over the comma/space-separated voice list.
for %%V in (!TTS_VOICE!) do (
    if exist "!TTS_DIR!\%%V" (
        echo [INFO]  Voice already present: %%V
    ) else (
        echo [INFO]  Downloading %%V ^(~65MB, ungated^)
        set "TTS_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/%%V.tar.bz2"
        set "TTS_TAR=%TEMP%\%%V.tar.bz2"
        where curl >nul 2>&1
        if not errorlevel 1 (
            curl -L --progress-bar -o "!TTS_TAR!" "!TTS_URL!"
        ) else (
            powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!TTS_URL!' -OutFile '!TTS_TAR!'"
        )
        if exist "!TTS_TAR!" (
            rem Windows 10+ ships bsdtar as tar.exe; auto-detects bz2.
            tar -xf "!TTS_TAR!" -C "!TTS_DIR!"
            del "!TTS_TAR!" 2>nul
            if exist "!TTS_DIR!\%%V" (
                echo [INFO]  Voice installed: %%V
            ) else (
                echo [WARN]  Extraction failed for %%V
            )
        ) else (
            echo [WARN]  Download failed for %%V -- skipping.
        )
    )
)
echo [INFO]  Default voice = first folder alphabetically; override at runtime
echo         with the RW_TTS_VOICE env var ^(e.g. set RW_TTS_VOICE=ryan^).

if exist "!TTS_DIR!\!TTS_VOICE!" (
    echo [INFO]  TTS voice installed: !TTS_DIR!\!TTS_VOICE!
) else (
    echo [WARN]  Extraction failed -- text-to-voice stays disabled.
)
echo -----------------------------------------------------------
echo.

:tts_done

<<<<<<< Updated upstream
rem ── Qwen3.5-2B Jin Yong chatbot fine-tune (default ON, idempotent) ───────────
rem   Delegates to the standalone jinyong_finetune\setup.bat so both entry points
rem   share one path. That script installs deps, prepares the dataset into
rem   ml_training\qwen3.5-2b-jinyong\data, then runs the full fp16 fine-tune --
rem   but SKIPS training if a trained model already exists, and skips on no-GPU.
rem   Disable entirely with -skip-jinyong.
if "!SKIP_JINYONG!"=="1" (
    echo [INFO]  Skipping Qwen3.5-2B Jin Yong fine-tune ^(-skip-jinyong^)
    goto :jinyong_done
)
if not exist "jinyong_finetune\setup.bat" (
    echo [WARN]  jinyong_finetune\setup.bat not found -- skipping Jin Yong fine-tune.
    goto :jinyong_done
)
echo.
echo -- Qwen3.5-2B Jin Yong fine-tune --------------------------
call "jinyong_finetune\setup.bat"
if errorlevel 1 echo [WARN]  Jin Yong fine-tune step reported an error.
:jinyong_done
=======
rem -----------------------------------------------------------
rem  Animation model fine-tune (19-joint Qwen -> Ollama).
rem  One-time + idempotent: the sub-script skips if 'anim-qwen-19joint'
rem  already exists or a prerequisite (Ollama/Python/CUDA) is missing, and
rem  always exits 0 so Setup never breaks.  Runtime animation generation uses
rem  this model, falling back to the stock model when it isn't installed.
rem  Skip explicitly with:  set RW_SKIP_ANIM_FT=1
rem -----------------------------------------------------------
echo.
echo ============================================================
echo  Animation model fine-tune (one-time, idempotent)
echo ============================================================
if exist "%~dp0ml_training\anim_finetune\build_anim_model.bat" (
    call "%~dp0ml_training\anim_finetune\build_anim_model.bat"
) else (
    echo [anim-ft] build_anim_model.bat not found -- skipping.
)
echo -----------------------------------------------------------
>>>>>>> Stashed changes

echo.
echo ============================================================
echo  Setup complete.  Next: GenerateProjectFiles.bat
echo ============================================================
echo.

endlocal
