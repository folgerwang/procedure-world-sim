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
rem   Unknown args are ignored (allows pass-through from GenerateProjectFiles.bat)
set "LIBTORCH_DIR_ARG="
set "SKIP_DEPS=0"
set "SKIP_MODEL=0"
set "SKIP_LIBTORCH=0"

:arg_loop
if "%~1"=="" goto :arg_done
set "_a=%~1"
if /i "!_a:~0,10!"=="-libtorch=" set "LIBTORCH_DIR_ARG=!_a:~10!"
if /i "!_a!"=="-skip-deps" set "SKIP_DEPS=1"
if /i "!_a!"=="-skip-model" set "SKIP_MODEL=1"
if /i "!_a!"=="-skip-libtorch" set "SKIP_LIBTORCH=1"
shift
goto :arg_loop
:arg_done

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

echo.
echo ============================================================
echo  Setup complete.  Next: GenerateProjectFiles.bat
echo ============================================================
echo.

endlocal
