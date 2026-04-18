@echo off
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo  RealWorld -- Visual Studio Solution Generator
echo ============================================================
echo.

rem ── Parse arguments ──────────────────────────────────────────────────────────
rem   -vulkan=<path>     Override Vulkan SDK location
rem   -libtorch=<path>   LibTorch installation directory
rem   -skip-model        Skip ML model export step
rem   Everything else (-game, -engine, -vscode, -project=...) is ignored safely
set "SDK_OVERRIDE="
set "LIBTORCH_DIR="
set "SKIP_MODEL=0"

:arg_loop
if "%~1"=="" goto :arg_done
set "_a=%~1"
if /i "!_a:~0,8!"=="-vulkan=" set "SDK_OVERRIDE=!_a:~8!"
if /i "!_a:~0,10!"=="-libtorch=" set "LIBTORCH_DIR=!_a:~10!"
if /i "!_a!"=="-skip-model" set "SKIP_MODEL=1"
shift
goto :arg_loop
:arg_done

rem ── Verify CMake is on PATH ───────────────────────────────────────────────────
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found. Install CMake 3.20+ and add it to PATH.
    echo         https://cmake.org/download/
    pause
    exit /b 1
)

for /f "tokens=3" %%v in ('cmake --version 2^>nul ^| findstr /i "cmake version"') do (
    echo [INFO]  CMake %%v
)

rem ── Detect Visual Studio 2022 via vswhere ─────────────────────────────────────
set "VS_GENERATOR=Visual Studio 17 2022"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_PATH="

if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (
        `"%VSWHERE%" -latest -version "[17.0,18.0)" -property installationPath 2^>nul`
    ) do set "VS_PATH=%%i"
)

if "!VS_PATH!"=="" (
    echo [WARN]  VS 2022 not detected via vswhere -- CMake will search automatically.
) else (
    echo [INFO]  VS 2022: !VS_PATH!
)

rem ── Resolve Vulkan SDK ────────────────────────────────────────────────────────
rem   Priority: 1) -vulkan= arg   2) VULKAN_SDK env var   3) LunarG registry key
set "CMAKE_VULKAN="

if not "!SDK_OVERRIDE!"=="" (
    echo [INFO]  Vulkan SDK ^(argument^): !SDK_OVERRIDE!
    set "CMAKE_VULKAN=-DCMAKE_PREFIX_PATH=!SDK_OVERRIDE!"
    goto :vulkan_done
)

if defined VULKAN_SDK (
    echo [INFO]  Vulkan SDK ^(env^): %VULKAN_SDK%
    set "CMAKE_VULKAN=-DCMAKE_PREFIX_PATH=%VULKAN_SDK%"
    goto :vulkan_done
)

rem Try LunarG registry key
for /f "usebackq skip=2 tokens=3*" %%a in (
    `reg query "HKLM\SOFTWARE\LunarG\VulkanSDK" /ve 2^>nul`
) do (
    if not "%%a"=="" set "CMAKE_VULKAN=-DCMAKE_PREFIX_PATH=%%a %%b"
)

if not "!CMAKE_VULKAN!"=="" (
    echo [INFO]  Vulkan SDK ^(registry^): found
    goto :vulkan_done
)

echo [WARN]  Vulkan SDK not found. CMake will try to locate it automatically.
echo         If generation fails, install from: https://vulkan.lunarg.com
echo         Or pass: GenerateProjectFiles.bat -vulkan="C:\VulkanSDK\1.3.x"

:vulkan_done

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
echo ── ML Model Export ──────────────────────────────────────────
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
echo ─────────────────────────────────────────────────────────────
echo.

:model_done

rem ── LibTorch auto-sync ───────────────────────────────────────────────────────
rem   Priority: 1) -libtorch= arg   2) third_parties/libtorch   3) auto-download
set "CMAKE_LIBTORCH="
set "LIBTORCH_LOCAL=realworld\src\sim_engine\third_parties\libtorch"
set "LIBTORCH_VERSION=2.7.0"
set "LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-!LIBTORCH_VERSION!%%2Bcpu.zip"

rem 1) Explicit argument
if not "!LIBTORCH_DIR!"=="" (
    echo [INFO]  LibTorch ^(argument^): !LIBTORCH_DIR!
    goto :libtorch_set
)

rem 2) Already downloaded to third_parties
if exist "!LIBTORCH_LOCAL!\share\cmake\Torch\TorchConfig.cmake" (
    set "LIBTORCH_DIR=!LIBTORCH_LOCAL!"
    echo [INFO]  LibTorch ^(third_parties^): !LIBTORCH_DIR!
    goto :libtorch_set
)

rem 3) Check common system locations
for %%d in (
    "C:\LibTorch"
    "%USERPROFILE%\libtorch"
    "%LOCALAPPDATA%\libtorch"
) do (
    if exist "%%~d\share\cmake\Torch\TorchConfig.cmake" (
        set "LIBTORCH_DIR=%%~d"
        echo [INFO]  LibTorch ^(auto-detected^): !LIBTORCH_DIR!
        goto :libtorch_set
    )
)

rem 4) Auto-download to third_parties
echo.
echo [INFO]  LibTorch not found — downloading v!LIBTORCH_VERSION! ^(CPU^) ...
echo [INFO]  URL: !LIBTORCH_URL!
echo [INFO]  This is a one-time ~200 MB download.
echo.

set "DL_ZIP=%TEMP%\libtorch-download.zip"

rem Try curl first (ships with Windows 10 1803+)
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
    set "LIBTORCH_DIR=!LIBTORCH_LOCAL!"
    echo [INFO]  LibTorch installed: !LIBTORCH_DIR!
    goto :libtorch_set
) else (
    echo [WARN]  Extraction failed — TorchConfig.cmake not found.
    echo         Auto-rig will use heuristic mode.
    goto :libtorch_done
)

:libtorch_set
rem Build CMAKE_PREFIX_PATH combining Vulkan + LibTorch
if "!CMAKE_VULKAN!"=="" (
    set "CMAKE_LIBTORCH=-DCMAKE_PREFIX_PATH=!LIBTORCH_DIR!"
) else (
    rem Extract the Vulkan path from -DCMAKE_PREFIX_PATH=<path> and combine
    set "_vk_path=!CMAKE_VULKAN:-DCMAKE_PREFIX_PATH=!"
    set "CMAKE_LIBTORCH=-DCMAKE_PREFIX_PATH=!_vk_path!;!LIBTORCH_DIR!"
    set "CMAKE_VULKAN="
)

:libtorch_done

rem ── Run CMake ─────────────────────────────────────────────────────────────────
set "BUILD_DIR=build_vs"

if exist "%BUILD_DIR%\RealWorld.sln" (
    echo [INFO]  Regenerating existing solution in %BUILD_DIR%\
) else (
    echo [INFO]  Output directory: %BUILD_DIR%\
)

rem Final prefix path
set "CMAKE_ALL_PREFIX="
if not "!CMAKE_LIBTORCH!"=="" (
    set "CMAKE_ALL_PREFIX=!CMAKE_LIBTORCH!"
) else if not "!CMAKE_VULKAN!"=="" (
    set "CMAKE_ALL_PREFIX=!CMAKE_VULKAN!"
)

echo.
echo [INFO]  cmake -G "%VS_GENERATOR%" -A x64 -B %BUILD_DIR% !CMAKE_ALL_PREFIX!
echo.

cmake -G "%VS_GENERATOR%" -A x64 -B "%BUILD_DIR%" !CMAKE_ALL_PREFIX!

if errorlevel 1 (
    echo.
    echo [ERROR] CMake generation failed. See output above for details.
    echo.
    echo  Common fixes:
    echo    - Install Vulkan SDK:  https://vulkan.lunarg.com
    echo    - Specify SDK path:    GenerateProjectFiles.bat -vulkan="C:\VulkanSDK\1.3.x"
    echo    - Install LibTorch:    https://pytorch.org/get-started/locally/
    echo    - Specify LibTorch:    GenerateProjectFiles.bat -libtorch="C:\LibTorch"
    echo    - Init submodules:     git submodule update --init --recursive
    echo.
    pause
    exit /b 1
)

rem ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  Done!  Solution: %BUILD_DIR%\RealWorld.sln
echo ============================================================
echo.
echo  Build Debug:    cmake --build %BUILD_DIR% --config Debug
echo  Build Release:  cmake --build %BUILD_DIR% --config Release
echo.

set /p "_open=Open in Visual Studio now? [Y/n]: "
if /i "!_open!"=="n"  goto :eof
if /i "!_open!"=="no" goto :eof

start "" "%BUILD_DIR%\RealWorld.sln"

endlocal
