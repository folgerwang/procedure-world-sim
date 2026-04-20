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
rem   -skip-setup        Skip Setup.bat (Python deps, ML model, LibTorch download)
rem   Pass-through to Setup.bat:  -skip-deps  -skip-model  -skip-libtorch
rem   Everything else (-game, -engine, -vscode, -project=...) is ignored safely
set "SDK_OVERRIDE="
set "LIBTORCH_DIR="
set "SKIP_SETUP=0"

:arg_loop
if "%~1"=="" goto :arg_done
set "_a=%~1"
if /i "!_a:~0,8!"=="-vulkan=" set "SDK_OVERRIDE=!_a:~8!"
if /i "!_a:~0,10!"=="-libtorch=" set "LIBTORCH_DIR=!_a:~10!"
if /i "!_a!"=="-skip-setup" set "SKIP_SETUP=1"
shift
goto :arg_loop
:arg_done

rem ── Run Setup.bat (environment + dependencies) ───────────────────────────────
rem   Setup handles Python deps, ML model export, and LibTorch download.
rem   Detection of where things ended up (for CMake) is done below.
if "!SKIP_SETUP!"=="1" (
    echo [INFO]  Skipping Setup.bat ^(-skip-setup^)
) else (
    if not exist "Setup.bat" (
        echo [WARN]  Setup.bat not found -- skipping environment setup.
    ) else (
        echo [INFO]  Running Setup.bat ^(use -skip-setup to bypass^)
        call Setup.bat %*
        if errorlevel 1 (
            echo [WARN]  Setup.bat reported errors -- continuing anyway.
        )
    )
)

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

rem ── Resolve LibTorch path (detection only -- Setup.bat did the download) ─────
rem   Priority: 1) -libtorch= arg  2) third_parties\libtorch  3) common locations
set "CMAKE_LIBTORCH="
set "LIBTORCH_LOCAL=realworld\src\sim_engine\third_parties\libtorch"

rem 1) Explicit argument
if not "!LIBTORCH_DIR!"=="" (
    echo [INFO]  LibTorch ^(argument^): !LIBTORCH_DIR!
    goto :libtorch_set
)

rem 2) third_parties\libtorch (where Setup.bat places downloads)
if exist "!LIBTORCH_LOCAL!\share\cmake\Torch\TorchConfig.cmake" (
    set "LIBTORCH_DIR=!LIBTORCH_LOCAL!"
    echo [INFO]  LibTorch ^(third_parties^): !LIBTORCH_DIR!
    goto :libtorch_set
)

rem 3) Common system locations
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

echo [WARN]  LibTorch not found. Auto-rig will fall back to heuristic mode.
echo         Run Setup.bat (or Setup.bat with network access) to download it,
echo         or pass -libtorch="C:\path\to\libtorch".
goto :libtorch_done

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
