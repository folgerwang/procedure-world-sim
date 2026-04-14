@echo off
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo  RealWorld -- Visual Studio Solution Generator
echo ============================================================
echo.

rem ── Parse arguments ──────────────────────────────────────────────────────────
rem   -vulkan=<path>   Override Vulkan SDK location
rem   Everything else (-game, -engine, -vscode, -project=...) is ignored safely
set "SDK_OVERRIDE="

:arg_loop
if "%~1"=="" goto :arg_done
set "_a=%~1"
if /i "!_a:~0,8!"=="-vulkan=" set "SDK_OVERRIDE=!_a:~8!"
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

rem ── Run CMake ─────────────────────────────────────────────────────────────────
set "BUILD_DIR=build_vs"

if exist "%BUILD_DIR%\RealWorld.sln" (
    echo [INFO]  Regenerating existing solution in %BUILD_DIR%\
) else (
    echo [INFO]  Output directory: %BUILD_DIR%\
)

echo.
echo [INFO]  cmake -G "%VS_GENERATOR%" -A x64 -B %BUILD_DIR% !CMAKE_VULKAN!
echo.

cmake -G "%VS_GENERATOR%" -A x64 -B "%BUILD_DIR%" !CMAKE_VULKAN!

if errorlevel 1 (
    echo.
    echo [ERROR] CMake generation failed. See output above for details.
    echo.
    echo  Common fixes:
    echo    - Install Vulkan SDK:  https://vulkan.lunarg.com
    echo    - Specify SDK path:    GenerateProjectFiles.bat -vulkan="C:\VulkanSDK\1.3.x"
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
