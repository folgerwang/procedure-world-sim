@echo off
REM ===========================================================================
REM Build RealWorld_verify.exe from the CURRENT source (which has the
REM --verify-apply / --verify-dump CLI). Renamed output so it never touches a
REM locked RealWorld.exe. Build-only: the Cowork loop drives the verify runs.
REM
REM Path-agnostic via %~dp0 -> works on whatever drive this repo lives on.
REM Requires CMake + Visual Studio 2022 (same toolchain that built the engine
REM before). All output is captured to build_log_verify.txt so it can be read
REM back over the mount.
REM ===========================================================================
setlocal
set "REPO=%~dp0"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"

set "LOG=%REPO%build_log_verify.txt"
echo VBUILD_STARTED %DATE% %TIME% > "%LOG%"
echo REPO=%REPO% >> "%LOG%"

echo Configuring (output name RealWorld_verify)...
"%CMAKE%" -S "%REPO%." -B "%REPO%build_vs" -DREALWORLD_OUTPUT_NAME=RealWorld_verify >> "%LOG%" 2>&1
echo --- CONFIGURE_EXIT=%ERRORLEVEL% --- >> "%LOG%"

echo Building Release (this can take a while)...
"%CMAKE%" --build "%REPO%build_vs" --config Release --target RealWorld >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo VBUILD_EXIT_CODE=%RC% >> "%LOG%"
echo VBUILD_DONE %DATE% %TIME% >> "%LOG%"

echo(
echo ============================================================
if "%RC%"=="0" (
  echo BUILD OK.  Expected: %REPO%realworld\RealWorld_verify.exe
) else (
  echo BUILD FAILED with code %RC%.  See %LOG%
)
echo Log: %LOG%
echo ============================================================
pause
