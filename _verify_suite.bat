@echo off
REM ===========================================================================
REM Terrain generation self-improvement suite.
REM
REM Runs tools/terrain/verify_loop.py across a spread of biomes. Each biome:
REM   generate heightmap -> engine one-shot verify-dump -> Claude scores the
REM   frame against the rubric -> on fail, Claude revises prompt/height-scale
REM   and regenerates (up to --max-iters).
REM
REM Path-agnostic: REPO is derived from THIS file's location (%~dp0), so it
REM works whether the repo lives on E:\ or G:\.
REM
REM Requires: ANTHROPIC_API_KEY set in the environment, a built RealWorld.exe,
REM and a Python interpreter with the terrain/torch/FLUX deps (edit PY below
REM if you use a venv).
REM ===========================================================================
setlocal enabledelayedexpansion

set "REPO=%~dp0"
cd /d "%REPO%realworld"

REM --- Python interpreter (point at your venv if needed) ---------------------
set "PY=python"

REM --- API key check --------------------------------------------------------
if "%ANTHROPIC_API_KEY%"=="" (
  echo(
  echo ERROR: ANTHROPIC_API_KEY is not set in this environment.
  echo Set it, then open a NEW terminal / Explorer session and re-run:
  echo(
  echo     setx ANTHROPIC_API_KEY sk-ant-xxxxxxxx
  echo(
  pause
  exit /b 1
)

set "OUTBASE=content\terrain\.verify_suite"
set "LOG=%CD%\verify_suite.log"

echo SUITE_STARTED %DATE% %TIME% > "%LOG%"
echo Repo: %REPO% >> "%LOG%"
echo Python: %PY% >> "%LOG%"

echo(
echo Running terrain self-improvement suite. This uses the GPU and the Claude
echo API; each biome may take several minutes. Progress is streamed to:
echo   %LOG%
echo(

call :run volcanic_island  "volcanic island with a caldera lake and dark basalt slopes"
call :run desert_canyon    "desert canyon with layered mesas, steep walls and a dry riverbed"
call :run snowy_alpine     "snowy alpine mountain range with sharp ridgelines and glacial valleys"
call :run wuxia_valley     "jagged wuxia mountain valley with a winding river canyon and towering peaks"
call :run rolling_farmland "rolling green farmland with gentle hills and a meandering river"

echo SUITE_DONE %DATE% %TIME% >> "%LOG%"
echo(
echo ============================================================
echo All runs complete.
echo   Reports : %OUTBASE%\<label>\verify_report.json
echo   Log     : %LOG%
echo ============================================================
pause
exit /b 0

REM ---------------------------------------------------------------------------
:run
set "LABEL=%~1"
set "PROMPT=%~2"
echo( >> "%LOG%"
echo ====================================================== >> "%LOG%"
echo RUN %LABEL% : %PROMPT%   (%DATE% %TIME%) >> "%LOG%"
echo ------------------------------------------------------ >> "%LOG%"
echo [suite] %LABEL% ...
"%PY%" tools\terrain\verify_loop.py --prompt "%PROMPT%" --color --max-iters 4 --out-dir "%OUTBASE%\%LABEL%" --report "%OUTBASE%\%LABEL%\verify_report.json" >> "%LOG%" 2>&1
echo RUN %LABEL% EXIT=%ERRORLEVEL% >> "%LOG%"
echo [suite] %LABEL% done (exit %ERRORLEVEL%^)
goto :eof
