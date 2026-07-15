@echo off
REM Build RealWorld_verify.exe (renamed output so it never touches the locked
REM RealWorld.exe), then, if the build succeeds, run it in one-shot verify-dump
REM mode. All output goes to files that can be read back over the command line.
set REPO=E:\work\procedure-world-sim
set CMAKE="C:\Program Files\CMake\bin\cmake.exe"
set LOG=%REPO%\build_log_verify.txt
echo VBUILD_STARTED %DATE% %TIME% > "%LOG%"
%CMAKE% -S "%REPO%" -B "%REPO%\build_vs" -DREALWORLD_OUTPUT_NAME=RealWorld_verify >> "%LOG%" 2>&1
echo --- CONFIGURE_EXIT=%ERRORLEVEL% --- >> "%LOG%"
%CMAKE% --build "%REPO%\build_vs" --config Release --target RealWorld >> "%LOG%" 2>&1
set BUILD_RC=%ERRORLEVEL%
echo VBUILD_EXIT_CODE=%BUILD_RC% >> "%LOG%"
echo VBUILD_DONE %DATE% %TIME% >> "%LOG%"
if not "%BUILD_RC%"=="0" goto :eof

REM --- build OK: run the verify dump ---
cd /d %REPO%\realworld
del screenshots\terrain_verify_done.txt 2>nul
del screenshots\verify_test.png 2>nul
echo RUN_STARTED %DATE% %TIME% > verify_run.log
RealWorld_verify.exe --verify-apply assets\map.png --verify-dump screenshots\verify_test.png --verify-delay 15 >> verify_run.log 2>&1
echo VERIFY_RUN_EXIT=%ERRORLEVEL% >> verify_run.log
echo RUN_DONE %DATE% %TIME% >> verify_run.log
