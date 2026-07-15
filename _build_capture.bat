@echo off
REM Build the RealWorld target (Release) and capture all output to a log the
REM assistant can read back. Generated to verify the frame-capture changes.
set REPO=E:\work\procedure-world-sim
set LOG=%REPO%\build_log.txt
echo BUILD_STARTED %DATE% %TIME% > "%LOG%"
cd /d "%REPO%\build_vs"
"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release --target RealWorld >> "%LOG%" 2>&1
echo BUILD_EXIT_CODE=%ERRORLEVEL% >> "%LOG%"
echo BUILD_DONE %DATE% %TIME% >> "%LOG%"
