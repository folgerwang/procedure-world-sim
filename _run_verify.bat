@echo off
REM Run the already-built verify exe in one-shot dump mode. Uses fresh output
REM names each time to avoid any file locks from Explorer thumbnails/previews.
cd /d E:\work\procedure-world-sim\realworld
del screenshots\verify_clean.png 2>nul
del screenshots\verify_done.txt 2>nul
echo RUN_STARTED %DATE% %TIME% > verify_run2.log
RealWorld_verify.exe --verify-apply assets\map.png --verify-dump screenshots\verify_clean.png --verify-delay 15 >> verify_run2.log 2>&1
echo VERIFY_RUN_EXIT=%ERRORLEVEL% >> verify_run2.log
echo RUN_DONE %DATE% %TIME% >> verify_run2.log
