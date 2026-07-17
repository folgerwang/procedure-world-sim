@echo off
setlocal
set "REPO=%~dp0"
cd /d "%REPO%realworld"
taskkill /f /im RealWorld_verify.exe >nul 2>&1
set "LOG=content\terrain\.cowork\town300.log"
echo TOWN2_STARTED %DATE% %TIME% > "%LOG%"
del screenshots\houses_town300.png 2>nul
del screenshots\terrain_verify_done.txt 2>nul
RealWorld_verify.exe --verify-apply content\terrain\terrain_1784007822.png --verify-color content\terrain\terrain_1784007822_color.png --verify-topdown --verify-cam-height=300 --verify-cam-x=-151 --verify-cam-z=-275 --verify-dump screenshots\houses_town300.png --verify-delay 12 >> "%LOG%" 2>&1
echo TOWN2_EXIT=%ERRORLEVEL% >> "%LOG%"
taskkill /f /im RealWorld_verify.exe >nul 2>&1
echo TOWN2_DONE %DATE% %TIME% >> "%LOG%"
