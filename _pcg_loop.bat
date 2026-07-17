@echo off
setlocal
set "REPO=%~dp0"
cd /d "%REPO%realworld"
set "LOG=content\terrain\.cowork\pcg_loop.log"
echo LOOP_STARTED %DATE% %TIME% > "%LOG%"
python tools\terrain\pcg_verify_loop.py --color content\terrain\terrain_1784007822_color.png --height content\terrain\terrain_1784007822.png --seg content\terrain\terrain_1784007822_seg.png --max-iters 4 >> "%LOG%" 2>&1
echo LOOP_EXIT=%ERRORLEVEL% >> "%LOG%"
set "TERRAIN_PCG_CFG=%CD%\content\terrain\.pcg_loop\best_cfg.json"
python tools\terrain\terrain_pcg.py --color content\terrain\terrain_1784007822_color.png --height content\terrain\terrain_1784007822.png --seg content\terrain\terrain_1784007822_seg.png --out content\terrain\terrain_1784007822_pcg.glb >> "%LOG%" 2>&1
echo REGEN_EXIT=%ERRORLEVEL% >> "%LOG%"
python tools\terrain\verify_buildings_overlay.py --glb content\terrain\terrain_1784007822_pcg.glb --color content\terrain\terrain_1784007822_color.png >> "%LOG%" 2>&1
echo OVERLAY_EXIT=%ERRORLEVEL% >> "%LOG%"
echo LOOP_DONE %DATE% %TIME% >> "%LOG%"
