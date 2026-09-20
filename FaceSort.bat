@rem Copyright © 2026 Hanafi Mohd Radi. All rights reserved.
@echo off
setlocal
cd /d "%~dp0"
python -m faceset_curator.gui 2>nul
if errorlevel 1 (
  set "PYTHONPATH=%~dp0src"
  python -m faceset_curator.gui
)
endlocal
