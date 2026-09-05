# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python -m pip install -e ".[gpu]" pyinstaller
python -c "import tkinter; root = tkinter.Tk(); root.withdraw(); root.destroy()"
python -m compileall -q src tests
python packaging\run_tests.py
python -m faceset_curator.cli doctor --device cuda
python -m PyInstaller --noconfirm --clean packaging\FaceSetCurator.spec
& "$ProjectRoot\dist\FaceSetCurator.exe" --doctor
if ($LASTEXITCODE -ne 0) {
    throw "Standalone executable CUDA verification failed."
}
& "$ProjectRoot\dist\FaceSetCurator.exe" --gui-smoke
if ($LASTEXITCODE -ne 0) {
    throw "Standalone executable GUI startup smoke failed."
}
Write-Host "Built dist\FaceSetCurator.exe"

$IsccCandidates = @(
    "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
    "$env:ProgramFiles(x86)\Inno Setup 7\ISCC.exe",
    "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($Iscc) {
    & $Iscc packaging\FaceSetCurator.iss
    if ($LASTEXITCODE -ne 0) { throw "Installer build failed." }
    Write-Host "Built dist\FaceSetCurator-Setup.exe"
} else {
    Write-Warning "Inno Setup 7 or 6 was not found. EXE is ready; install Inno Setup to build the installer."
}
