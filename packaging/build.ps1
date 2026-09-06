# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python -m pip install -e ".[gpu]" pyinstaller
if ($LASTEXITCODE -ne 0) { throw "GPU build dependencies could not be installed." }
# InsightFace depends on the CPU distribution, whose files overlap the GPU
# distribution. Remove it and reinstall the GPU wheel last so CUDA wins.
python -m pip uninstall -y onnxruntime
if ($LASTEXITCODE -ne 0) { throw "CPU ONNX Runtime could not be removed." }
python -m pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.21,<1.27"
if ($LASTEXITCODE -ne 0) { throw "GPU ONNX Runtime could not be installed." }
python -c "import tkinter; root = tkinter.Tk(); root.withdraw(); root.destroy()"
if ($LASTEXITCODE -ne 0) { throw "Tkinter startup verification failed." }
python -m compileall -q src tests
if ($LASTEXITCODE -ne 0) { throw "Source compilation failed." }
python packaging\run_tests.py
if ($LASTEXITCODE -ne 0) { throw "Release tests failed." }
python -m faceset_curator.cli doctor --device cuda
if ($LASTEXITCODE -ne 0) { throw "Build environment CUDA verification failed." }
python -m PyInstaller --noconfirm --clean packaging\FaceSetCurator.spec
if ($LASTEXITCODE -ne 0) { throw "Standalone executable build failed." }
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
    "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
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
