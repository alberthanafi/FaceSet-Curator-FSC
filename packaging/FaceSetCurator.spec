# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.
# PyInstaller build definition for the Windows desktop application.
import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

datas, binaries, hiddenimports = [], [], []
for package in ("insightface", "onnxruntime", "cv2"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# ONNX Runtime's pip CUDA extras install runtime DLLs in namespace packages.
# Keep their directory layout so FSC can register each */bin directory.
for package in ("nvidia.cublas", "nvidia.cuda_nvrtc", "nvidia.cuda_runtime",
                "nvidia.cudnn", "nvidia.cufft", "nvidia.curand", "nvidia.nvjitlink"):
    binaries += collect_dynamic_libs(package)
    hiddenimports.append(package)

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[os.path.join(project_root, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas,
    name="FaceSetCurator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    version=os.path.join(SPECPATH, "version_info.txt"),
)
