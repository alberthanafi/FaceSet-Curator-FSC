# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


class ModelDownloadError(RuntimeError):
    pass


BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_L_ARCHIVE_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
BUFFALO_L_FILES = {
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
VERIFICATION_STAMP = ".fsc-verified.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_state(model_dir: Path) -> dict[str, dict[str, int]] | None:
    state: dict[str, dict[str, int]] = {}
    for filename in BUFFALO_L_FILES:
        path = model_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        stat = path.stat()
        state[filename] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return state


def validate_buffalo_l(model_dir: Path, full: bool = False) -> tuple[bool, str]:
    state = _file_state(model_dir)
    if state is None:
        missing = [name for name in BUFFALO_L_FILES if not (model_dir / name).is_file()]
        return False, f"missing model files: {', '.join(missing) if missing else 'empty model file'}"
    stamp_path = model_dir / VERIFICATION_STAMP
    if not full and stamp_path.is_file():
        try:
            stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            if stamp.get("archive_sha256") == BUFFALO_L_ARCHIVE_SHA256 and stamp.get("files") == state:
                return True, "verified"
        except (OSError, ValueError, TypeError):
            pass
    for filename, expected in BUFFALO_L_FILES.items():
        if sha256_file(model_dir / filename) != expected:
            return False, f"checksum mismatch: {filename}"
    stamp_path.write_text(json.dumps({"archive_sha256": BUFFALO_L_ARCHIVE_SHA256,
                                      "files": state}, indent=2), encoding="utf-8")
    return True, "verified"


def _download_once(url: str, destination: Path, progress: Callable[[dict[str, Any]], None] | None = None,
                   cancelled: Callable[[], bool] | None = None,
                   opener: Callable[..., Any] = urllib.request.urlopen) -> None:
    existing = destination.stat().st_size if destination.is_file() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "FaceSet-Curator/0.1"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    try:
        response_context = opener(request, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and destination.is_file():
            return
        raise
    with response_context as response:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        append = existing > 0 and status == 206
        if existing and not append:
            existing = 0
        content_length = int(response.headers.get("Content-Length", "0") or 0)
        total = existing + content_length if content_length else 0
        mode = "ab" if append else "wb"
        received = existing
        with destination.open(mode) as stream:
            while True:
                if cancelled and cancelled():
                    raise ModelDownloadError("Model download cancelled; the partial download was saved for resuming.")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                received += len(chunk)
                if progress:
                    progress({"stage": "model_download", "current": received, "total": total,
                              "message": "Downloading verified InsightFace models"})


def download_with_retries(url: str, destination: Path,
                          progress: Callable[[dict[str, Any]], None] | None = None,
                          cancelled: Callable[[], bool] | None = None,
                          attempts: int = 4, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _download_once(url, destination, progress, cancelled, opener)
            return
        except ModelDownloadError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if progress:
                progress({"stage": "model_download", "current": destination.stat().st_size if destination.exists() else 0,
                          "total": 0, "message": f"Download interrupted; retrying ({attempt}/{attempts})"})
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise ModelDownloadError(
        f"The InsightFace model download failed after {attempts} attempts. "
        "The partial file was kept and FSC will resume it next time."
    ) from last_error


def _extract_verified_archive(archive_path: Path, staging: Path) -> None:
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for member in archive.infolist():
                by_name.setdefault(Path(member.filename).name, []).append(member)
            for filename in BUFFALO_L_FILES:
                matches = by_name.get(filename, [])
                if len(matches) != 1:
                    raise ModelDownloadError(f"Downloaded archive does not contain exactly one {filename}.")
                with archive.open(matches[0]) as source, (staging / filename).open("wb") as target:
                    shutil.copyfileobj(source, target)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ModelDownloadError("Downloaded InsightFace archive is incomplete or invalid.") from exc
    valid, reason = validate_buffalo_l(staging, full=True)
    if not valid:
        raise ModelDownloadError(f"Downloaded InsightFace models failed verification: {reason}.")


def ensure_buffalo_l(root: Path | None = None,
                     progress: Callable[[dict[str, Any]], None] | None = None,
                     cancelled: Callable[[], bool] | None = None) -> Path:
    insightface_root = root or (Path.home() / ".insightface")
    models_root = insightface_root / "models"
    model_dir = models_root / "buffalo_l"
    valid, reason = validate_buffalo_l(model_dir)
    if valid:
        if progress:
            progress({"stage": "model_verification", "current": 1, "total": 1,
                      "message": "InsightFace models verified"})
        return model_dir
    if progress:
        progress({"stage": "model_verification", "current": 0, "total": 1,
                  "message": f"Model repair required ({reason})"})
    models_root.mkdir(parents=True, exist_ok=True)
    partial = models_root / "buffalo_l.zip.part"
    legacy = models_root / "buffalo_l.zip"
    if legacy.is_file() and not partial.exists():
        os.replace(legacy, partial)
    download_with_retries(BUFFALO_L_URL, partial, progress, cancelled)
    if sha256_file(partial) != BUFFALO_L_ARCHIVE_SHA256:
        invalid = models_root / f"buffalo_l-invalid-{int(time.time())}.zip"
        os.replace(partial, invalid)
        raise ModelDownloadError(
            f"The downloaded model checksum is wrong. It was preserved as {invalid.name}; "
            "start again to download a clean copy."
        )
    staging = models_root / f".buffalo_l-installing-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        _extract_verified_archive(partial, staging)
        if model_dir.exists():
            backup = models_root / f"buffalo_l-invalid-{int(time.time())}"
            os.replace(model_dir, backup)
        os.replace(staging, model_dir)
        partial.unlink()
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if progress:
        progress({"stage": "model_verification", "current": 1, "total": 1,
                  "message": "InsightFace models installed and verified"})
    return model_dir
