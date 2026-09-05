# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections.abc import Callable

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def discover_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    return [path.resolve() for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED]


def fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def automatic_cpu_workers() -> int:
    return max(2, min(8, (os.cpu_count() or 4) - 1))


def scan(root: Path, workers: int = 0,
         progress: Callable[[int, int], None] | None = None) -> list[tuple[Path, str]]:
    paths = discover_paths(root)
    resolved_workers = workers or automatic_cpu_workers()
    results = []
    with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="fsc-hash") as executor:
        for index, digest in enumerate(executor.map(fingerprint, paths), start=1):
            results.append((paths[index - 1], digest))
            if progress:
                progress(index, len(paths))
    return results
