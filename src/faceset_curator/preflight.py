# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import CuratorConfig
from .scanner import discover_paths


class InsufficientDiskSpace(RuntimeError):
    pass


@dataclass(frozen=True)
class DiskPreflight:
    image_count: int
    source_bytes: int
    copy_bytes: int
    overhead_bytes: int
    required_bytes: int
    available_bytes: int


def check_disk_space(source: Path, output_root: Path, config: CuratorConfig,
                     available_bytes: int | None = None) -> DiskPreflight:
    paths = discover_paths(source)
    sizes = [path.stat().st_size for path in paths]
    source_bytes = sum(sizes)
    copy_bytes = source_bytes if config.copy_rejected else sum(
        sorted(sizes, reverse=True)[:config.target_count]
    )
    overhead = max(256 * 1024 * 1024, int(source_bytes * 0.05))
    required = copy_bytes + overhead
    output_root.mkdir(parents=True, exist_ok=True)
    available = available_bytes if available_bytes is not None else shutil.disk_usage(output_root).free
    result = DiskPreflight(len(paths), source_bytes, copy_bytes, overhead, required, available)
    if available < required:
        needed_gib = required / 1024 ** 3
        free_gib = available / 1024 ** 3
        raise InsufficientDiskSpace(
            f"Not enough free space in the output location: {free_gib:.2f} GiB available, "
            f"approximately {needed_gib:.2f} GiB required. Choose another output drive or "
            "turn off Copy rejected."
        )
    return result
