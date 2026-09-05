# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageFilter, ImageStat

from .models import ImageAnalysis


class Analyzer(Protocol):
    version: str
    def analyze(self, path: Path, fingerprint: str) -> ImageAnalysis: ...


class BaselineAnalyzer:
    """Runnable baseline. Production face semantics belong in the CUDA backend."""

    version = "baseline-pillow-v1"

    def analyze(self, path: Path, fingerprint: str) -> ImageAnalysis:
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                gray = image.convert("L")
                stat = ImageStat.Stat(gray)
                mean = stat.mean[0]
                exposure = max(0.0, 1.0 - abs(mean - 127.5) / 127.5)
                edges = gray.filter(ImageFilter.FIND_EDGES)
                sharpness = min(1.0, ImageStat.Stat(edges).var[0] / 1800.0)
                resolution = min(1.0, math.sqrt(width * height) / 1600.0)
                quality = 0.45 * sharpness + 0.30 * exposure + 0.25 * resolution
                phash = self._average_hash(gray)
                embedding = [int(fingerprint[i:i + 2], 16) / 255 for i in range(0, 32, 2)]
                return ImageAnalysis(
                    path=str(path), fingerprint=fingerprint, size_bytes=path.stat().st_size,
                    width=width, height=height, quality_score=round(quality, 6),
                    sharpness_score=round(sharpness, 6), exposure_score=round(exposure, 6),
                    phash=phash, embedding=embedding, visual_embedding=embedding,
                    flags=["baseline analyzer: face and identity semantics unavailable"],
                )
        except Exception as exc:
            return ImageAnalysis(
                path=str(path), fingerprint=fingerprint, size_bytes=path.stat().st_size,
                face_count=0, flags=[f"unreadable: {type(exc).__name__}"], category="invalid",
            )

    @staticmethod
    def _average_hash(gray: Image.Image) -> str:
        tiny = gray.resize((8, 8))
        values = list(tiny.getdata())
        average = sum(values) / len(values)
        bits = "".join("1" if value >= average else "0" for value in values)
        return f"{int(bits, 2):016x}"
