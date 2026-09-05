# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CuratorConfig:
    target_count: int = 100
    profile: str = "balanced"
    duplicate_strength: str = "strong"
    identity_verification: str = "high"
    identity_threshold: float | None = None
    minimum_quality: float = 0.30
    near_duplicate_hamming: int = 8
    cache_enabled: bool = True
    copy_rejected: bool = True
    require_single_face: bool = True
    cpu_workers: int = 0
    gpu_batch_size: int = 0
    decode_queue_size: int = 32

    def __post_init__(self) -> None:
        if self.profile not in {"balanced", "quality", "diversity"}:
            raise ValueError(f"Unsupported selection profile: {self.profile}")
        if self.duplicate_strength not in {"strong", "normal", "exact"}:
            raise ValueError(f"Unsupported duplicate mode: {self.duplicate_strength}")
        if self.identity_verification not in {"high", "normal", "custom"}:
            raise ValueError(f"Unsupported identity verification: {self.identity_verification}")
        threshold = self.identity_threshold
        if threshold is None:
            if self.identity_verification == "custom":
                raise ValueError("Custom identity verification requires a threshold")
            threshold = {"high": 0.72, "normal": 0.65}[self.identity_verification]
            object.__setattr__(self, "identity_threshold", threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Identity threshold must be between 0 and 1")
        if self.cpu_workers < 0 or self.gpu_batch_size < 0:
            raise ValueError("Worker and GPU batch settings cannot be negative")
        if self.decode_queue_size < 1:
            raise ValueError("Decode queue size must be at least 1")


@dataclass
class ImageAnalysis:
    path: str
    fingerprint: str
    size_bytes: int
    width: int = 0
    height: int = 0
    face_count: int = 1
    detection_score: float = 0.0
    face_area_ratio: float = 0.0
    identity_score: float = 1.0
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    blur_score: float = 0.0
    occlusion_score: float = 0.0
    compression_artifact_score: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    expression: str = "unknown"
    expression_confidence: float = 0.0
    phash: str = ""
    embedding: list[float] = field(default_factory=list)
    visual_embedding: list[float] = field(default_factory=list)
    scene_embedding: list[float] = field(default_factory=list)
    appearance_embedding: list[float] = field(default_factory=list)
    pose_cluster: str = ""
    scene_cluster: str = ""
    appearance_cluster: str = ""
    session_cluster: str = ""
    flags: list[str] = field(default_factory=list)
    category: str = "pending"
    duplicate_group: str | None = None
    dataset_value: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def source(self) -> Path:
        return Path(self.path)
