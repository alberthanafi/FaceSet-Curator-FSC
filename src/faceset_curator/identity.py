# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import math
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ImageAnalysis


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not magnitude:
        raise ValueError("Cannot normalize an empty identity embedding")
    return [value / magnitude for value in vector]


def centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("At least one reference face is required")
    width = len(embeddings[0])
    if not width or any(len(item) != width for item in embeddings):
        raise ValueError("Reference embeddings have inconsistent dimensions")
    combined = [sum(item[index] for item in embeddings) for index in range(width)]
    return normalize(combined)


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(normalize(left), normalize(right)))


def confidence(similarity: float) -> float:
    """Map cosine [-1, 1] into a report-friendly [0, 1] confidence."""
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def reference_consistency(embeddings: list[list[float]], labels: list[str] | None = None) -> dict[str, Any]:
    labels = labels or [f"Reference {index + 1}" for index in range(len(embeddings))]
    pairs = [{"left": labels[left], "right": labels[right],
              "score": round(confidence(cosine(embeddings[left], embeddings[right])), 6)}
             for left in range(len(embeddings))
             for right in range(left + 1, len(embeddings))]
    scores = [pair["score"] for pair in pairs]
    minimum = min(scores) if scores else None
    average = sum(scores) / len(scores) if scores else None
    warning = None
    if len(embeddings) == 1:
        warning = "Only one reference was supplied; 2–5 varied, clear references are recommended."
    elif minimum is not None and minimum < 0.65:
        lowest = min(pairs, key=lambda pair: pair["score"])
        warning = (f"Reference images may not show the same person consistently "
                   f"({lowest['left']} vs {lowest['right']}: {minimum:.3f}). "
                   "Review or replace the outlier reference.")
    elif len(embeddings) > 5:
        warning = f"{len(embeddings)} references were supplied; use the clearest 2–5 images for reliable enrollment."
    return {
        "reference_count": len(embeddings),
        "pair_count": len(scores),
        "minimum_pair_score": round(minimum, 6) if minimum is not None else None,
        "average_pair_score": round(average, 6) if average is not None else None,
        "pairs": pairs,
        "warning": warning,
    }


def identity_statistics(items: list["ImageAnalysis"], threshold: float) -> dict[str, Any]:
    scores = sorted(item.identity_score for item in items if item.face_count > 0)
    buckets = {
        "below_0_50": sum(score < 0.50 for score in scores),
        "0_50_to_0_60": sum(0.50 <= score < 0.60 for score in scores),
        "0_60_to_0_70": sum(0.60 <= score < 0.70 for score in scores),
        "0_70_to_0_80": sum(0.70 <= score < 0.80 for score in scores),
        "0_80_and_above": sum(score >= 0.80 for score in scores),
    }
    return {
        "faces_scored": len(scores),
        "threshold": threshold,
        "passed_threshold": sum(score >= threshold for score in scores),
        "minimum": round(scores[0], 6) if scores else None,
        "median": round(median(scores), 6) if scores else None,
        "maximum": round(scores[-1], 6) if scores else None,
        "buckets": buckets,
    }
