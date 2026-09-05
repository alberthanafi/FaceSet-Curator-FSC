# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y", "positive", "same", "accept"}
FALSE_VALUES = {"0", "false", "no", "n", "negative", "different", "reject"}


def _label(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported benchmark label: {value}")


def calibrate_threshold(samples: list[tuple[float, bool]]) -> dict[str, Any]:
    if not samples or not any(label for _, label in samples) or not any(not label for _, label in samples):
        raise ValueError("Calibration requires at least one positive and one negative labeled sample")
    values = sorted({max(0.0, min(1.0, score)) for score, _ in samples})
    thresholds = [0.0] + [(left + right) / 2.0 for left, right in zip(values, values[1:])] + [1.0]
    best = None
    for threshold in thresholds:
        tp = sum(score >= threshold and label for score, label in samples)
        tn = sum(score < threshold and not label for score, label in samples)
        fp = sum(score >= threshold and not label for score, label in samples)
        fn = sum(score < threshold and label for score, label in samples)
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        precision = tp / max(1, tp + fp)
        balanced_accuracy = (recall + specificity) / 2.0
        candidate = (balanced_accuracy, precision, recall, threshold, tp, tn, fp, fn)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    balanced_accuracy, precision, recall, threshold, tp, tn, fp, fn = best
    return {
        "threshold": round(threshold, 6), "balanced_accuracy": round(balanced_accuracy, 6),
        "precision": round(precision, 6), "recall": round(recall, 6),
        "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
        "sample_count": len(samples),
    }


def calibrate_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "identity_score" not in rows[0] or "same_identity" not in rows[0]:
        raise ValueError("Benchmark CSV requires identity_score and same_identity columns")
    identity = [(float(row["identity_score"]), _label(row["same_identity"])) for row in rows]
    result: dict[str, Any] = {"benchmark": str(path), "identity": calibrate_threshold(identity)}
    if "quality_score" in rows[0] and "acceptable_quality" in rows[0]:
        quality = [(float(row["quality_score"]), _label(row["acceptable_quality"])) for row in rows
                   if row.get("quality_score", "").strip() and row.get("acceptable_quality", "").strip()]
        if quality:
            result["quality"] = calibrate_threshold(quality)
    return result
