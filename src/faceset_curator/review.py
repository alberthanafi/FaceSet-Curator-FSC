# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from .identity import identity_statistics
from .models import CuratorConfig, ImageAnalysis
from .reporting import write_reports
from .selection import select


CATEGORY_FILTERS = {
    "All": None,
    "Selected": "selected",
    "Rejected (all)": "rejected/",
    "Identity": "rejected/identity",
    "Quality": "rejected/quality",
    "Duplicates": "rejected/duplicate",
    "Redundant": "rejected/redundant",
    "Multiple faces": "rejected/multiple_faces",
    "No face / invalid": "rejected/no_face_or_invalid",
    "Manual exclusions": "rejected/manual",
}


def filter_items(items: list[ImageAnalysis], category_filter: str = "All",
                 minimum_identity: float = 0.0, minimum_quality: float = 0.0) -> list[ImageAnalysis]:
    category = CATEGORY_FILTERS.get(category_filter)
    filtered = []
    for item in items:
        if category == "rejected/" and not item.category.startswith("rejected/"):
            continue
        if category not in {None, "rejected/"} and item.category != category:
            continue
        if item.identity_score < minimum_identity or item.quality_score < minimum_quality:
            continue
        filtered.append(item)
    return sorted(filtered, key=lambda item: (item.category != "selected", item.category,
                                              -item.dataset_value, item.path))


def duplicate_peers(item: ImageAnalysis, items: list[ImageAnalysis]) -> list[ImageAnalysis]:
    if not item.duplicate_group:
        return []
    return sorted((peer for peer in items
                   if peer.path != item.path and peer.duplicate_group == item.duplicate_group),
                  key=lambda peer: (peer.category != "selected", -peer.quality_score, peer.path))


def why_rejected(item: ImageAnalysis) -> str:
    if item.category == "selected":
        return "; ".join(item.reasons) or "Selected for the collective set"
    return "; ".join(item.reasons) or item.category.replace("rejected/", "Rejected: ").replace("_", " ")


def apply_manual_selection(items: list[ImageAnalysis], config: CuratorConfig,
                           include_path: str | None = None,
                           exclude_path: str | None = None) -> dict:
    by_path = {item.path: item for item in items}
    include = by_path.get(include_path) if include_path else None
    exclude = by_path.get(exclude_path) if exclude_path else None
    if include_path and include is None:
        raise ValueError("The requested image is no longer available")
    if exclude_path and (exclude is None or exclude.category != "selected"):
        raise ValueError("Only a currently selected image can be excluded")
    if include and include.category not in {"selected", "rejected/redundant", "rejected/manual"}:
        raise ValueError("This rejection is safety-related and cannot be manually included")
    if include and exclude and include.path == exclude.path:
        raise ValueError("The included and excluded images must be different")

    old_selected = {item.path for item in items if item.category == "selected"}
    forced_paths = {item.path for item in items if item.category == "selected" and
                    any(reason.startswith("Manually included") for reason in item.reasons)}
    if include:
        forced_paths.add(include.path)
    if exclude:
        forced_paths.discard(exclude.path)

    pool = [item for item in items if item.category in {"selected", "rejected/redundant"}]
    if include and include not in pool:
        pool.append(include)
    if exclude:
        pool = [item for item in pool if item.path != exclude.path]
    for item in pool:
        item.category = "pending"
        item.dataset_value = 0.0
        item.score_components = {}
        item.reasons = []
    if exclude:
        exclude.category = "rejected/manual"
        exclude.dataset_value = 0.0
        exclude.score_components = {}
        exclude.reasons = ["Manually excluded during results review"]
    forced = [by_path[path] for path in sorted(forced_paths) if path in by_path and by_path[path] in pool]
    select(pool, config, forced=forced)
    new_selected = {item.path for item in items if item.category == "selected"}
    for item in items:
        if item.path in old_selected - new_selected and item.category == "rejected/redundant":
            item.reasons = ["Removed by manual review after collective values were recalculated"]
    return {
        "included": include.path if include else None,
        "excluded": exclude.path if exclude else None,
        "added": sorted(new_selected - old_selected),
        "removed": sorted(old_selected - new_selected),
        "selected_count": len(new_selected),
        "recalculated": True,
    }


def sync_review_output(run_dir: Path, items: list[ImageAnalysis], config: CuratorConfig,
                       previous_categories: dict[str, str], review_summary: dict) -> None:
    manifest_path = run_dir / ".fsc-materialized.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "version": 1, "destinations": {}
    }
    destinations: dict[str, str] = manifest.get("destinations", {})
    revision = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    history = run_dir / ".fsc-review-history" / revision
    history.mkdir(parents=True)
    for report_name in ("report.json", "report.csv", "report.html"):
        report_path = run_dir / report_name
        if report_path.is_file():
            shutil.copy2(report_path, history / report_name)

    for index, item in enumerate(items, start=1):
        if previous_categories.get(item.path) == item.category:
            continue
        old_relative = destinations.pop(item.path, None)
        old_copy = run_dir / old_relative if old_relative else None
        should_copy = item.category == "selected" or config.copy_rejected
        if should_copy:
            destination_dir = run_dir / item.category
            destination_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{index:04d}_" if item.category == "selected" else ""
            destination = destination_dir / f"{prefix}{item.source.name}"
            if destination.exists() and destination != old_copy:
                destination = destination_dir / f"{prefix}{item.fingerprint[:8]}_{item.source.name}"
            if old_copy and old_copy.is_file():
                os.replace(old_copy, destination)
            else:
                shutil.copy2(item.source, destination)
            destinations[item.path] = destination.relative_to(run_dir).as_posix()
        elif old_copy and old_copy.is_file():
            preserved = history / "removed-files"
            preserved.mkdir(exist_ok=True)
            os.replace(old_copy, preserved / old_copy.name)

    manifest["destinations"] = destinations
    manifest["last_review"] = review_summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    old_payload = {}
    old_report = history / "report.json"
    if old_report.is_file():
        old_payload = json.loads(old_report.read_text(encoding="utf-8"))
    write_reports(
        run_dir, items, config,
        identity_statistics(items, float(config.identity_threshold)),
        old_payload.get("reference_enrollment"), old_payload.get("performance"), review_summary,
    )
