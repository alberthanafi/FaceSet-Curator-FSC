# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import csv
import html
import json
import shutil
import os
from datetime import datetime, timezone
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .models import CuratorConfig, ImageAnalysis
from . import __copyright__


def materialize(run_dir: Path, items: list[ImageAnalysis], config: CuratorConfig,
                progress=None, cancelled=None) -> None:
    manifest_path = run_dir / ".fsc-materialized.json"
    destinations = {}
    if manifest_path.is_file():
        try:
            destinations = json.loads(manifest_path.read_text(encoding="utf-8")).get("destinations", {})
        except (OSError, ValueError):
            destinations = {}

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "destinations": destinations}, indent=2),
                             encoding="utf-8")
        os.replace(temporary, manifest_path)

    copied_items = [(index, item) for index, item in enumerate(items, start=1)
                    if item.category == "selected" or config.copy_rejected]
    try:
        for copy_index, (index, item) in enumerate(copied_items, start=1):
            if cancelled and cancelled():
                raise InterruptedError("Curation cancelled while copying output")
            existing = destinations.get(item.path)
            destination = run_dir / existing if existing else None
            if not destination or not destination.is_file() or destination.stat().st_size != item.size_bytes:
                destinations.pop(item.path, None)
                destination_dir = run_dir / item.category
                destination_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{index:04d}_" if item.category == "selected" else ""
                destination = destination_dir / f"{prefix}{item.source.name}"
                if destination.is_file() and destination.stat().st_size == item.size_bytes:
                    destinations[item.path] = destination.relative_to(run_dir).as_posix()
                elif destination.exists():
                    destination = destination_dir / f"{prefix}{item.fingerprint[:8]}_{item.source.name}"
                if item.path not in destinations:
                    temporary = destination.with_name(f".{destination.name}.fsc-copying")
                    shutil.copy2(item.source, temporary)
                    os.replace(temporary, destination)
                    destinations[item.path] = destination.relative_to(run_dir).as_posix()
            if progress:
                progress(copy_index, len(copied_items))
            if copy_index % 25 == 0:
                save_manifest()
    finally:
        save_manifest()


def write_reports(run_dir: Path, items: list[ImageAnalysis], config: CuratorConfig,
                  identity_summary: dict | None = None,
                  enrollment_summary: dict | None = None,
                  performance_summary: dict | None = None,
                  manual_review: dict | None = None,
                  progress=None) -> None:
    payload = {"program": "FaceSet Curator", "version": "0.1.0", "copyright": __copyright__,
               "generated_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
               "summary": dict(Counter(item.category for item in items)),
               "identity_distribution": identity_summary or {},
               "reference_enrollment": enrollment_summary or {},
               "performance": performance_summary or {},
               "manual_review": manual_review or {},
               "images": [item.to_dict() for item in items]}
    (run_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if progress:
        progress(1, 3)
    with (run_dir / "report.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "path", "category", "dataset_value", "quality_score", "identity_score", "expression",
            "blur_score", "occlusion_score", "compression_artifact_score", "pose_cluster",
            "appearance_cluster", "scene_cluster", "session_cluster", "duplicate_group", "reasons",
            "copyright",
        ])
        writer.writeheader()
        for item in items:
            writer.writerow({"path": item.path, "category": item.category, "dataset_value": item.dataset_value,
                             "quality_score": item.quality_score, "identity_score": item.identity_score,
                             "expression": item.expression, "blur_score": item.blur_score,
                             "occlusion_score": item.occlusion_score,
                             "compression_artifact_score": item.compression_artifact_score,
                             "pose_cluster": item.pose_cluster, "appearance_cluster": item.appearance_cluster,
                             "scene_cluster": item.scene_cluster, "session_cluster": item.session_cluster,
                             "duplicate_group": item.duplicate_group, "reasons": "; ".join(item.reasons),
                             "copyright": __copyright__})
    if progress:
        progress(2, 3)
    selected = sorted((item for item in items if item.category == "selected"), key=lambda x: -x.dataset_value)
    identity = identity_summary or {}
    identity_line = ""
    if identity.get("minimum") is not None:
        identity_line = (f"<p>Identity scores: {identity['minimum']:.3f}–{identity['maximum']:.3f}; "
                         f"median {identity['median']:.3f}; {identity['passed_threshold']} passed "
                         f"threshold {identity['threshold']:.2f}.</p>")
    enrollment = enrollment_summary or {}
    enrollment_line = ""
    if enrollment:
        enrollment_line = f"<p>References: {enrollment.get('reference_count', 0)}"
        if enrollment.get("minimum_pair_score") is not None:
            enrollment_line += f"; lowest pair score {enrollment['minimum_pair_score']:.3f}"
        enrollment_line += ".</p>"
    review = manual_review or {}
    review_line = ""
    if review:
        review_line = (f"<p>Manual review: {len(review.get('added', []))} added; "
                       f"{len(review.get('removed', []))} removed; collective values recalculated.</p>")
    cards = "".join(f"<article><h3>{html.escape(x.source.name)}</h3><p>Dataset value: {x.dataset_value:.1f}</p><p>Quality: {x.quality_score:.2f} · Identity: {x.identity_score:.2f}</p><p>Expression: {html.escape(x.expression)} · Pose: {html.escape(x.pose_cluster or 'unknown')}</p><p>Scene: {html.escape(x.scene_cluster or 'unknown')} · Appearance: {html.escape(x.appearance_cluster or 'unknown')}</p><p>{html.escape('; '.join(x.reasons))}</p></article>" for x in selected)
    page = f"""<!doctype html><!-- {__copyright__} --><meta charset='utf-8'><title>FaceSet Curator report</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#17202a}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}article{{border:1px solid #ccd6dd;border-radius:12px;padding:16px}}h1{{margin-bottom:4px}}footer{{margin-top:32px;color:#667}}</style><h1>FaceSet Curator</h1><p>Selected {len(selected)} of {len(items)} analyzed images.</p>{identity_line}{enrollment_line}{review_line}<div class='grid'>{cards}</div><footer>{html.escape(__copyright__)}</footer>"""
    (run_dir / "report.html").write_text(page, encoding="utf-8")
    if progress:
        progress(3, 3)
