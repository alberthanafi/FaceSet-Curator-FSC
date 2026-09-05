# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

import json
import tempfile
from pathlib import Path

from PIL import Image

from faceset_curator.models import CuratorConfig, ImageAnalysis
from faceset_curator.reporting import materialize, write_reports
from faceset_curator.review import (apply_manual_selection, duplicate_peers, filter_items,
                                    sync_review_output, why_rejected)
from faceset_curator.selection import select


def review_item(path: Path, index: int, quality: float, embedding: list[float]) -> ImageAnalysis:
    return ImageAnalysis(path=str(path), fingerprint=f"{index:064x}", size_bytes=path.stat().st_size,
                         quality_score=quality, identity_score=0.9,
                         phash=f"{index * 1000:016x}", embedding=embedding)


def test_result_filters_and_rejection_explanation():
    selected = ImageAnalysis("a.jpg", "a" * 64, 1, category="selected", identity_score=.9,
                             quality_score=.8, reasons=["Adds pose coverage"])
    rejected = ImageAnalysis("b.jpg", "b" * 64, 1, category="rejected/identity", identity_score=.4,
                             quality_score=.9, reasons=["Identity below threshold"])
    assert filter_items([selected, rejected], "Selected") == [selected]
    assert filter_items([selected, rejected], "Rejected (all)") == [rejected]
    assert filter_items([selected, rejected], "All", minimum_identity=.8) == [selected]
    assert why_rejected(rejected) == "Identity below threshold"


def test_duplicate_group_returns_representative_for_comparison():
    kept = ImageAnalysis("kept.jpg", "a" * 64, 1, category="selected", quality_score=.9,
                         duplicate_group="group")
    duplicate = ImageAnalysis("copy.jpg", "b" * 64, 1, category="rejected/duplicate",
                              quality_score=.8, duplicate_group="group")
    assert duplicate_peers(duplicate, [kept, duplicate]) == [kept]


def test_manual_swap_keeps_target_and_recalculates_values():
    items = [ImageAnalysis(f"{name}.jpg", name * 64, 1, quality_score=quality,
                           identity_score=.9, embedding=embedding)
             for name, quality, embedding in (("a", .95, [1, 0]), ("b", .9, [0, 1]),
                                               ("c", .8, [.7, .7]))]
    config = CuratorConfig(target_count=2)
    select(items, config)
    candidate = next(item for item in items if item.category == "rejected/redundant")
    removed = next(item for item in items if item.category == "selected")
    summary = apply_manual_selection(items, config, candidate.path, removed.path)
    assert sum(item.category == "selected" for item in items) == 2
    assert candidate.category == "selected"
    assert candidate.reasons == ["Manually included; collective values recalculated"]
    assert removed.category == "rejected/manual"
    assert summary["recalculated"] is True
    assert all(item.dataset_value > 0 for item in items if item.category == "selected")


def test_manual_include_blocks_safety_rejection():
    item = ImageAnalysis("unsafe.jpg", "a" * 64, 1, category="rejected/identity")
    try:
        apply_manual_selection([item], CuratorConfig(target_count=1), include_path=item.path)
    except ValueError as exc:
        assert "safety-related" in str(exc)
    else:
        raise AssertionError("A safety rejection must not be manually included")


def test_review_sync_updates_copies_manifest_and_reports():
    with tempfile.TemporaryDirectory(prefix="fsc-review-test-") as temporary:
        root = Path(temporary)
        source = root / "source"
        run_dir = root / "run"
        source.mkdir()
        run_dir.mkdir()
        paths = []
        for index in range(3):
            path = source / f"{index}.png"
            Image.new("RGB", (24, 24), (index * 60, 80, 140)).save(path)
            paths.append(path)
        items = [review_item(paths[index], index + 1, .95 - index * .08,
                             [1.0, float(index)]) for index in range(3)]
        config = CuratorConfig(target_count=2, copy_rejected=True)
        select(items, config)
        materialize(run_dir, items, config)
        write_reports(run_dir, items, config)
        previous = {item.path: item.category for item in items}
        candidate = next(item for item in items if item.category == "rejected/redundant")
        removed = next(item for item in items if item.category == "selected")
        summary = apply_manual_selection(items, config, candidate.path, removed.path)
        sync_review_output(run_dir, items, config, previous, summary)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / ".fsc-materialized.json").read_text(encoding="utf-8"))
        assert report["manual_review"]["recalculated"] is True
        assert len(list((run_dir / "selected").glob("*.png"))) == 2
        assert all((run_dir / relative).is_file() for relative in manifest["destinations"].values())
        assert any((run_dir / ".fsc-review-history").rglob("report.json"))
