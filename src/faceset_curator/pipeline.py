# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from collections.abc import Callable
import logging
import time
from typing import Any

from .analyzer import Analyzer, BaselineAnalyzer
from .cache import AnalysisCache
from .identity import identity_statistics
from .models import CuratorConfig, ImageAnalysis
from .reporting import materialize, write_reports
from .preflight import check_disk_space
from .run_state import RunJournal
from .scanner import scan
from .selection import eligible, remove_duplicates, select
from .visual_analysis import assign_diversity_clusters


class CurationCancelled(RuntimeError):
    pass


LOGGER = logging.getLogger(__name__)


def curate(source: Path, output_root: Path, config: CuratorConfig, analyzer: Analyzer | None = None,
           progress: Callable[[str, int, int], None] | None = None,
           progress_details: Callable[[dict[str, Any]], None] | None = None,
           cancelled: Callable[[], bool] | None = None) -> tuple[Path, list[ImageAnalysis]]:
    analyzer = analyzer or BaselineAnalyzer()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = check_disk_space(source, output_root, config)
    if progress_details:
        progress_details({"stage": "preflight", "current": preflight.required_bytes,
                          "total": preflight.available_bytes,
                          "message": f"Disk space checked for {preflight.image_count} images"})
    source_output = output_root / (source.resolve().name or "source")
    source_output.mkdir(parents=True, exist_ok=True)
    analyzer_version = getattr(analyzer, "version", type(analyzer).__name__)
    resumed_dir = RunJournal.find(source_output, source, config, analyzer_version)
    run_dir = resumed_dir or source_output / f"fsc-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    run_dir.mkdir(exist_ok=True)
    journal = RunJournal(run_dir, source, config, analyzer_version, resumed=bool(resumed_dir))

    def emit(stage: str, current: int = 0, total: int = 0, message: str | None = None, **details: Any) -> None:
        text = message or stage.replace("_", " ").title()
        payload = {"stage": stage, "current": current, "total": total,
                   "message": text, **details}
        journal.update(payload)
        if progress:
            progress(text, current, total)
        if progress_details:
            progress_details(payload)

    if resumed_dir:
        emit("resuming", 0, 0, f"Resuming incomplete run {run_dir.name}")
        LOGGER.info("Resuming run %s", run_dir)
    else:
        LOGGER.info("Starting run %s", run_dir)
    completed_successfully = False
    try:
        outcome = _curate_impl(source, output_root, run_dir, config, analyzer, emit, cancelled)
        completed_successfully = True
        return outcome
    except (InterruptedError, CurationCancelled) as exc:
        journal.fail(exc)
        LOGGER.info("Run interrupted: %s", run_dir)
        if isinstance(exc, CurationCancelled):
            raise
        raise CurationCancelled(str(exc)) from exc
    except BaseException as exc:
        journal.fail(exc)
        LOGGER.exception("Run failed: %s", run_dir)
        raise
    finally:
        if completed_successfully:
            journal.complete()
            LOGGER.info("Run complete: %s", run_dir)


def _curate_impl(source: Path, output_root: Path, run_dir: Path, config: CuratorConfig,
                 analyzer: Analyzer, emit: Callable[..., None],
                 cancelled: Callable[[], bool] | None) -> tuple[Path, list[ImageAnalysis]]:
    if hasattr(analyzer, "warm_up"):
        if cancelled and cancelled():
            raise CurationCancelled("Curation cancelled by user")
        emit("gpu_warmup", 0, 1, "Warming up batched GPU inference")
        try:
            analyzer.warm_up(cancelled)
        except TypeError:
            analyzer.warm_up()
        if cancelled and cancelled():
            raise CurationCancelled("Curation cancelled by user")
        emit("gpu_warmup", 1, 1, "GPU warm-up complete")
    emit("scanning", 0, 0, "Scanning source files")
    fingerprint_started = time.monotonic()

    def fingerprint_progress(current: int, total: int) -> None:
        elapsed = max(0.001, time.monotonic() - fingerprint_started)
        rate = current / elapsed
        emit("fingerprinting", current, total, "Fingerprinting source files", rate=rate,
             eta_seconds=(total - current) / rate if rate else None, rate_label="files/s")
    discovered = scan(
        source, config.cpu_workers,
        fingerprint_progress,
    )
    cache = AnalysisCache(output_root / ".fsc-cache" / "analysis.sqlite3")
    emit("cache_lookup", 0, len(discovered), "Checking analysis cache", cache_hits=0)
    try:
        unresolved: list[tuple[int, Path, str]] = []
        ordered_results: list[ImageAnalysis | None] = [None] * len(discovered)
        for index, (path, digest) in enumerate(discovered):
            if cancelled and cancelled():
                raise CurationCancelled("Curation cancelled by user")
            item = cache.get(digest, analyzer.version) if config.cache_enabled else None
            if item:
                item.path = str(path)
                ordered_results[index] = item
            else:
                unresolved.append((index, path, digest))
            if index == len(discovered) - 1 or (index + 1) % 250 == 0:
                emit("cache_lookup", index + 1, len(discovered), "Checking analysis cache",
                     cache_hits=(index + 1) - len(unresolved))
        completed = len(discovered) - len(unresolved)
        cache_hits = completed
        if completed:
            emit("cache_lookup", len(discovered), len(discovered),
                 f"Loaded {completed} cached analyses", cache_hits=cache_hits)
        if hasattr(analyzer, "analyze_many"):
            analyzed = analyzer.analyze_many(
                [(path, digest) for _, path, digest in unresolved],
                workers=config.cpu_workers,
                queue_size=config.decode_queue_size,
                batch_size=config.gpu_batch_size,
                cancelled=cancelled,
            )
        else:
            analyzed = (analyzer.analyze(path, digest) for _, path, digest in unresolved)
        analysis_started = time.monotonic()
        analyzed_new = 0
        for (result_index, path, _digest), item in zip(unresolved, analyzed):
            if cancelled and cancelled():
                raise CurationCancelled("Curation cancelled by user")
            ordered_results[result_index] = item
            completed += 1
            analyzed_new += 1
            if config.cache_enabled:
                cache.put(item, analyzer.version)
            elapsed = max(0.001, time.monotonic() - analysis_started)
            rate = analyzed_new / elapsed
            remaining = len(unresolved) - analyzed_new
            emit("analyzing", completed, len(discovered), f"Analyzing faces: {path.name}",
                 cache_hits=cache_hits, analyzed_new=analyzed_new,
                 analysis_total=len(unresolved), rate=rate, eta_seconds=remaining / rate if rate else None)
        if cancelled and cancelled():
            raise CurationCancelled("Curation cancelled by user")
        results = [item for item in ordered_results if item is not None]
        if len(results) != len(discovered):
            raise RuntimeError("Analyzer returned fewer results than requested")
    finally:
        cache.close()
    identity_summary = identity_statistics(results, config.identity_threshold)
    minimum = identity_summary["minimum"]
    middle = identity_summary["median"]
    maximum = identity_summary["maximum"]
    if minimum is None:
        identity_text = "Identity scoring complete: no faces scored"
    else:
        identity_text = (f"Identity scores {minimum:.3f}–{maximum:.3f}, median {middle:.3f}; "
                         f"{identity_summary['passed_threshold']} passed {config.identity_threshold:.2f}")
    emit("identity_filtering", len(discovered), len(discovered), identity_text,
         cache_hits=cache_hits, eligible_count=identity_summary["passed_threshold"])
    eligible_items = eligible(results, config)
    emit("duplicate_filtering", 0, len(eligible_items), "Filtering exact and near duplicates",
         cache_hits=cache_hits, eligible_count=len(eligible_items))
    candidates = remove_duplicates(eligible_items, config)
    emit("duplicate_filtering", len(eligible_items), len(eligible_items),
         f"Duplicate filtering complete: {len(candidates)} unique candidates",
         cache_hits=cache_hits, eligible_count=len(eligible_items), unique_count=len(candidates))
    emit("clustering", 0, len(candidates), "Clustering pose, expression, scene, and appearance",
         cache_hits=cache_hits, eligible_count=len(eligible_items), unique_count=len(candidates))
    assign_diversity_clusters(candidates)
    emit("clustering", len(candidates), len(candidates), "Diversity clustering complete",
         cache_hits=cache_hits, eligible_count=len(eligible_items), unique_count=len(candidates))
    emit("optimizing", 0, config.target_count, "Optimizing the best collective set",
         cache_hits=cache_hits, eligible_count=len(eligible_items), unique_count=len(candidates))
    select(candidates, config)
    selected_count = sum(item.category == "selected" for item in results)
    copy_total = sum(item.category == "selected" or config.copy_rejected for item in results)
    emit("copying", 0, copy_total, "Copying categorized output",
         cache_hits=cache_hits, eligible_count=len(eligible_items), selected_count=selected_count)
    copying_started = time.monotonic()

    def copying_progress(current: int, total: int) -> None:
        elapsed = max(0.001, time.monotonic() - copying_started)
        rate = current / elapsed
        emit("copying", current, total, "Copying categorized output",
             cache_hits=cache_hits, eligible_count=len(eligible_items), selected_count=selected_count,
             rate=rate, eta_seconds=(total - current) / rate if rate else None, rate_label="files/s")
    materialize(run_dir, results, config, copying_progress, cancelled)
    emit("reporting", 0, 3, "Generating JSON, CSV, and HTML reports",
         cache_hits=cache_hits, eligible_count=len(eligible_items), selected_count=selected_count)
    write_reports(run_dir, results, config, identity_summary,
                  getattr(analyzer, "enrollment_summary", None),
                  getattr(analyzer, "performance_summary", None),
                  progress=lambda current, total: emit(
                      "reporting", current, total, "Generating JSON, CSV, and HTML reports",
                      cache_hits=cache_hits, eligible_count=len(eligible_items),
                      selected_count=selected_count,
                  ))
    emit("complete", len(discovered), len(discovered), "Complete", cache_hits=cache_hits,
         eligible_count=len(eligible_items), selected_count=selected_count)
    return run_dir, results
