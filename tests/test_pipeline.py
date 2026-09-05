# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

import tempfile
import json
from pathlib import Path

from PIL import Image

from faceset_curator.analyzer import BaselineAnalyzer
from faceset_curator.models import CuratorConfig
from faceset_curator.pipeline import curate


class BatchBaselineAnalyzer(BaselineAnalyzer):
    def __init__(self):
        self.options = None
        self.performance_summary = {"batched_detection": True}
        self.warmed_up = False

    def warm_up(self, cancelled=None):
        self.warmed_up = True

    def analyze_many(self, entries, workers=0, queue_size=32, batch_size=0, cancelled=None):
        self.options = (workers, queue_size, batch_size)
        for path, digest in entries:
            yield self.analyze(path, digest)


def test_run_is_grouped_by_source_folder_name():
    with tempfile.TemporaryDirectory(prefix="fsc-test-") as temporary:
        root = Path(temporary)
        source = root / "person-name"
        output = root / "output"
        source.mkdir()
        Image.new("RGB", (32, 32), "white").save(source / "sample.png")

        progress_events = []
        run_dir, _ = curate(
            source,
            output,
            CuratorConfig(target_count=1, minimum_quality=0),
            BaselineAnalyzer(),
            progress_details=progress_events.append,
        )

        assert run_dir.parent == output / "person-name"
        assert run_dir.name.startswith("fsc-")
        assert (run_dir / "report.json").is_file()
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert report["identity_distribution"]["faces_scored"] == 1
        assert report["identity_distribution"]["threshold"] == 0.72
        stages = [event["stage"] for event in progress_events]
        for stage in ("preflight", "scanning", "fingerprinting", "cache_lookup", "analyzing",
                      "identity_filtering", "duplicate_filtering", "optimizing",
                      "clustering", "copying", "reporting", "complete"):
            assert stage in stages

        cached_events = []
        curate(source, output, CuratorConfig(target_count=1, minimum_quality=0),
               BaselineAnalyzer(), progress_details=cached_events.append)
        assert any(event.get("cache_hits") == 1 for event in cached_events)


def test_pipeline_uses_batch_analyzer_and_performance_settings():
    with tempfile.TemporaryDirectory(prefix="fsc-batch-test-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        for index in range(3):
            Image.new("RGB", (32, 32), (index * 50, 100, 150)).save(source / f"{index}.png")
        analyzer = BatchBaselineAnalyzer()
        progress_events = []
        run_dir, items = curate(source, root / "output",
                                CuratorConfig(target_count=1, minimum_quality=0, cache_enabled=False,
                                              cpu_workers=3, gpu_batch_size=7, decode_queue_size=11),
                                analyzer, progress_details=progress_events.append)
        assert len(items) == 3
        assert analyzer.warmed_up is True
        assert analyzer.options == (3, 11, 7)
        assert any(event["stage"] == "gpu_warmup" for event in progress_events)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert report["performance"]["batched_detection"] is True


def test_cancelled_run_is_marked_and_resumed():
    with tempfile.TemporaryDirectory(prefix="fsc-resume-test-") as temporary:
        root = Path(temporary)
        source, output = root / "source", root / "output"
        source.mkdir()
        Image.new("RGB", (32, 32), "white").save(source / "sample.png")
        cancelled = [False]

        class CancellingAnalyzer(BaselineAnalyzer):
            version = "resume-test-v1"

            def analyze(self, path, digest):
                cancelled[0] = True
                return super().analyze(path, digest)

        try:
            curate(source, output, CuratorConfig(target_count=1, minimum_quality=0),
                   CancellingAnalyzer(), cancelled=lambda: cancelled[0])
        except Exception:
            pass
        runs = list((output / "source").glob("fsc-*"))
        assert len(runs) == 1
        assert (runs[0] / "INCOMPLETE.txt").is_file()
        cancelled[0] = False
        events = []
        run_dir, _ = curate(source, output, CuratorConfig(target_count=1, minimum_quality=0),
                            CancellingAnalyzer(), progress_details=events.append)
        assert run_dir == runs[0]
        assert any(event["stage"] == "resuming" for event in events)
        assert not (run_dir / "INCOMPLETE.txt").exists()
        state = json.loads((run_dir / ".fsc-run.json").read_text(encoding="utf-8"))
        assert state["status"] == "complete"
        assert state["resumed"] == 1
