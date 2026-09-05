# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

import json
import logging
import tempfile
from pathlib import Path

from PIL import Image

from faceset_curator.app_logging import configure_logging
from faceset_curator.models import CuratorConfig, ImageAnalysis
from faceset_curator.preflight import InsufficientDiskSpace, check_disk_space
from faceset_curator.reporting import materialize


def test_disk_preflight_rejects_insufficient_space():
    with tempfile.TemporaryDirectory(prefix="fsc-space-test-") as temporary:
        root = Path(temporary)
        source, output = root / "source", root / "output"
        source.mkdir()
        Image.new("RGB", (10, 10), "white").save(source / "one.png")
        try:
            check_disk_space(source, output, CuratorConfig(), available_bytes=1)
        except InsufficientDiskSpace as exc:
            assert "Not enough free space" in str(exc)
        else:
            raise AssertionError("Expected disk preflight failure")


def test_materialize_reuses_verified_manifest_copy():
    with tempfile.TemporaryDirectory(prefix="fsc-copy-test-") as temporary:
        root = Path(temporary)
        source, run = root / "source.png", root / "run"
        Image.new("RGB", (10, 10), "white").save(source)
        run.mkdir()
        item = ImageAnalysis(str(source), "abcd" * 16, source.stat().st_size,
                             category="selected")
        materialize(run, [item], CuratorConfig(target_count=1))
        first = json.loads((run / ".fsc-materialized.json").read_text(encoding="utf-8"))
        destination = run / first["destinations"][str(source)]
        timestamp = destination.stat().st_mtime_ns
        materialize(run, [item], CuratorConfig(target_count=1))
        assert destination.stat().st_mtime_ns == timestamp
        assert len(list((run / "selected").iterdir())) == 1


def test_rotating_log_is_configured_once():
    with tempfile.TemporaryDirectory(prefix="fsc-log-test-") as temporary:
        path = Path(temporary) / "fsc.log"
        root = logging.getLogger("faceset_curator")
        existing_handlers = set(root.handlers)
        matching = []
        try:
            configure_logging(path)
            configure_logging(path)
            logger = logging.getLogger("faceset_curator.test")
            logger.warning("reliability test")
            matching = [
                handler for handler in root.handlers
                if handler not in existing_handlers
            ]
            assert len(matching) == 1
            for handler in matching:
                handler.flush()
            assert path.is_file()
            assert "reliability test" in path.read_text(encoding="utf-8")
        finally:
            for handler in matching:
                root.removeHandler(handler)
                handler.close()
