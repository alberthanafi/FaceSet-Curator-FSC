# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path


def default_log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "FaceSetCurator" / "logs" / "fsc.log"


def configure_logging(path: Path | None = None) -> Path:
    log_path = path or default_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8"):
            pass
    except OSError:
        if path is not None:
            raise
        log_path = Path(tempfile.gettempdir()) / "FaceSetCurator" / "logs" / "fsc.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("faceset_curator")
    root.setLevel(logging.INFO)
    resolved = os.path.normcase(str(log_path.resolve()))
    if not any(isinstance(handler, RotatingFileHandler)
               and os.path.normcase(getattr(handler, "baseFilename", "")) == resolved
               for handler in root.handlers):
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        root.addHandler(handler)
    return log_path
