# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CuratorConfig

STATE_NAME = ".fsc-run.json"
INCOMPLETE_NAME = "INCOMPLETE.txt"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunJournal:
    def __init__(self, run_dir: Path, source: Path, config: CuratorConfig,
                 analyzer_version: str, resumed: bool = False) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / STATE_NAME
        self.marker_path = run_dir / INCOMPLETE_NAME
        if resumed:
            self.data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.data["resumed"] = int(self.data.get("resumed", 0)) + 1
            self.data["status"] = "running"
            self.data["error"] = None
        else:
            self.data = {
                "version": 1, "status": "running", "source": str(source.resolve()),
                "config": asdict(config), "analyzer_version": analyzer_version,
                "created_at": _now(), "updated_at": _now(), "stage": "starting",
                "current": 0, "total": 0, "resumed": 0, "error": None,
            }
        self.marker_path.write_text(
            "This FaceSet Curator run is incomplete and is safe to resume.\n"
            "Source images were not changed.\n", encoding="utf-8"
        )
        self._write()

    @classmethod
    def find(cls, source_output: Path, source: Path, config: CuratorConfig,
             analyzer_version: str) -> Path | None:
        expected_source = str(source.resolve())
        expected_config = asdict(config)
        for state_path in sorted(source_output.glob(f"fsc-*/{STATE_NAME}"), reverse=True):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (data.get("status") != "complete" and data.get("source") == expected_source
                    and data.get("config") == expected_config
                    and data.get("analyzer_version") == analyzer_version):
                return state_path.parent
        return None

    def update(self, details: dict[str, Any]) -> None:
        self.data.update({key: details[key] for key in ("stage", "current", "total", "message")
                          if key in details})
        self.data["updated_at"] = _now()
        current = int(details.get("current", 0) or 0)
        total = int(details.get("total", 0) or 0)
        if current in (0, total) or current % 100 == 0:
            self._write()

    def fail(self, error: BaseException) -> None:
        is_cancelled = isinstance(error, InterruptedError) or error.__class__.__name__ == "CurationCancelled"
        self.data.update(status="cancelled" if is_cancelled else "failed",
                         error=f"{type(error).__name__}: {error}", updated_at=_now())
        self._write()

    def complete(self) -> None:
        self.data.update(status="complete", stage="complete", error=None, updated_at=_now())
        self._write()
        self.marker_path.unlink(missing_ok=True)

    def _write(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)
