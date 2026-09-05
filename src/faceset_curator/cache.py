# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import ImageAnalysis


class AnalysisCache:
    def __init__(self, path: Path, commit_interval: int = 100):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS analysis (fingerprint TEXT, analyzer TEXT, payload TEXT, PRIMARY KEY(fingerprint, analyzer))"
        )
        self.commit_interval = max(1, commit_interval)
        self.pending_writes = 0

    def get(self, fingerprint: str, analyzer: str) -> ImageAnalysis | None:
        row = self.connection.execute(
            "SELECT payload FROM analysis WHERE fingerprint=? AND analyzer=?", (fingerprint, analyzer)
        ).fetchone()
        return ImageAnalysis(**json.loads(row[0])) if row else None

    def put(self, item: ImageAnalysis, analyzer: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO analysis VALUES (?, ?, ?)",
            (item.fingerprint, analyzer, json.dumps(item.to_dict(), separators=(",", ":"))),
        )
        self.pending_writes += 1
        if self.pending_writes >= self.commit_interval:
            self.connection.commit()
            self.pending_writes = 0

    def close(self) -> None:
        if self.pending_writes:
            self.connection.commit()
        self.connection.close()
