# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

"""Small dependency-free release test runner used by packaging/build.ps1."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> None:
    failures = []
    count = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f"fsc_tests.{path.stem}", path)
        if spec is None or spec.loader is None:
            failures.append(f"Could not load {path}")
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and not inspect.signature(function).parameters:
                count += 1
                try:
                    function()
                except Exception as exc:
                    failures.append(f"{path.name}::{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"{count} release tests passed")


if __name__ == "__main__":
    main()
