# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

import tempfile
from pathlib import Path

from faceset_curator.calibration import calibrate_csv, calibrate_threshold


def test_calibration_finds_separating_threshold():
    result = calibrate_threshold([(.92, True), (.81, True), (.74, True),
                                  (.61, False), (.45, False), (.2, False)])
    assert .61 < result["threshold"] < .74
    assert result["balanced_accuracy"] == 1.0
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0


def test_calibration_csv_handles_identity_and_quality_labels():
    with tempfile.TemporaryDirectory(prefix="fsc-calibration-test-") as temporary:
        path = Path(temporary) / "benchmark.csv"
        path.write_text(
            "identity_score,same_identity,quality_score,acceptable_quality\n"
            "0.90,yes,0.85,yes\n0.78,true,0.72,true\n"
            "0.55,no,0.41,no\n0.30,false,0.20,false\n",
            encoding="utf-8",
        )
        result = calibrate_csv(path)
        assert result["identity"]["balanced_accuracy"] == 1.0
        assert result["quality"]["balanced_accuracy"] == 1.0
