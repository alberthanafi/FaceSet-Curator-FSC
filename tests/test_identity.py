# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

import math

from faceset_curator.identity import centroid, confidence, cosine, identity_statistics, reference_consistency
from faceset_curator.models import CuratorConfig, ImageAnalysis


def test_reference_centroid_is_normalized():
    result = centroid([[1.0, 0.0], [0.8, 0.2]])
    assert math.isclose(cosine(result, result), 1.0)
    assert result[0] > result[1]


def test_cosine_maps_to_confidence():
    assert confidence(-1.0) == 0.0
    assert confidence(0.0) == 0.5
    assert confidence(1.0) == 1.0


def test_identity_verification_profiles_resolve_thresholds():
    assert CuratorConfig().identity_threshold == 0.72
    assert CuratorConfig(identity_verification="normal").identity_threshold == 0.65
    assert CuratorConfig(identity_verification="custom", identity_threshold=0.58).identity_threshold == 0.58


def test_reference_consistency_warns_for_disagreement():
    summary = reference_consistency([[1.0, 0.0], [-1.0, 0.0]])
    assert summary["reference_count"] == 2
    assert summary["minimum_pair_score"] == 0.0
    assert "may not show the same person" in summary["warning"]


def test_identity_statistics_reports_distribution():
    items = [ImageAnalysis(path=str(index), fingerprint=str(index), size_bytes=1,
                           face_count=1, identity_score=score)
             for index, score in enumerate((0.49, 0.55, 0.68, 0.75, 0.88))]
    summary = identity_statistics(items, 0.72)
    assert summary["faces_scored"] == 5
    assert summary["passed_threshold"] == 2
    assert summary["median"] == 0.68
    assert summary["buckets"] == {
        "below_0_50": 1,
        "0_50_to_0_60": 1,
        "0_60_to_0_70": 1,
        "0_70_to_0_80": 1,
        "0_80_and_above": 1,
    }
