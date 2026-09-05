# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from PIL import Image

from faceset_curator.models import ImageAnalysis
from faceset_curator.visual_analysis import (analyze_face_visuals, assign_diversity_clusters,
                                             classify_expression, cluster_signature)


def smiling_landmarks():
    points = [[50.0, 50.0] for _ in range(68)]
    points[48], points[54] = [30, 60], [70, 60]
    points[51], points[57] = [50, 68], [50, 72]
    points[60], points[64], points[62], points[66] = [38, 66], [62, 66], [50, 65], [50, 68]
    points[36], points[37], points[38], points[39], points[40], points[41] = (
        [28, 32], [32, 28], [40, 28], [45, 32], [40, 36], [32, 36]
    )
    points[42], points[43], points[44], points[45], points[46], points[47] = (
        [55, 32], [60, 28], [68, 28], [73, 32], [68, 36], [60, 36]
    )
    return points


def test_landmark_expression_classifier_detects_smile():
    expression, confidence = classify_expression(smiling_landmarks(), 100, 100)
    assert expression == "smiling"
    assert confidence >= .55


def test_visual_analysis_measures_quality_and_face_excluded_descriptors():
    image = Image.new("RGB", (160, 200), (40, 90, 150))
    result = analyze_face_visuals(image, [40, 30, 120, 120], .95,
                                  [[55, 55], [105, 55], [80, 75], [60, 95], [100, 95]],
                                  smiling_landmarks())
    assert result["expression"] == "smiling"
    assert len(result["scene_embedding"]) == 48
    assert len(result["appearance_embedding"]) == 48
    assert 0 <= result["quality_score"] <= 1
    assert 0 <= result["occlusion_score"] <= 1
    assert 0 <= result["compression_artifact_score"] <= 1


def test_diversity_clusters_are_deterministic_and_session_aware():
    first = ImageAnalysis("session-a/one.jpg", "a" * 64, 1, yaw=-20, pitch=3,
                          scene_embedding=[.1, .8] * 24, appearance_embedding=[.2, .7] * 24)
    second = ImageAnalysis("session-a/two.jpg", "b" * 64, 1, yaw=-20, pitch=3,
                           scene_embedding=[.1, .8] * 24, appearance_embedding=[.2, .7] * 24)
    assign_diversity_clusters([first, second])
    assert first.pose_cluster == second.pose_cluster
    assert first.scene_cluster == second.scene_cluster
    assert first.appearance_cluster == second.appearance_cluster
    assert first.session_cluster == second.session_cluster
    assert cluster_signature([.1, .8] * 24) == cluster_signature([.1, .8] * 24)
