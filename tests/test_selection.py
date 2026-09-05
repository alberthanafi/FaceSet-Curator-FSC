# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from faceset_curator.models import CuratorConfig, ImageAnalysis
from faceset_curator.selection import remove_duplicates, select


def item(name, quality, phash, embedding, yaw=0):
    return ImageAnalysis(path=name, fingerprint=name * 8, size_bytes=1, quality_score=quality,
                         phash=phash, embedding=embedding, yaw=yaw)


def test_strong_duplicates_keep_best_quality():
    low = item("a", .6, "0000000000000000", [1, 0])
    high = item("b", .9, "0000000000000001", [1, 0])
    kept = remove_duplicates([low, high], CuratorConfig())
    assert kept == [high]
    assert low.category == "rejected/duplicate"


def test_selection_rewards_collective_diversity():
    frontal = item("a", .95, "0000000000000000", [1, 0], 0)
    redundant = item("b", .94, "ffff000000000000", [1, 0], 0)
    profile = item("c", .80, "ffffffffffffffff", [0, 1], 50)
    chosen = select([frontal, redundant, profile], CuratorConfig(target_count=2))
    assert frontal in chosen
    assert profile in chosen
    assert redundant.category == "rejected/redundant"


def test_exact_only_does_not_remove_near_duplicate():
    first = item("a", .9, "0000000000000000", [1, 0])
    second = item("b", .8, "0000000000000001", [0, 1])
    kept = remove_duplicates([first, second], CuratorConfig(duplicate_strength="exact"))
    assert kept == [first, second]


def test_quality_profile_prioritizes_image_quality():
    sharp = item("a", 1.0, "0000000000000000", [1, 0])
    varied = item("b", .1, "ffffffffffffffff", [0, 1], 50)
    chosen = select([sharp, varied], CuratorConfig(target_count=1, profile="quality"))
    assert chosen == [sharp]


def test_near_duplicate_index_finds_hashes_within_strong_radius():
    best = item("best", .9, "0123456789abcdef", [1, 0])
    nearby = item("nearby", .8, "0123456789abcdee", [0, 1])
    distant = item("distant", .7, "fedcba9876543210", [0, 1])
    kept = remove_duplicates([distant, nearby, best], CuratorConfig())
    assert kept == [best, distant]
    assert nearby.category == "rejected/duplicate"


def test_selection_handles_15000_candidates():
    items = [item(f"image-{index:05d}", .5 + (index % 50) / 100,
                  f"{index:016x}", [float(index % 7), 1.0], yaw=(index % 91) - 45)
             for index in range(15_000)]
    chosen = select(items, CuratorConfig(target_count=100))
    assert len(chosen) == 100
    assert all(value.category == "selected" for value in chosen)
    assert sum(value.category == "rejected/redundant" for value in items) == 14_900


def test_selection_limits_repeated_scene_outfit_and_session():
    first = item("a", .95, "0000000000000000", [1, 0])
    repeated = item("b", .94, "1111111111111111", [1, 0])
    varied = item("c", .88, "2222222222222222", [1, 0])
    for value in (first, repeated, varied):
        value.visual_embedding = [1, 0]
        value.pose_cluster = "frontal"
        value.expression = "neutral"
    first.scene_cluster = repeated.scene_cluster = "scene-a"
    first.appearance_cluster = repeated.appearance_cluster = "outfit-a"
    first.session_cluster = repeated.session_cluster = "session-a"
    varied.scene_cluster = "scene-b"
    varied.appearance_cluster = "outfit-b"
    varied.session_cluster = "session-b"
    chosen = select([first, repeated, varied], CuratorConfig(target_count=2))
    assert first in chosen
    assert varied in chosen
    assert repeated.category == "rejected/redundant"
