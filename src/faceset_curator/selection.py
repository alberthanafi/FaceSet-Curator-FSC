# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .models import CuratorConfig, ImageAnalysis


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norm if norm else 0.0


def pose_bucket(item: ImageAnalysis) -> str:
    if item.yaw <= -35: return "left_profile"
    if item.yaw <= -12: return "left_three_quarter"
    if item.yaw >= 35: return "right_profile"
    if item.yaw >= 12: return "right_three_quarter"
    return "frontal"


def eligible(items: list[ImageAnalysis], config: CuratorConfig) -> list[ImageAnalysis]:
    accepted = []
    for item in items:
        if item.category == "invalid" or item.face_count == 0:
            item.category, item.reasons = "rejected/no_face_or_invalid", ["No usable face or unreadable image"]
        elif config.require_single_face and item.face_count > 1:
            item.category, item.reasons = "rejected/multiple_faces", ["More than one face detected"]
        elif item.identity_score < config.identity_threshold:
            item.category, item.reasons = "rejected/identity", ["Identity confidence below High threshold"]
        elif item.quality_score < config.minimum_quality:
            item.category, item.reasons = "rejected/quality", ["Quality below minimum threshold"]
        else:
            accepted.append(item)
    return accepted


def remove_duplicates(items: list[ImageAnalysis], config: CuratorConfig) -> list[ImageAnalysis]:
    representatives: list[ImageAnalysis] = []
    near_threshold = {
        "strong": config.near_duplicate_hamming,
        "normal": min(4, config.near_duplicate_hamming),
        "exact": -1,
    }[config.duplicate_strength]
    exact_index: dict[str, int] = {}
    hash_index: dict[tuple[int, int], list[int]] = defaultdict(list)

    # Splitting a 64-bit perceptual hash into radius+1 blocks gives an exact
    # candidate index: hashes within the radius must share at least one block.
    # This avoids comparing every image with every representative at 15k scale.
    block_count = near_threshold + 1 if near_threshold >= 0 else 0
    block_sizes = ([64 // block_count + (index < 64 % block_count)
                    for index in range(block_count)] if block_count else [])

    def block_keys(value: str) -> list[tuple[int, int]]:
        number = int(value, 16)
        keys = []
        shift = 0
        for index, size in enumerate(block_sizes):
            keys.append((index, (number >> shift) & ((1 << size) - 1)))
            shift += size
        return keys

    for candidate in sorted(items, key=lambda x: (-x.quality_score, x.path)):
        match_index = exact_index.get(candidate.fingerprint)
        if match_index is None and block_count and candidate.phash:
            possible = set()
            for key in block_keys(candidate.phash):
                possible.update(hash_index.get(key, ()))
            match_index = next((index for index in sorted(possible)
                                if representatives[index].phash and
                                hamming(candidate.phash, representatives[index].phash) <= near_threshold), None)
        match = representatives[match_index] if match_index is not None else None
        if match:
            candidate.category = "rejected/duplicate"
            candidate.duplicate_group = match.fingerprint[:12]
            candidate.reasons = [f"Exact or near duplicate of {match.source.name}"]
            match.duplicate_group = match.fingerprint[:12]
        else:
            representative_index = len(representatives)
            representatives.append(candidate)
            exact_index[candidate.fingerprint] = representative_index
            if block_count and candidate.phash:
                for key in block_keys(candidate.phash):
                    hash_index[key].append(representative_index)
    return representatives


def select(items: list[ImageAnalysis], config: CuratorConfig,
           forced: list[ImageAnalysis] | None = None) -> list[ImageAnalysis]:
    weights = {
        "balanced": {"quality": .34, "identity": .16, "embedding_diversity": .16,
                     "pose_novelty": .10, "expression_novelty": .07,
                     "appearance_novelty": .07, "scene_novelty": .05, "session_novelty": .05},
        "quality": {"quality": .54, "identity": .19, "embedding_diversity": .08,
                    "pose_novelty": .05, "expression_novelty": .03,
                    "appearance_novelty": .04, "scene_novelty": .03, "session_novelty": .04},
        "diversity": {"quality": .21, "identity": .13, "embedding_diversity": .20,
                      "pose_novelty": .13, "expression_novelty": .08,
                      "appearance_novelty": .10, "scene_novelty": .08, "session_novelty": .07},
    }[config.profile]
    remaining = list(items)
    chosen: list[ImageAnalysis] = []
    buckets: Counter[str] = Counter()
    expressions: Counter[str] = Counter()
    appearances: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    sessions: Counter[str] = Counter()
    active = [True] * len(remaining)
    normalized_features: list[tuple[float, ...]] = []
    for item in remaining:
        features = item.visual_embedding or item.embedding
        norm = math.sqrt(sum(value * value for value in features))
        normalized_features.append(tuple(value / norm for value in features) if norm else ())
    nearest_similarity = [0.0] * len(remaining)

    def score_components(index: int) -> tuple[float, dict[str, float]]:
        item = remaining[index]
        diversity = 1.0 - nearest_similarity[index]
        pose_key = item.pose_cluster or pose_bucket(item)
        pose_novelty = 1.0 / (1.0 + buckets[pose_key])
        expression_novelty = 1.0 / (1.0 + expressions[item.expression])
        appearance_novelty = 1.0 / (1.0 + appearances[item.appearance_cluster or "unknown"])
        scene_novelty = 1.0 / (1.0 + scenes[item.scene_cluster or "unknown"])
        session_novelty = 1.0 / (1.0 + sessions[item.session_cluster or "unknown"])
        components = {
            "quality": weights["quality"] * item.quality_score,
            "identity": weights["identity"] * item.identity_score,
            "embedding_diversity": weights["embedding_diversity"] * diversity,
            "pose_novelty": weights["pose_novelty"] * pose_novelty,
            "expression_novelty": weights["expression_novelty"] * expression_novelty,
            "appearance_novelty": weights["appearance_novelty"] * appearance_novelty,
            "scene_novelty": weights["scene_novelty"] * scene_novelty,
            "session_novelty": weights["session_novelty"] * session_novelty,
        }
        return sum(components.values()), components

    def choose(index: int, manual: bool = False) -> None:
        best = remaining[index]
        score, components = score_components(index)
        best.category = "selected"
        best.dataset_value = round(score * 100, 3)
        best.score_components = {key: round(value * 100, 3) for key, value in components.items()}
        best.reasons = [
            ("Manually included; collective values recalculated" if manual else
             f"Selected by {config.profile} profile; adds {pose_bucket(best)} / {best.expression} coverage")
        ]
        chosen.append(best)
        active[index] = False
        buckets[best.pose_cluster or pose_bucket(best)] += 1
        expressions[best.expression] += 1
        appearances[best.appearance_cluster or "unknown"] += 1
        scenes[best.scene_cluster or "unknown"] += 1
        sessions[best.session_cluster or "unknown"] += 1
        chosen_features = normalized_features[index]
        if chosen_features:
            for candidate_index, features in enumerate(normalized_features):
                if active[candidate_index] and features:
                    similarity = sum(left * right for left, right in zip(features, chosen_features))
                    nearest_similarity[candidate_index] = max(nearest_similarity[candidate_index], similarity)

    forced_paths = {item.path for item in (forced or [])}
    if len(forced_paths) > config.target_count:
        raise ValueError("More manual inclusions were requested than the target set size")
    for forced_item in forced or []:
        forced_index = next((index for index, item in enumerate(remaining)
                             if active[index] and item.path == forced_item.path), None)
        if forced_index is None:
            raise ValueError(f"Manual inclusion is not an eligible candidate: {forced_item.path}")
        choose(forced_index, manual=True)

    while len(chosen) < min(config.target_count, len(remaining)):
        best_index = None
        best_score = -1.0
        for index, item in enumerate(remaining):
            if not active[index]:
                continue
            score, _components = score_components(index)
            if score > best_score or (score == best_score and
                                      item.path < (remaining[best_index].path if best_index is not None else "")):
                best_index, best_score = index, score
        assert best_index is not None
        choose(best_index)
    for index, item in enumerate(remaining):
        if active[index]:
            item.category = "rejected/redundant"
            item.reasons = ["Eligible, but added less value than the selected set"]
    return chosen
