# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageStat


def _distance(left: list[float], right: list[float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def classify_expression(landmarks: Any, face_width: float, face_height: float) -> tuple[str, float]:
    if landmarks is None or len(landmarks) < 68 or face_width <= 0 or face_height <= 0:
        return "neutral", 0.25
    points = [[float(value) for value in point[:2]] for point in landmarks]
    mouth_width = _distance(points[48], points[54]) / face_width
    mouth_open = _distance(points[62], points[66]) / max(1.0, _distance(points[60], points[64]))
    mouth_center_y = (points[51][1] + points[57][1]) / 2.0
    corner_lift = (mouth_center_y - (points[48][1] + points[54][1]) / 2.0) / face_height

    def eye_ratio(indices: tuple[int, int, int, int, int, int]) -> float:
        p1, p2, p3, p4, p5, p6 = (points[index] for index in indices)
        return (_distance(p2, p6) + _distance(p3, p5)) / max(1.0, 2.0 * _distance(p1, p4))

    eyes = (eye_ratio((36, 37, 38, 39, 40, 41)) +
            eye_ratio((42, 43, 44, 45, 46, 47))) / 2.0
    if eyes < 0.14:
        return "eyes_closed", min(1.0, 0.55 + (0.14 - eyes) * 4.0)
    if mouth_open > 0.22 and eyes > 0.22:
        return "surprised", min(1.0, 0.55 + (mouth_open - 0.22) * 1.8)
    if mouth_width > 0.34 and corner_lift > 0.005:
        return "smiling", min(1.0, 0.55 + (mouth_width - 0.34) * 2.5 + corner_lift * 5.0)
    if mouth_open > 0.12:
        return "mouth_open", min(1.0, 0.5 + (mouth_open - 0.12) * 2.0)
    return "neutral", min(1.0, 0.55 + max(0.0, 0.12 - mouth_open) * 2.0)


def _block_artifact_score(gray: Image.Image) -> float:
    sample = gray.copy()
    sample.thumbnail((256, 256))
    width, height = sample.size
    pixels = list(sample.getdata())
    if width < 16 or height < 16:
        return 0.0
    boundary, interior = [], []
    for y in range(height):
        row = y * width
        for x in range(1, width):
            difference = abs(pixels[row + x] - pixels[row + x - 1])
            (boundary if x % 8 == 0 else interior).append(difference)
    for y in range(1, height):
        for x in range(width):
            difference = abs(pixels[y * width + x] - pixels[(y - 1) * width + x])
            (boundary if y % 8 == 0 else interior).append(difference)
    boundary_mean = sum(boundary) / max(1, len(boundary))
    interior_mean = sum(interior) / max(1, len(interior))
    return max(0.0, min(1.0, (boundary_mean - interior_mean) / max(8.0, interior_mean)))


def _embedding(image: Image.Image, size: tuple[int, int] = (4, 4)) -> list[float]:
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return [round(channel / 255.0, 5) for pixel in resized.getdata() for channel in pixel]


def analyze_face_visuals(image: Image.Image, bbox: list[float], detection: float,
                         keypoints: Any = None, landmarks: Any = None) -> dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    ox1, oy1, ox2, oy2 = bbox
    x1, y1 = max(0, int(ox1)), max(0, int(oy1))
    x2, y2 = min(width, int(ox2)), min(height, int(oy2))
    face_width, face_height = max(1, x2 - x1), max(1, y2 - y1)
    crop = rgb.crop((x1, y1, x2, y2))
    gray = crop.convert("L")
    values = sorted(gray.resize((64, 64), Image.Resampling.BILINEAR).getdata())
    mean = sum(values) / max(1, len(values))
    low, high = values[int(len(values) * .05)], values[int(len(values) * .95)]
    clipping = (sum(value <= 5 or value >= 250 for value in values) / max(1, len(values)))
    midtone = max(0.0, 1.0 - abs(mean - 127.5) / 127.5)
    dynamic_range = min(1.0, (high - low) / 170.0)
    exposure = max(0.0, min(1.0, .55 * midtone + .30 * dynamic_range + .15 * (1.0 - clipping)))
    sharpness = min(1.0, ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0] / 1800.0)
    compression = _block_artifact_score(gray)
    original_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
    visible_area = face_width * face_height / original_area
    points = keypoints if keypoints is not None else []
    visible_points = (sum(x1 <= float(point[0]) <= x2 and y1 <= float(point[1]) <= y2 for point in points) /
                      max(1, len(points)))
    visibility = max(0.0, min(1.0, .45 * visible_area + .30 * visible_points + .25 * detection))
    occlusion = 1.0 - visibility
    face_area = face_width * face_height / max(1, width * height)
    size_score = min(1.0, math.sqrt(face_area) * 3.0)
    quality = (.27 * sharpness + .18 * exposure + .16 * size_score + .12 * detection +
               .14 * (1.0 - occlusion) + .13 * (1.0 - compression))
    expression, expression_confidence = classify_expression(landmarks, face_width, face_height)

    scene = rgb.copy()
    average = tuple(int(value) for value in ImageStat.Stat(rgb.resize((32, 32))).mean[:3])
    ImageDraw.Draw(scene).rectangle((x1, y1, x2, y2), fill=average)
    outfit_top = min(height - 1, y2)
    outfit_bottom = min(height, y2 + max(face_height, 1) * 2)
    outfit_left = max(0, x1 - face_width // 2)
    outfit_right = min(width, x2 + face_width // 2)
    appearance = rgb.crop((outfit_left, outfit_top, max(outfit_left + 1, outfit_right),
                           max(outfit_top + 1, outfit_bottom)))
    flags = []
    if sharpness < .25:
        flags.append("noticeable blur")
    if exposure < .35:
        flags.append("poor exposure or clipped lighting")
    if occlusion > .35:
        flags.append("possible face occlusion or edge clipping")
    if compression > .35:
        flags.append("visible block-compression artifacts")
    return {
        "quality_score": round(quality, 6),
        "sharpness_score": round(sharpness, 6),
        "exposure_score": round(exposure, 6),
        "blur_score": round(1.0 - sharpness, 6),
        "occlusion_score": round(occlusion, 6),
        "compression_artifact_score": round(compression, 6),
        "face_area_ratio": round(face_area, 6),
        "expression": expression,
        "expression_confidence": round(expression_confidence, 6),
        "scene_embedding": _embedding(scene),
        "appearance_embedding": _embedding(appearance),
        "flags": flags,
    }


def cluster_signature(values: list[float], bits: int = 6) -> str:
    if not values:
        return "unknown"
    mean = sum(values) / len(values)
    signature = 0
    for bit in range(bits):
        segment = values[bit::bits]
        if segment and sum(segment) / len(segment) >= mean:
            signature |= 1 << bit
    return f"{signature:0{max(1, (bits + 3) // 4)}x}"


def assign_diversity_clusters(items: list[Any]) -> None:
    for item in items:
        yaw_bucket = int((max(-90.0, min(90.0, item.yaw)) + 90) // 15)
        pitch_bucket = int((max(-60.0, min(60.0, item.pitch)) + 60) // 15)
        item.pose_cluster = f"pose-{yaw_bucket:02d}-{pitch_bucket:02d}"
        item.scene_cluster = f"scene-{cluster_signature(item.scene_embedding)}"
        item.appearance_cluster = f"appearance-{cluster_signature(item.appearance_embedding)}"
        parent = Path(item.path).parent.name.lower()[:24] or "root"
        item.session_cluster = f"{parent}:{item.scene_cluster}:{item.appearance_cluster}"
        item.visual_embedding = list(item.scene_embedding) + list(item.appearance_embedding)
