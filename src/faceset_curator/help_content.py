# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations


HELP_TOPICS = {
    "overview": (
        "What is FaceSort?",
        "FaceSort by HMR Studio builds a strong collective set of target-face images. "
        "It considers identity accuracy, image quality, pose, expression, uniqueness, "
        "appearance, scene, and diversity instead of simply keeping the highest individual scores.",
        (
            ("Designed for local work", "FaceSort analyzes images on this Windows computer and keeps the source collection unchanged."),
            ("Recommended defaults", "Balanced profile, Strong duplicate filtering, High identity verification, 100 best images, CUDA device, and analysis caching."),
            ("Library size", "The optimized workflow is intended for approximately 1,000–15,000 source images."),
        ),
    ),
    "quick_start": (
        "Quick start",
        "Choose the source collection, a separate output location, and clear reference images before starting curation.",
        (
            ("1. Source images", "Choose the folder containing the images to analyze. FaceSort scans supported image files recursively."),
            ("2. Output location", "Choose a folder outside the source tree. FaceSort creates a subfolder matching the source folder name and a timestamped run folder."),
            ("3. Target references", "Use 2–5 clear, varied photos that each contain exactly one face of the target person."),
            ("4. Start curation", "Keep the recommended defaults unless the collection needs a different quality/diversity balance."),
        ),
    ),
    "references": (
        "Target references",
        "Reference images establish the identity FaceSort should keep. Multiple varied references are more reliable than one image.",
        (
            ("Good references", "Use sharp, well-lit, unobstructed faces from different angles or sessions."),
            ("Avoid", "Group photos, tiny faces, heavy filters, strong occlusion, extreme blur, and images of different people."),
            ("Enrollment", "FaceSort validates and combines the accepted reference embeddings before analyzing the source collection."),
        ),
    ),
    "profiles": (
        "Selection profiles",
        "Profiles change how the final set balances individual image strength against collective coverage.",
        (
            ("Balanced — default", "Combines quality, identity confidence, pose, expression, appearance, scene, uniqueness, and diversity."),
            ("Quality first", "Favors the cleanest individual images while retaining modest diversity protection."),
            ("Diversity first", "Favors broader pose and visual coverage, accepting somewhat lower quality when it adds useful representation."),
        ),
    ),
    "identity": (
        "Identity verification",
        "Identity filtering compares each detected face with the enrolled target references.",
        (
            ("High — default", "Uses a strict 0.72 threshold for stronger identity precision."),
            ("Normal", "Uses 0.65 for broader matching when references or source images are more varied."),
            ("Custom", "Allows a threshold from 0.00 to 1.00. Review identity distributions before lowering it."),
        ),
    ),
    "duplicates": (
        "Duplicate filtering",
        "Duplicate filtering prevents repeated or nearly identical frames from dominating the selected set.",
        (
            ("Strong — default", "Uses the widest perceptual-similarity threshold."),
            ("Normal", "Removes exact files and closer near-duplicates."),
            ("Exact only", "Removes byte-identical files but keeps visually similar images."),
        ),
    ),
    "performance": (
        "Performance and large libraries",
        "FaceSort is tuned for large image libraries and uses CUDA on the NVIDIA GPU for face inference while the CPU assists with scanning, decoding, and hashing.",
        (
            ("Live status", "Progress stages, throughput, cache hits, selected/eligible counts, ETA, CPU, RAM, GPU, VRAM, and temperature update during a run."),
            ("Caching", "Completed analysis is cached. Compatible reruns and resumed jobs avoid repeating valid work."),
            ("15,000 images", "Indexed duplicate lookup, incremental diversity scoring, bounded decode queues, and batched cache writes control time and memory use."),
        ),
    ),
    "results": (
        "Results and manual review",
        "The Results tab shows selected images and every rejection category with the recorded reason for each decision.",
        (
            ("Filter", "Filter by category, minimum identity score, and minimum quality score."),
            ("Compare", "Review images side by side. Duplicate entries automatically show their group representative when available."),
            ("Adjust", "Eligible redundant images can be included and selected images can be excluded. FaceSort recalculates the collective set and reports."),
        ),
    ),
    "reports": (
        "Output and reports",
        "Each completed run provides categorized copies and auditable JSON, CSV, and HTML reports.",
        (
            ("Selected", "Contains the final collective set."),
            ("Rejected", "Separates identity, quality, duplicate, redundancy, multiple-face, invalid, and manual-review outcomes."),
            ("Audit trail", "Reports record configuration, scores, categories, reasons, source fingerprints, identity statistics, and selection data."),
        ),
    ),
    "troubleshooting": (
        "Troubleshooting",
        "Use the Diagnostics panel on Setup & Run when models, references, CUDA, or a curation stage cannot start.",
        (
            ("CUDA unavailable", "FaceSort stops instead of silently using CPU. Confirm a compatible NVIDIA driver and run the packaged doctor check."),
            ("Model download interrupted", "Start again. FaceSort preserves the partial archive and resumes the download when possible."),
            ("No images selected", "Review rejection totals, use clearer references, or cautiously choose Normal identity verification."),
            ("Log file", "Detailed operational messages are stored under %LOCALAPPDATA%\\FaceSetCurator\\logs\\fsc.log."),
        ),
    ),
    "privacy": (
        "Privacy and source safety",
        "FaceSort is a local application. Its curation pipeline does not upload source or reference images.",
        (
            ("Originals", "Source images are never modified, moved, renamed, or deleted."),
            ("Separate output", "The output must be outside the source tree. Selected and categorized rejected files are copied."),
            ("Recovery", "An incomplete marker and recovery journal protect interrupted runs; the marker is removed only after all reports finish."),
        ),
    ),
    "about": (
        "About FaceSort",
        "FaceSort by HMR Studio sorts face images and selects a diverse, high-quality set locally on Windows.",
        (
            ("Technology", "Python desktop interface, InsightFace, ONNX Runtime CUDA, Pillow, and OpenCV."),
            ("Project", "FaceSet-Curator-FSC on GitHub."),
            ("Rights", "This project is publicly visible but proprietary. Public access does not grant reuse or redistribution rights."),
        ),
    ),
}


DOCUMENTATION_TOPICS = tuple(key for key in HELP_TOPICS if key != "about")


def matching_help_topics(query: str) -> tuple[str, ...]:
    words = tuple(word.casefold() for word in query.split() if word.strip())
    if not words:
        return tuple(HELP_TOPICS)
    matches = []
    for key, (title, summary, sections) in HELP_TOPICS.items():
        text = " ".join((title, summary, *(part for section in sections for part in section))).casefold()
        if all(word in text for word in words):
            matches.append(key)
    return tuple(matches)
