<!-- Copyright © 2026 Hanafi Mohd Radi. All rights reserved. -->

# Implementation plan

## Phase 1 — runnable vertical slice (implemented)

- Project/package structure, configuration profiles, immutable scanner
- SQLite content-addressed analysis cache
- Deterministic baseline measurements and perceptual hashing
- Eligibility gates, exact/near-duplicate categorization
- Diversity-aware selection of up to 100 images
- Categorized copies and auditable JSON/CSV/HTML reporting
- Unit tests for safety-relevant selection behavior
- Indexed duplicate lookup, incremental selection scoring, and batched cache commits for collections up to 15,000 images

## Phase 2 — production CUDA analysis (core backend implemented)

- InsightFace detection and ArcFace identity embeddings through ONNX Runtime CUDA
- Reference-image workflow with multi-reference centroid enrollment
- Model-derived yaw/pitch/roll and face-size/detection diagnostics
- Composite face-image quality, exposure, and sharpness diagnostics
- Expression embedding/classifier and embedding clustering
- Batched GPU inference with CPU decode workers and VRAM-aware batch sizing
- Concurrent fingerprinting, bounded decode prefetch, dynamic detector batches, native recognition batches, and auditable performance settings (implemented)
- Landmark expression classification, expanded quality diagnostics, face-masked scene/outfit descriptors, and scalable diversity clusters (implemented)
- Scene, appearance, pose, expression, and likely-session balancing in collective selection (implemented)

## Phase 3 — desktop review UI

- Folder/reference selection, profile controls, progress and pause/resume
- Reference previews, 2–5 image guidance, pairwise consistency warnings, adjustable identity verification, score distributions, and zero-result diagnostics (implemented)
- Structured stage progress, GPU warm-up, inference throughput, cache/eligibility counters, and inference-only ETA smoothing (implemented)
- In-app diagnostics, explicit CUDA-active state, real provider inference probe, and verified resumable model downloads (implemented)
- Duplicate-group comparison, rejection/score filters, and side-by-side “why rejected” review (implemented)
- Controlled manual include/exclude swaps with safety locks and synchronized categorized output (implemented)
- Recompute dataset-value impact and refresh auditable reports after a manual change (implemented)

## Phase 4 — validation and packaging

- Curated benchmark with labeled identity, duplicate, pose, and quality cases
- Threshold calibration CLI and deterministic synthetic regression fixtures (implemented; curated real benchmark remains)
- Windows standalone packaging, CUDA capability check, model download verification, and installer-time GPU verification (implemented; code signing remains)
- Crash recovery, incomplete-run markers, resumable materialization, disk-space preflight, cooperative startup cancellation, and rotating logs (implemented)
- Signed release artifacts (remaining)

## Acceptance criteria

- No operation modifies a source file.
- Default run proposes at most 100 unique images.
- Strong duplicate mode selects at most one member of each duplicate group.
- Adding a selected image changes marginal scores so redundant images lose value.
- Every file has a category, component scores, and reason in the report.
- Re-running unchanged inputs uses cached analysis.
