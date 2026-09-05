<!-- Copyright © 2026 Hanafi Mohd Radi. All rights reserved. -->

# Architecture

## Design goals

FSC targets 1,000–15,000 local images, an RTX 5080 CUDA GPU, and Ryzen 7 5700X3D CPU assistance. The default policy is Full/Balanced, 100 selected images, Strong duplicate filtering, High identity verification, and enabled quality, pose, expression, clustering, diversity balancing, and caching.

## Pipeline

1. **Scanner** validates supported files and computes immutable source fingerprints.
2. **Cache** stores analysis by `(content fingerprint, analyzer version)` in SQLite.
3. **Analyzer backend** validates 2–5 recommended enrollment references, reports pairwise reference consistency, and produces face count, identity confidence, embedding, perceptual hash, quality, pose, expression, and diagnostic flags. The production implementation uses an in-memory dynamic-batch view of the detector model and native batched ArcFace recognition; downloaded ONNX files remain untouched. It is a narrow protocol so CUDA inference and test/baseline implementations are interchangeable.
4. **Eligibility** applies the selected High, Normal, or Custom identity threshold and rejects corrupt/no-face/multiple-face/wrong-identity/low-quality records before optimization. Identity distributions are recorded before selection so zero-result runs can be diagnosed without inference reruns.
5. **Duplicate grouper** uses exact hashes first and an indexed perceptual-hash search for near duplicates. Strong mode retains one representative per duplicate group without an all-pairs scan.
6. **Clusterer** organizes remaining embeddings to expose redundant visual coverage.
7. **Selector** greedily maximizes marginal dataset value, not raw image quality. The marginal score combines individual quality and identity confidence with pose/expression novelty and embedding diversity, while penalizing similarity to already selected images and overrepresented buckets. Nearest-selected similarity is updated incrementally, reducing selection work from repeated rescans of the chosen set to one update per candidate and selected image.
8. **Materializer** groups output beneath the source folder name and copies files into a new timestamped run directory. It never mutates sources.
9. **Reporter** writes JSON (complete audit), CSV (review), and HTML (human summary).

The desktop review layer filters the durable `ImageAnalysis` records without rerunning inference. Manual inclusion is limited to eligible redundant candidates; identity, face-count, quality, invalid-file, and duplicate safety decisions cannot be bypassed. A forced inclusion becomes the first member of a fresh collective optimization, after which marginal values are recalculated and the weakest selected member is replaced. Changed categorized copies are moved within the run, current reports are regenerated, and the prior report revision is retained in `.fsc-review-history`.

Structured progress events carry stage, completed/total work, cache hits, inference throughput, ETA inputs, eligible candidates, and selected counts. GUI ETA smoothing starts with face inference rather than application/model startup.

Model provisioning is performed before InsightFace initialization. The downloader supports HTTP range resume and bounded retries, validates the official archive and extracted model SHA-256 values, installs through a staging directory, and preserves invalid prior files for diagnosis. A verification stamp avoids hashing hundreds of megabytes again unless model file metadata changes. Runtime failures are translated into user-facing guidance while full tracebacks and the confirmed execution provider remain available in the expandable UI diagnostics panel. The doctor command runs a profiled ONNX kernel and confirms its actual execution provider.

## Production backend boundary

`Analyzer.analyze(path, fingerprint)` is the single-image hardware/model seam and the production backend additionally exposes bounded `analyze_many` processing. The CUDA backend uses InsightFace/ONNX Runtime CUDA for detection, recognition embeddings, landmarks/pose, and face quality, with CPU workers for concurrent decode, hashing, metadata, cache I/O, and report generation. A bounded decode queue overlaps CPU preparation with GPU work. Detection and recognition are batched; pose landmark post-processing runs for the largest detected face in each image. VRAM-based automatic sizing chooses conservative batches from 2 to 12 images.

The 68-point landmarks feed a deterministic facial-geometry expression classifier with confidence reporting. Quality diagnostics measure edge sharpness, tonal range and clipping, landmark/bounding-box visibility, JPEG-style block discontinuities, face area, and detection confidence independently. The face region is masked before building the scene descriptor, while a region below and around the face supplies an outfit/appearance descriptor. Compact deterministic signatures form pose, scene, appearance, and likely-session clusters without quadratic comparisons. The selector applies separate novelty counters for these clusters, protecting the final set from repeated outfits, locations, and capture sessions.

Threshold calibration is an offline, reproducible CLI boundary. Labeled identity and optional quality scores are evaluated over every separating threshold using balanced accuracy, then reported with precision, recall, and confusion counts. Calibration results are advisory and do not silently alter the application defaults.

## Data model

An `ImageAnalysis` is the durable unit. It includes source identity, analyzer version, measurements, derived buckets, decision state, component scores, duplicate group, and human-readable reasons. Reports never need to re-run a model.

## Safety and recovery

Runs use timestamped directories. A versioned, atomically written run journal records the exact source, configuration, analyzer version, status, and last durable progress point. Matching incomplete runs are resumed in place; cached analyses and size-verified manifest copies are reused. `INCOMPLETE.txt` remains visible until reports are complete. Copies are atomic at the per-file level, and their manifest is checkpointed every 25 files. A conservative disk preflight reserves categorized-copy space plus report/cache overhead before model startup. The cache is separate from outputs and can be deleted without affecting sources. Rotating local logs retain startup, recovery, completion, and full failure details without placing diagnostics in source folders.
