<!-- Copyright © 2026 Hanafi Mohd Radi. All rights reserved. -->

# FaceSet Curator (FSC)

FaceSet Curator builds the best *collective* face dataset from a collection of approximately 1,000–15,000 images. Its default `balanced` profile selects 100 images while penalizing duplicates and redundant coverage and rewarding useful variation in pose, expression, and visual conditions.

## Safety contract

- Source files are opened read-only and are never renamed, changed, or deleted.
- Every run writes into a new output directory.
- Selected and categorized files are copied, never moved.
- Analysis is cached by file content fingerprint.
- Reports preserve the score components and reason for every decision.

## Current vertical slice

The initial implementation provides a working scanner, SQLite cache, deterministic baseline analyzer, strong exact/near-duplicate grouping, diversity-aware greedy selection, categorized copies, JSON/CSV/HTML reports, CLI, and tests. The baseline analyzer uses image metadata and byte-level fingerprints so the full workflow is runnable before CUDA model packages are installed. The analyzer interface is intentionally replaceable by the production InsightFace/ONNX CUDA backend.

## Run

Requires Python 3.11+. For production face analysis, install the optional GPU stack:

```powershell
python -m pip install -e ".[gpu]"
fsc doctor
fsc curate C:\photos\target --reference C:\references\target-1.jpg --reference C:\references\target-2.jpg --output C:\fsc-runs
```

Use `fsc plan` to print the effective configuration without touching images.

The production command defaults to InsightFace and CUDA-first execution. It deliberately refuses to silently downgrade to the metadata-only baseline. For pipeline demonstrations without face semantics, pass `--backend baseline` explicitly.

The GPU dependency set installs compatible CUDA and cuDNN runtime libraries with ONNX Runtime. FSC registers the pip-installed NVIDIA DLL directories for the lifetime of the Windows process, preloads the libraries, and verifies that every InsightFace model is actually using `CUDAExecutionProvider`; a silent session fallback to CPU is treated as a startup error. Run `fsc doctor` to check runtime availability.
GPU runtime and analyzer revisions are part of the cache key, so results produced by an older or incorrectly loaded backend are not reused after an upgrade.

## Desktop application

Double-click `FaceSet Curator.bat`, or run:

```powershell
python -m pip install -e ".[gpu]"
fsc-gui
```

In the application:

1. Choose the folder containing the 1,000–15,000 source images.
2. Choose a separate output folder.
3. Choose one or more clear reference images containing only the target face.
4. Leave Backend on `insightface`, Device on `auto`, Best images on `100`, and start curation.
5. Review selected and rejected images, compare them side by side, open the HTML report, or inspect categorized output folders.

During analysis, FSC reports model loading, GPU warm-up, scanning, fingerprinting, cache lookup, face analysis, identity filtering, duplicate filtering, collective optimization, categorized copying, report generation, and completion as distinct stages. Face analysis shows smoothed images-per-second throughput, cache hits, eligible-image count, selected count, and an ETA based only on active inference time so startup does not distort the estimate.

Before loading the AI models, FSC estimates output-copy and report space and stops with a clear message when the selected drive is too full. Every active output contains `INCOMPLETE.txt` and a `.fsc-run.json` recovery journal. A later run with the same source, settings, and analyzer version resumes that directory, reuses cached analysis, and skips categorized copies already verified by size. The incomplete marker is removed only after all three reports are complete. Model loading, enrollment, GPU warm-up, analysis, and copying all honor cancellation between safe processing steps.

Operational messages and full failures are written to a rotating log at `%LOCALAPPDATA%\FaceSetCurator\logs\fsc.log` (5 MB per file, three backups). The path is shown in Diagnostics.

The setup screen explicitly reports whether the AI engine is running on CUDA or CPU. A collapsible Diagnostics panel keeps provider details and full startup errors inside the application while error dialogs show a short corrective explanation. FSC validates the official `buffalo_l` model package with SHA-256 checksums, detects missing or partial files, retries interrupted downloads, and resumes from the saved partial archive. `fsc doctor --device cuda` now executes and profiles a real ONNX kernel to verify that CUDA—not merely the CUDA provider registration—is active.

The setup screen continuously displays total CPU load, system RAM use, NVIDIA GPU utilization, GPU memory use, and GPU temperature. Hardware statistics refresh once per second without blocking the interface.

The desktop controls and dropdown lists use a high-contrast dark theme, including read-only selection fields.

The Results tab can filter selected images and every rejection category, including identity, quality, duplicates, redundancy, multiple faces, invalid images, and manual exclusions. Minimum identity and quality filters narrow large result sets. Selecting a duplicate automatically shows its group representative for side-by-side comparison, and every row displays its recorded “why rejected” reason. Eligible redundant images can be manually included and selected images can be excluded; safety rejections such as wrong identity, multiple faces, invalid files, low quality, and duplicates remain locked. Each controlled edit keeps the target size when alternatives exist, recalculates collective dataset values, updates categorized copies and all reports, and preserves the previous report revision under `.fsc-review-history`.

Large-library processing uses indexed perceptual-hash duplicate lookup, incremental diversity scoring, and batched SQLite cache commits. These keep sorting and collective selection responsive without changing the Balanced, Strong, or High defaults.

Production analysis classifies `neutral`, `smiling`, `mouth_open`, `surprised`, and `eyes_closed` expressions from the detected 68-point facial landmarks and records a confidence value. Quality scoring separately measures blur, lighting range and clipping, face visibility/edge clipping, block-compression artifacts, face size, and detector confidence. Face-masked scene descriptors and the region below the face provide separate background and outfit/appearance embeddings. Deterministic pose, expression, appearance, scene, and likely-session clusters prevent one pose, outfit, location, or photo session from dominating the selected set.

For dataset-specific threshold calibration, prepare a labeled CSV with `identity_score,same_identity` columns and optional `quality_score,acceptable_quality` columns, then run:

```powershell
fsc calibrate C:\benchmarks\faces.csv --output C:\benchmarks\calibration.json
```

Labels accept `yes/no`, `true/false`, or `1/0`. FSC reports the deterministic threshold with the best balanced accuracy together with precision, recall, and confusion counts; it never changes the High default automatically.

The InsightFace backend uses concurrent CPU decoding and hashing, a bounded prefetch queue, batched CUDA face detection, and batched recognition. GPU batch size defaults to automatic VRAM-based selection (8 images on a 16 GB RTX 5080); CPU worker count defaults from available logical processors. CLI users can override these with `--gpu-batch-size`, `--cpu-workers`, and `--decode-queue-size`. The effective performance configuration is saved in `report.json`.

The first InsightFace run may initialize model assets before analysis begins. References should cover the same person clearly; two to five varied, high-quality references are preferable.

FSC displays thumbnails for up to five enrollment references and compares every reference pair after face analysis. It warns when only one reference is supplied, when more than five are supplied, or when the references may not show the same identity consistently.

Identity verification defaults to **High** (`0.72`). **Normal** uses `0.65` for broader matching, while **Custom** accepts a threshold from `0.00` to `1.00`. Changing this threshold does not rerun face inference when cached analysis for the same references is available. Results and reports include the minimum, median, maximum, threshold-pass count, and score-distribution buckets. A run with no qualifying images displays the rejection totals and recommended corrective action.

### Adjustable selection controls

- **Balanced** combines quality, identity confidence, pose coverage, expression coverage, and visual diversity. This remains the default.
- **Quality first** gives substantially more weight to individually strong, clean images while retaining modest diversity protection.
- **Diversity first** favors broader pose and visual coverage, allowing somewhat lower-quality images when they add meaningful representation.
- **Strong duplicates** uses the widest perceptual-hash threshold and is the default for dataset creation.
- **Normal duplicates** removes exact matches and closer near-duplicates.
- **Exact only** removes byte-identical images but keeps visually similar frames.

## Standalone Windows build

The PyInstaller definition is in `packaging/`. To produce `dist/FaceSetCurator.exe`:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

The release script compiles the project, runs the dependency-free test suite, verifies real CUDA inference, builds the standalone executable, and runs the executable's own `--doctor` and self-closing `--gui-smoke` checks. It removes InsightFace's overlapping CPU-only ONNX Runtime distribution and reinstalls the GPU distribution last, preventing an apparently successful CPU-only package. If Inno Setup 7 (preferred) or 6 is installed, the script also creates `dist\FaceSetCurator-Setup.exe`; the installer repeats CUDA verification after installation and displays a repair warning if it fails.

## Output

Runs are grouped beneath a folder matching the source folder name. For example, source `C:\photos\target` and output root `C:\fsc-runs` produce `C:\fsc-runs\target\fsc-<timestamp>`. Each timestamped run contains `selected/`, categorized `rejected/` folders, and `report.{html,json,csv}`. The JSON report records configuration and source fingerprints.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Copyright

Copyright © 2026 Hanafi Mohd Radi. All rights reserved. See [COPYRIGHT](COPYRIGHT).

## GitHub distribution

This repository is intended to be publicly visible while remaining proprietary. Public access does
not grant reuse or redistribution rights. Build products and installers are excluded from Git history
and must be published as GitHub Release assets. See
[docs/GITHUB_RELEASE.md](docs/GITHUB_RELEASE.md) for the verified release procedure.
