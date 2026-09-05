# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .models import CuratorConfig
from .app_logging import configure_logging
from .pipeline import curate
from .preflight import check_disk_space

LOGGER = logging.getLogger(__name__)


def _backend(args):
    if args.backend == "baseline":
        from .analyzer import BaselineAnalyzer
        return BaselineAnalyzer()
    from .cuda_analyzer import BackendUnavailable, InsightFaceAnalyzer
    try:
        return InsightFaceAnalyzer(args.reference, provider=args.device,
                                   batch_size=args.gpu_batch_size, cpu_workers=args.cpu_workers)
    except (BackendUnavailable, ValueError) as exc:
        raise SystemExit(f"AI backend could not start: {exc}") from exc


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="fsc", description="FaceSet Curator")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Show default Full/Balanced policy")
    run = commands.add_parser("curate", help="Analyze and curate a source directory")
    run.add_argument("source", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--count", type=int, default=100)
    run.add_argument("--profile", choices=("balanced", "quality", "diversity"), default="balanced")
    run.add_argument("--duplicates", choices=("strong", "normal", "exact"), default="strong")
    run.add_argument("--identity-verification", "--identity", choices=("high", "normal", "custom"),
                     default="high", help="Identity filtering strictness")
    run.add_argument("--identity-threshold", type=float,
                     help="Custom confidence threshold from 0 to 1; implies custom verification")
    run.add_argument("--no-copy-rejected", action="store_true")
    run.add_argument("--reference", type=Path, action="append", default=[], help="Clear single-face image of the target; repeat for stronger enrollment")
    run.add_argument("--backend", choices=("insightface", "baseline"), default="insightface")
    run.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    run.add_argument("--gpu-batch-size", type=int, default=0, help="GPU batch size; 0 selects from VRAM")
    run.add_argument("--cpu-workers", type=int, default=0, help="Decode/hash workers; 0 selects from CPU count")
    run.add_argument("--decode-queue-size", type=int, default=32, help="Maximum prefetched image decodes")
    doctor = commands.add_parser("doctor", help="Check the local AI runtime")
    doctor.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    calibrate = commands.add_parser("calibrate", help="Calibrate thresholds from a labeled benchmark CSV")
    calibrate.add_argument("benchmark", type=Path)
    calibrate.add_argument("--output", type=Path, help="Optional JSON result file")
    return root


def doctor_report(device: str) -> dict:
    import onnxruntime as ort
    from .cuda_analyzer import BackendUnavailable, inference_probe, prepare_cuda_runtime
    from .model_download import validate_buffalo_l

    providers = ort.get_available_providers()
    requested: list[str] = []
    probe = {"passed": False, "requested_provider": None,
             "active_provider": None, "kernel_providers": []}
    error = None
    try:
        requested = prepare_cuda_runtime(ort, device)
        probe = inference_probe(ort, requested)
    except BackendUnavailable as exc:
        error = str(exc)
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    model_ready, model_status = validate_buffalo_l(model_dir, full=True)
    cuda_ready = probe.get("active_provider") == "CUDAExecutionProvider"
    return {
        "onnxruntime": ort.__version__,
        "available_providers": providers,
        "requested_providers": requested,
        "inference_test": probe,
        "cuda_ready": cuda_ready,
        "model_ready": model_ready,
        "model_directory": str(model_dir),
        "model_status": model_status,
        "error": error,
    }


def main() -> None:
    log_path = configure_logging()
    LOGGER.info("CLI started; log=%s", log_path)
    args = parser().parse_args()
    identity_verification = getattr(args, "identity_verification", "high")
    identity_threshold = getattr(args, "identity_threshold", None)
    if identity_threshold is not None:
        identity_verification = "custom"
    elif identity_verification == "custom":
        raise SystemExit("--identity-verification custom requires --identity-threshold")
    config = CuratorConfig(target_count=getattr(args, "count", 100),
                           profile=getattr(args, "profile", "balanced"),
                           duplicate_strength=getattr(args, "duplicates", "strong"),
                           identity_verification=identity_verification,
                           identity_threshold=identity_threshold,
                           gpu_batch_size=getattr(args, "gpu_batch_size", 0),
                           cpu_workers=getattr(args, "cpu_workers", 0),
                           decode_queue_size=getattr(args, "decode_queue_size", 32),
                           copy_rejected=not getattr(args, "no_copy_rejected", False))
    if args.command == "plan":
        print(json.dumps(asdict(config), indent=2))
        return
    if args.command == "doctor":
        try:
            report = doctor_report(args.device)
            print(json.dumps(report, indent=2))
            if not report["inference_test"]["passed"] or (
                    args.device != "cpu" and not report["cuda_ready"]):
                raise SystemExit(1)
        except ImportError:
            raise SystemExit("ONNX Runtime is not installed. Install the optional GPU analysis dependencies first.")
        return
    if args.command == "calibrate":
        from .calibration import calibrate_csv
        try:
            result = calibrate_csv(args.benchmark)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Calibration failed: {exc}") from exc
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return
    if args.source.resolve() == args.output.resolve() or args.output.resolve().is_relative_to(args.source.resolve()):
        raise SystemExit("Output must be outside the source tree to preserve immutable inputs.")
    try:
        check_disk_space(args.source, args.output, config)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Output preflight failed: {exc}") from exc
    run_dir, items = curate(args.source, args.output, config, analyzer=_backend(args))
    print(f"Completed: {sum(x.category == 'selected' for x in items)} selected from {len(items)}")
    print(run_dir)


if __name__ == "__main__":
    main()
