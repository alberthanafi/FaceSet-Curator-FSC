# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from .analyzer import BaselineAnalyzer
from .identity import centroid, confidence, cosine, reference_consistency
from .model_download import ModelDownloadError, ensure_buffalo_l
from .models import ImageAnalysis
from .scanner import automatic_cpu_workers, fingerprint
from .visual_analysis import analyze_face_visuals


class BackendUnavailable(RuntimeError):
    pass


_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORIES: set[str] = set()


def batch_size_for_vram(total_mib: int) -> int:
    if total_mib < 8 * 1024:
        return 2
    if total_mib < 12 * 1024:
        return 4
    if total_mib < 20 * 1024:
        return 8
    return 12


def automatic_gpu_batch_size() -> int:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return 4
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return batch_size_for_vram(int(completed.stdout.splitlines()[0].strip()))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 4


def _register_nvidia_dll_directories() -> list[str]:
    """Keep pip-installed NVIDIA DLL directories active for this process.

    cuDNN loads several sublibraries lazily during the first convolution. Merely
    preloading cudnn64_9.dll is insufficient on Windows if those sibling DLLs
    are outside the normal search path.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    try:
        import nvidia  # type: ignore
    except ImportError:
        return []
    discovered = []
    for package_root in nvidia.__path__:
        for bin_dir in sorted(Path(package_root).glob("*/bin")):
            resolved = str(bin_dir.resolve())
            if resolved in _DLL_DIRECTORIES or not bin_dir.is_dir():
                continue
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(resolved))
            _DLL_DIRECTORIES.add(resolved)
            discovered.append(resolved)
    if discovered:
        os.environ["PATH"] = os.pathsep.join(discovered + [os.environ.get("PATH", "")])
    return sorted(_DLL_DIRECTORIES)


def prepare_cuda_runtime(ort: Any, provider: str = "auto") -> list[str]:
    """Preload CUDA dependencies and return the requested provider order.

    Listing CUDAExecutionProvider only proves that the provider plugin is
    installed. Its dependent CUDA/cuDNN DLLs may still be missing on Windows,
    in which case ONNX Runtime otherwise falls back to CPU without failing.
    """
    available = ort.get_available_providers()
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in available:
        raise BackendUnavailable(f"CUDA is unavailable; ONNX providers: {available}")
    _register_nvidia_dll_directories()
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        raise BackendUnavailable(
            f"ONNX Runtime {ort.__version__} cannot preload CUDA dependencies. "
            "Install onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27."
        )
    try:
        preload(directory="")
    except Exception as exc:
        raise BackendUnavailable(
            "CUDA/cuDNN runtime DLLs could not be loaded. Reinstall the FSC GPU dependencies "
            "with onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27."
        ) from exc
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def inference_probe(ort: Any, providers: list[str]) -> dict[str, Any]:
    """Run a real ONNX kernel and confirm which execution provider handled it."""
    try:
        import numpy as np  # type: ignore
        import onnx  # type: ignore
        from onnx import TensorProto, helper  # type: ignore
    except ImportError as exc:
        raise BackendUnavailable("The CUDA inference check requires numpy and onnx.") from exc
    node = helper.make_node("Relu", ["input"], ["output"])
    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    model = helper.make_model(helper.make_graph([node], "fsc-provider-probe", [value], [output]),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    options = ort.SessionOptions()
    options.enable_profiling = True
    profile_prefix = str(Path(tempfile.gettempdir()) / f"fsc-ort-probe-{os.getpid()}")
    options.profile_file_prefix = profile_prefix
    profile_path: Path | None = None
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=providers)
        result = session.run(None, {"input": np.asarray([[-1.0, 0.0, 2.0, 3.0]], dtype=np.float32)})[0]
        if not np.allclose(result, [[0.0, 0.0, 2.0, 3.0]]):
            raise BackendUnavailable("ONNX inference returned an unexpected result.")
        profile_path = Path(session.end_profiling())
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        kernel_providers = sorted({event.get("args", {}).get("provider") for event in profile
                                   if event.get("args", {}).get("provider")})
        requested = providers[0]
        if requested not in kernel_providers:
            raise BackendUnavailable(
                f"The inference probe requested {requested}, but kernels ran on "
                f"{', '.join(kernel_providers) if kernel_providers else 'an unknown provider'}."
            )
        return {"passed": True, "requested_provider": requested,
                "active_provider": requested, "kernel_providers": kernel_providers}
    except BackendUnavailable:
        raise
    except Exception as exc:
        raise BackendUnavailable(f"Real ONNX inference check failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if profile_path and profile_path.exists():
            profile_path.unlink()


class InsightFaceAnalyzer:
    """InsightFace/ONNX Runtime analyzer with CUDA-first provider selection.

    Imports are lazy so the baseline workflow does not require the AI stack.
    """

    def __init__(self, references: list[Path], provider: str = "auto", model_name: str = "buffalo_l",
                 batch_size: int = 0, cpu_workers: int = 0,
                 status_callback: Callable[[dict[str, Any]], None] | None = None,
                 cancelled: Callable[[], bool] | None = None):
        def check_cancelled(stage: str) -> None:
            if cancelled and cancelled():
                raise InterruptedError(f"Curation cancelled during {stage}")

        if not references:
            raise ValueError("High identity verification requires at least one reference image")
        check_cancelled("model loading")
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            import onnxruntime as ort  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise BackendUnavailable(
                "InsightFace backend needs insightface, onnxruntime-gpu, numpy, and opencv-python-headless"
            ) from exc
        if model_name == "buffalo_l":
            try:
                ensure_buffalo_l(progress=status_callback, cancelled=cancelled)
            except ModelDownloadError as exc:
                if cancelled and cancelled():
                    raise InterruptedError("Curation cancelled during model download") from exc
                raise BackendUnavailable(str(exc)) from exc
        check_cancelled("model verification")
        self.cv2, self.np = cv2, np
        providers = prepare_cuda_runtime(ort, provider)
        self.providers = providers
        check_cancelled("ONNX model initialization")
        self.app = FaceAnalysis(name=model_name, providers=providers)
        check_cancelled("ONNX model initialization")
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        check_cancelled("ONNX model preparation")
        if providers[0] == "CUDAExecutionProvider":
            cpu_only = [name for name, model in self.app.models.items()
                        if "CUDAExecutionProvider" not in model.session.get_providers()]
            if cpu_only:
                raise BackendUnavailable(
                    "CUDA initialization failed and ONNX Runtime fell back to CPU for: "
                    f"{', '.join(cpu_only)}. Reinstall the FSC GPU dependencies and run 'fsc doctor'."
                )
        self.default_batch_size = (batch_size or automatic_gpu_batch_size()) if providers[0] == "CUDAExecutionProvider" else 1
        self.default_cpu_workers = cpu_workers or automatic_cpu_workers()
        self._enable_batched_detection(ort)
        check_cancelled("batched model preparation")
        self.performance_summary = {
            "gpu_batch_size": self.default_batch_size,
            "cpu_decode_workers": self.default_cpu_workers,
            "decode_queue_size": 32,
            "batched_detection": True,
            "batched_recognition": True,
            "provider": providers[0],
            "warmed_up": False,
        }
        self._warmed_up = False
        reference_embeddings = []
        for path in references:
            check_cancelled("reference enrollment")
            reference_embeddings.append(self._reference_embedding(path))
        check_cancelled("reference enrollment")
        self.enrollment_summary = reference_consistency(reference_embeddings, [path.name for path in references])
        self.target = centroid(reference_embeddings)
        reference_key = "".join(fingerprint(path) for path in references)
        self.version = (f"insightface-{model_name}-v5-{providers[0]}-ort{ort.__version__}-"
                        f"{hashlib.sha256(reference_key.encode()).hexdigest()[:12]}")

    def _enable_batched_detection(self, ort: Any) -> None:
        try:
            import onnx  # type: ignore
        except ImportError as exc:
            raise BackendUnavailable("Batched detection requires the onnx package") from exc
        detector = self.app.det_model
        model = onnx.load(detector.model_file)
        batch_dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
        batch_dimension.ClearField("dim_value")
        batch_dimension.dim_param = "batch"
        for output in model.graph.output:
            output_dimension = output.type.tensor_type.shape.dim[0]
            output_dimension.ClearField("dim_value")
            output_dimension.dim_param = "batch_anchors"
        detector.session = ort.InferenceSession(model.SerializeToString(), providers=self.providers)
        detector.input_name = detector.session.get_inputs()[0].name
        detector.output_names = [output.name for output in detector.session.get_outputs()]

    def _read(self, path: Path) -> Any:
        data = self.np.fromfile(str(path), dtype=self.np.uint8)
        image = self.cv2.imdecode(data, self.cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoder returned no pixels")
        return image

    def _reference_embedding(self, path: Path) -> list[float]:
        faces = self.app.get(self._read(path))
        if len(faces) != 1:
            raise ValueError(f"Reference must contain exactly one face: {path} (found {len(faces)})")
        return [float(value) for value in faces[0].normed_embedding]

    def _decode(self, entry: tuple[Path, str]) -> tuple[Path, str, Any | None, ImageAnalysis | None]:
        path, digest = entry
        try:
            return path, digest, self._read(path), None
        except Exception as exc:
            return path, digest, None, ImageAnalysis(
                path=str(path), fingerprint=digest, size_bytes=path.stat().st_size,
                face_count=0, flags=[f"analysis failed: {type(exc).__name__}: {exc}"], category="invalid"
            )

    def warm_up(self, cancelled: Callable[[], bool] | None = None) -> None:
        if self._warmed_up or self.providers[0] != "CUDAExecutionProvider":
            return
        if cancelled and cancelled():
            raise InterruptedError("Curation cancelled during GPU warm-up")
        blank_detection = self.np.zeros((640, 640, 3), dtype=self.np.uint8)
        self._detect_batch([blank_detection] * self.default_batch_size)
        if cancelled and cancelled():
            raise InterruptedError("Curation cancelled during GPU warm-up")
        recognition = self.app.models["recognition"]
        blank_face = self.np.zeros((recognition.input_size[1], recognition.input_size[0], 3),
                                   dtype=self.np.uint8)
        recognition.get_feat([blank_face] * self.default_batch_size)
        if cancelled and cancelled():
            raise InterruptedError("Curation cancelled during GPU warm-up")
        self._warmed_up = True
        self.performance_summary["warmed_up"] = True

    def _detect_batch(self, images: list[Any]) -> list[list[Any]]:
        from insightface.app.common import Face  # type: ignore
        from insightface.model_zoo.scrfd import distance2bbox, distance2kps  # type: ignore

        detector = self.app.det_model
        input_size = detector.input_size
        prepared, scales = [], []
        for image in images:
            image_ratio = float(image.shape[0]) / image.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if image_ratio > model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / image_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * image_ratio)
            scales.append(float(new_height) / image.shape[0])
            resized = self.cv2.resize(image, (new_width, new_height))
            canvas = self.np.zeros((input_size[1], input_size[0], 3), dtype=self.np.uint8)
            canvas[:new_height, :new_width] = resized
            prepared.append(canvas)
        blob = self.cv2.dnn.blobFromImages(
            prepared, 1.0 / detector.input_std, input_size,
            (detector.input_mean, detector.input_mean, detector.input_mean), swapRB=True,
        )
        outputs = detector.session.run(detector.output_names, {detector.input_name: blob})
        batch_count = len(images)
        results = []
        for batch_index in range(batch_count):
            score_parts, box_parts, keypoint_parts = [], [], []
            for output_index, stride in enumerate(detector._feat_stride_fpn):
                height, width = input_size[1] // stride, input_size[0] // stride
                anchor_centers = self.np.stack(self.np.mgrid[:height, :width][::-1], axis=-1).astype(self.np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if detector._num_anchors > 1:
                    anchor_centers = self.np.stack([anchor_centers] * detector._num_anchors, axis=1).reshape((-1, 2))
                # SCRFD exports H,W,N,C before flattening. Batch entries are
                # interleaved inside each spatial location and before anchors.
                spatial = height * width
                anchors = detector._num_anchors
                scores = outputs[output_index].reshape(spatial, batch_count, anchors, 1)[:, batch_index].reshape(-1)
                boxes = outputs[output_index + detector.fmc].reshape(
                    spatial, batch_count, anchors, 4
                )[:, batch_index].reshape(-1, 4) * stride
                positive = self.np.where(scores >= detector.det_thresh)[0]
                if not positive.size:
                    continue
                score_parts.append(scores[positive, None])
                box_parts.append(distance2bbox(anchor_centers, boxes)[positive] / scales[batch_index])
                if detector.use_kps:
                    keypoints = outputs[output_index + detector.fmc * 2].reshape(
                        spatial, batch_count, anchors, 10
                    )[:, batch_index].reshape(-1, 10) * stride
                    keypoints = distance2kps(anchor_centers, keypoints).reshape((-1, 5, 2))
                    keypoint_parts.append(keypoints[positive] / scales[batch_index])
            if not score_parts:
                results.append([])
                continue
            scores = self.np.vstack(score_parts)
            boxes = self.np.vstack(box_parts)
            order = scores.reshape(-1).argsort()[::-1]
            detections = self.np.hstack((boxes, scores)).astype(self.np.float32, copy=False)[order]
            keypoints = self.np.vstack(keypoint_parts)[order] if detector.use_kps else None
            keep = detector.nms(detections)
            detections = detections[keep]
            keypoints = keypoints[keep] if keypoints is not None else None
            results.append([Face(bbox=row[:4], det_score=row[4],
                                      kps=keypoints[index] if keypoints is not None else None)
                            for index, row in enumerate(detections)])
        return results

    def _enrich_largest_faces(self, images: list[Any], faces_by_image: list[list[Any]]) -> None:
        from insightface.utils import face_align  # type: ignore

        selected = []
        for image_index, faces in enumerate(faces_by_image):
            if not faces:
                continue
            face = max(faces, key=lambda value: float((value.bbox[2] - value.bbox[0]) *
                                                       (value.bbox[3] - value.bbox[1])))
            selected.append((image_index, face))
        landmark = self.app.models.get("landmark_3d_68")
        if landmark:
            for image_index, face in selected:
                landmark.get(images[image_index], face)
        recognition = self.app.models["recognition"]
        crops = [face_align.norm_crop(images[image_index], landmark=face.kps,
                                      image_size=recognition.input_size[0])
                 for image_index, face in selected]
        if crops:
            embeddings = recognition.get_feat(crops)
            for (_, face), embedding in zip(selected, embeddings):
                face.embedding = embedding

    def _analyze_decoded_batch(self, records: list[tuple[Path, str, Any]]) -> list[ImageAnalysis]:
        images = [record[2] for record in records]
        faces = self._detect_batch(images)
        self._enrich_largest_faces(images, faces)
        return [self._analysis_from_faces(path, digest, image, detected)
                for (path, digest, image), detected in zip(records, faces)]

    def analyze_many(self, entries: list[tuple[Path, str]], workers: int = 0, queue_size: int = 32,
                     batch_size: int = 0, cancelled: Callable[[], bool] | None = None) -> Iterator[ImageAnalysis]:
        worker_count = workers or self.default_cpu_workers
        batch_limit = batch_size or self.default_batch_size
        self.performance_summary.update({"gpu_batch_size": batch_limit,
                                         "cpu_decode_workers": worker_count,
                                         "decode_queue_size": queue_size})
        source: Iterator[tuple[Path, str]] = iter(entries)
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="fsc-decode") as executor:
            pending = deque()
            for _ in range(min(max(queue_size, batch_limit), len(entries))):
                pending.append(executor.submit(self._decode, next(source)))
            while pending:
                if cancelled and cancelled():
                    return
                group = []
                while pending and len(group) < batch_limit:
                    group.append(pending.popleft().result())
                    try:
                        pending.append(executor.submit(self._decode, next(source)))
                    except StopIteration:
                        pass
                valid = [(path, digest, image) for path, digest, image, error in group if error is None]
                analyzed = iter(self._analyze_decoded_batch(valid)) if valid else iter(())
                for _path, _digest, _image, error in group:
                    yield error if error is not None else next(analyzed)

    def analyze(self, path: Path, digest: str) -> ImageAnalysis:
        try:
            bgr = self._read(path)
            faces = self.app.get(bgr)
            return self._analysis_from_faces(path, digest, bgr, faces)
        except Exception as exc:
            return ImageAnalysis(path=str(path), fingerprint=digest, size_bytes=path.stat().st_size,
                                 face_count=0, flags=[f"analysis failed: {type(exc).__name__}: {exc}"], category="invalid")

    def _analysis_from_faces(self, path: Path, digest: str, bgr: Any, faces: list[Any]) -> ImageAnalysis:
        try:
            height, width = bgr.shape[:2]
            base = ImageAnalysis(path=str(path), fingerprint=digest, size_bytes=path.stat().st_size,
                                 width=width, height=height, face_count=len(faces))
            if not faces:
                base.flags.append("no face detected")
                return base
            face = max(faces, key=lambda value: float((value.bbox[2] - value.bbox[0]) * (value.bbox[3] - value.bbox[1])))
            x1, y1, x2, y2 = [int(value) for value in face.bbox]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            crop_bgr = bgr[y1:y2, x1:x2]
            crop_rgb = self.cv2.cvtColor(crop_bgr, self.cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop_rgb)
            gray = crop.convert("L")
            detection = float(face.det_score)
            pose = getattr(face, "pose", [0.0, 0.0, 0.0])
            embedding = [float(value) for value in face.normed_embedding]
            full_rgb = Image.fromarray(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))
            visual = analyze_face_visuals(
                full_rgb, [float(value) for value in face.bbox], detection,
                getattr(face, "kps", None), getattr(face, "landmark_3d_68", None),
            )
            base.detection_score = round(detection, 6)
            base.identity_score = round(confidence(cosine(embedding, self.target)), 6)
            for field in ("quality_score", "sharpness_score", "exposure_score", "blur_score",
                          "occlusion_score", "compression_artifact_score", "face_area_ratio",
                          "expression", "expression_confidence", "scene_embedding",
                          "appearance_embedding"):
                setattr(base, field, visual[field])
            base.pitch, base.yaw, base.roll = (round(float(value), 3) for value in pose[:3])
            base.phash = BaselineAnalyzer._average_hash(gray)
            base.embedding = embedding
            base.visual_embedding = list(base.scene_embedding) + list(base.appearance_embedding)
            base.flags.extend(visual["flags"])
            if len(faces) > 1:
                base.flags.append(f"multiple faces detected: {len(faces)}")
            return base
        except Exception as exc:
            return ImageAnalysis(path=str(path), fingerprint=digest, size_bytes=path.stat().st_size,
                                 face_count=0, flags=[f"analysis failed: {type(exc).__name__}: {exc}"], category="invalid")
