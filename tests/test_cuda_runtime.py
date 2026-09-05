# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from pathlib import Path

from faceset_curator.cuda_analyzer import (BackendUnavailable, InsightFaceAnalyzer,
                                           batch_size_for_vram, prepare_cuda_runtime)
from faceset_curator.models import ImageAnalysis


class FakeRuntime:
    __version__ = "1.26.0"

    def __init__(self, providers, preload_error=None):
        self.providers = providers
        self.preload_error = preload_error
        self.preloaded_from = None

    def get_available_providers(self):
        return self.providers

    def preload_dlls(self, directory=None):
        self.preloaded_from = directory
        if self.preload_error:
            raise self.preload_error


def test_cuda_runtime_preloads_packaged_libraries():
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert prepare_cuda_runtime(runtime) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.preloaded_from == ""


def test_cuda_runtime_refuses_silent_cpu_fallback():
    runtime = FakeRuntime(["CPUExecutionProvider"])
    try:
        prepare_cuda_runtime(runtime)
    except BackendUnavailable as exc:
        assert "CUDA is unavailable" in str(exc)
    else:
        raise AssertionError("CPU fallback should have been rejected")


def test_explicit_cpu_mode_does_not_preload_cuda():
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert prepare_cuda_runtime(runtime, "cpu") == ["CPUExecutionProvider"]
    assert runtime.preloaded_from is None


def test_gpu_batch_size_scales_with_vram():
    assert batch_size_for_vram(6 * 1024) == 2
    assert batch_size_for_vram(10 * 1024) == 4
    assert batch_size_for_vram(16 * 1024) == 8
    assert batch_size_for_vram(24 * 1024) == 12


def test_bounded_batch_pipeline_preserves_input_order():
    analyzer = InsightFaceAnalyzer.__new__(InsightFaceAnalyzer)
    analyzer.default_cpu_workers = 2
    analyzer.default_batch_size = 2
    analyzer.performance_summary = {}
    batches = []

    def decode(entry):
        path, digest = entry
        return path, digest, object(), None

    def analyze_batch(records):
        batches.append([digest for _path, digest, _image in records])
        return [ImageAnalysis(path=str(path), fingerprint=digest, size_bytes=1)
                for path, digest, _image in records]

    analyzer._decode = decode
    analyzer._analyze_decoded_batch = analyze_batch
    entries = [(Path(f"{index}.jpg"), str(index)) for index in range(5)]
    results = list(analyzer.analyze_many(entries, workers=2, queue_size=3, batch_size=2))
    assert [item.fingerprint for item in results] == ["0", "1", "2", "3", "4"]
    assert batches == [["0", "1"], ["2", "3"], ["4"]]
