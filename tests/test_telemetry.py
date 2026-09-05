# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from types import SimpleNamespace

from faceset_curator.telemetry import format_metrics, parse_gpu_metrics


def test_parse_nvidia_smi_metrics():
    assert parse_gpu_metrics("37, 2048, 16384, 61\n") == (37, 2048, 16384, 61)


def test_format_live_hardware_metrics():
    memory = SimpleNamespace(used=8 * 1024 ** 3, total=32 * 1024 ** 3, percent=25)
    text = format_metrics(42, memory, (75, 4096, 16384, 64))
    assert "CPU 42%" in text
    assert "RAM 8.0/32.0 GB (25%)" in text
    assert "GPU 75%" in text
    assert "VRAM 4.0/16.0 GB" in text
    assert "64°C" in text
