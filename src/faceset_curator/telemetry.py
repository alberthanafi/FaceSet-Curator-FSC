# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import shutil
import subprocess

import psutil


def parse_gpu_metrics(output: str) -> tuple[int, int, int, int]:
    fields = [field.strip() for field in output.splitlines()[0].split(",")]
    if len(fields) != 4:
        raise ValueError("Unexpected nvidia-smi output")
    return tuple(int(float(field)) for field in fields)  # type: ignore[return-value]


def format_metrics(cpu: float, memory: object, gpu: tuple[int, int, int, int] | None) -> str:
    used_gib = memory.used / (1024 ** 3)  # type: ignore[attr-defined]
    total_gib = memory.total / (1024 ** 3)  # type: ignore[attr-defined]
    text = f"CPU {cpu:.0f}%   •   RAM {used_gib:.1f}/{total_gib:.1f} GB ({memory.percent:.0f}%)"  # type: ignore[attr-defined]
    if gpu is None:
        return f"{text}   •   GPU unavailable"
    utilization, used_mib, total_mib, temperature = gpu
    return (f"{text}   •   GPU {utilization}%   •   "
            f"VRAM {used_mib / 1024:.1f}/{total_mib / 1024:.1f} GB   •   {temperature}°C")


class SystemMonitor:
    def __init__(self) -> None:
        self.nvidia_smi = shutil.which("nvidia-smi")
        psutil.cpu_percent(interval=None)

    def sample(self) -> str:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        gpu = None
        if self.nvidia_smi:
            try:
                completed = subprocess.run(
                    [self.nvidia_smi, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                gpu = parse_gpu_metrics(completed.stdout)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
        return format_metrics(cpu, memory, gpu)
