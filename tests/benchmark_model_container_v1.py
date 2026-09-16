"""Measure v1 conversion/load peak RSS with real YOLO architectures.

Run on the target host with ``python -m tests.benchmark_model_container_v1``.
Generated .pt fixtures are confined to a temporary directory.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

import psutil
import torch
from ultralytics import YOLO

from toolkit.model_crypto.container_v1 import read_model_container
from toolkit.model_crypto.offline_convert import convert_trusted_pt

T = TypeVar("T")


def _measure(operation: Callable[[], T]) -> tuple[T, float, float]:
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.005):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        result = operation()
    finally:
        elapsed = time.perf_counter() - started
        peak = max(peak, process.memory_info().rss)
        stop.set()
        sampler.join()
    return result, elapsed, (peak - baseline) / 1024**2


def main() -> None:
    kek = b"P0-test-key-material".ljust(32, b"\0")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        keys = root / "keys"
        keys.mkdir()
        (keys / "kek.key").write_bytes(kek)
        for task, architecture in (
            ("detect", "yolo11n.yaml"),
            ("segment", "yolo11n-seg.yaml"),
        ):
            model = YOLO(architecture).model
            source = root / f"{task}.pt"
            target = root / f"{task}.niii-model"
            torch.save({"model": model}, source)

            _, convert_seconds, convert_peak_mib = _measure(
                lambda: convert_trusted_pt(source, target, task=task, key_dir=keys)
            )
            container, load_seconds, load_peak_mib = _measure(
                lambda: read_model_container(target, kek=kek)
            )
            print(
                f"{task}: tensors={len(container.state_dict)} "
                f"artifact_mib={target.stat().st_size / 1024**2:.1f} "
                f"convert_s={convert_seconds:.2f} "
                f"convert_peak_rss_delta_mib={convert_peak_mib:.1f} "
                f"load_s={load_seconds:.2f} "
                f"load_peak_rss_delta_mib={load_peak_mib:.1f}"
            )
            del container, model


if __name__ == "__main__":
    main()
