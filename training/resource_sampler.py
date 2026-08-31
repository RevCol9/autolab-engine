"""训练期间 CPU/GPU/内存采样。"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import psutil

SAMPLING_INTERVAL_SEC = 1.0


def capture_gpu_snapshot(device: str | int = 0):
    device_index = int(str(device).strip() or "0")
    result = {
        "gpu": 0,
        "gpuName": "N/A",
        "gpuMemUsedMb": 0,
        "gpuMemTotalMb": 0,
        "device_index": device_index,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        stdout = (proc.stdout or "").strip()
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        line = lines[device_index] if device_index < len(lines) else (lines[0] if lines else "")
        if line:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                result["gpu"] = _to_int(parts[0])
                result["gpuMemUsedMb"] = _to_int(parts[1])
                result["gpuMemTotalMb"] = _to_int(parts[2])
                result["gpuName"] = parts[3]
    except Exception:
        pass
    return result


def capture_resource_snapshot(device: str | int = 0):
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    gpu = capture_gpu_snapshot(device)
    now = time.time()
    return {
        "timestamp": now,
        "cpu": round(psutil.cpu_percent(interval=None), 2),
        "mem": round(mem.percent, 2),
        "memUsedMb": int(mem.used / 1024 / 1024),
        "memTotalMb": int(mem.total / 1024 / 1024),
        "memAvailableMb": int(mem.available / 1024 / 1024),
        "gpu": gpu["gpu"],
        "gpuName": gpu["gpuName"],
        "gpuMemUsedMb": gpu["gpuMemUsedMb"],
        "gpuMemTotalMb": gpu["gpuMemTotalMb"],
        "disk": round(disk.percent, 2),
        "diskFreeRatio": round(100 - disk.percent, 2),
    }


class ResourceSampler:
    def __init__(self, interval=SAMPLING_INTERVAL_SEC, device: str | int = 0):
        self.interval = interval
        self.device = device
        self._stop = threading.Event()
        self._thread = None
        self._latest = capture_resource_snapshot(self.device)
        self.samples = [self._latest]

    def start(self):
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self._latest = capture_resource_snapshot(self.device)
            self.samples.append(self._latest)

    def snapshot(self):
        if not self._latest:
            self._latest = capture_resource_snapshot(self.device)
            self.samples.append(self._latest)
        return dict(self._latest)

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.interval * 2))
        final_sample = capture_resource_snapshot(self.device)
        self._latest = final_sample
        self.samples.append(final_sample)
        return self.summary()

    def summary(self):
        if not self.samples:
            return {}
        cpu_values = [float(sample.get("cpu", 0)) for sample in self.samples]
        mem_values = [float(sample.get("mem", 0)) for sample in self.samples]
        gpu_values = [float(sample.get("gpu", 0)) for sample in self.samples]
        gpu_mem_values = [float(sample.get("gpuMemUsedMb", 0)) for sample in self.samples]
        disk_values = [float(sample.get("disk", 0)) for sample in self.samples]
        durations = [sample.get("timestamp") for sample in self.samples]
        duration_sec = 0
        if len(durations) >= 2:
            duration_sec = max(0, int(round(durations[-1] - durations[0])))
        return {
            "durationSec": duration_sec,
            "sampleCount": len(self.samples),
            "cpuAvg": round(sum(cpu_values) / len(cpu_values), 2),
            "cpuPeak": round(max(cpu_values), 2),
            "memAvg": round(sum(mem_values) / len(mem_values), 2),
            "memPeakMb": int(max(float(sample.get("memUsedMb", 0)) for sample in self.samples)),
            "gpuAvg": round(sum(gpu_values) / len(gpu_values), 2),
            "gpuUsagePeak": round(max(gpu_values), 2),
            "gpuMemPeakMb": int(max(gpu_mem_values)),
            "diskPeak": round(max(disk_values), 2),
        }


def _to_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0
