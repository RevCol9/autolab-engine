#!/usr/bin/env python3
"""Closed-loop training via ultralytics 8.3+ (supports YOLO11 baseline PT).

从 niii_training_algorithm/algorithm_model/yolo/closed_loop_train.py 原样移植。
由 training.trainer 通过 NIII_YOLO_PYTHON 子进程调用；产出 trainning_data.csv / report.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from ultralytics import YOLO

SAMPLING_INTERVAL_SEC = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--save-dir", required=True, help="absolute path to train folder")
    return p.parse_args()


class ResourceSampler:
    def __init__(self, interval=SAMPLING_INTERVAL_SEC):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._latest = capture_resource_snapshot()
        self.samples = [self._latest]

    def start(self):
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self._latest = capture_resource_snapshot()
            self.samples.append(self._latest)

    def snapshot(self):
        if not self._latest:
            self._latest = capture_resource_snapshot()
            self.samples.append(self._latest)
        return dict(self._latest)

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.interval * 2))
        final_sample = capture_resource_snapshot()
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


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "trainning_data.csv"
    process_start = time.time()
    train_start = process_start
    sampler = None
    state = {"epoch_start": train_start, "header_written": False}
    baseline_eval = {}
    trained_eval = {}
    actual_resource = {}
    sampler_started = False

    def on_fit_epoch_end(trainer):
        now = time.time()
        epoch = int(trainer.epoch) + 1
        epochs = int(trainer.epochs)
        epoch_spend = now - state["epoch_start"]
        state["epoch_start"] = now
        total_spend = now - train_start
        remaining = total_spend * (epochs - epoch) / epoch if epoch > 0 else 0
        metrics = trainer.metrics or {}
        resource = sampler.snapshot() if sampler else capture_resource_snapshot()
        row = {
            "epoch": epoch - 1,
            "train/box_loss": _metric(trainer, "train/box_loss"),
            "train/cls_loss": _metric(trainer, "train/cls_loss"),
            "train/dfl_loss": _metric(trainer, "train/dfl_loss"),
            "metrics/precision(B)": metric_lookup(metrics, "precision"),
            "metrics/recall(B)": metric_lookup(metrics, "recall"),
            "metrics/mAP50(B)": metric_lookup(metrics, "map50"),
            "metrics/mAP50-95(B)": metric_lookup(metrics, "map5095"),
            "val/box_loss": metrics.get("val/box_loss", 0),
            "val/cls_loss": metrics.get("val/cls_loss", 0),
            "val/dfl_loss": metrics.get("val/dfl_loss", 0),
            "lr/pg0": _lr(trainer, 0),
            "lr/pg1": _lr(trainer, 1),
            "lr/pg2": _lr(trainer, 2),
            "epoch_spend_time": round(epoch_spend, 4),
            "total_spend_time": round(total_spend, 4),
            "remaining_time": round(remaining, 4),
            "resource_cpu": resource.get("cpu", 0),
            "resource_mem": resource.get("mem", 0),
            "resource_gpu": resource.get("gpu", 0),
            "resource_gpu_mem_used_mb": resource.get("gpuMemUsedMb", 0),
            "resource_gpu_mem_total_mb": resource.get("gpuMemTotalMb", 0),
            "resource_disk": resource.get("disk", 0),
        }
        write_csv(csv_path, row, state)

    report = {}
    try:
        model = YOLO(args.model)
        baseline_eval = evaluate_model(model_path=args.model, data=args.data, batch=args.batch,
                                       imgsz=args.imgsz, device=args.device, project_dir=save_dir,
                                       run_name="baseline_eval")
        train_start = time.time()
        state["epoch_start"] = train_start
        sampler = ResourceSampler().start()
        sampler_started = True
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            save=True,
            amp=False,
        )
        trained_pt = save_dir / "weights" / "best.pt"
        actual_resource = sampler.stop()
        sampler_started = False
        if trained_pt.is_file():
            trained_eval = evaluate_model(model_path=str(trained_pt), data=args.data, batch=args.batch,
                                          imgsz=args.imgsz, device=args.device, project_dir=save_dir,
                                          run_name="trained_eval")
        else:
            trained_eval = {"error": "trained weights not found", "modelPath": str(trained_pt)}
    finally:
        if sampler_started:
            actual_resource = sampler.stop()
            sampler_started = False
        finished_at = utc_now()
        report = build_report(
            args=args,
            save_dir=save_dir,
            csv_path=csv_path,
            started_at=utc_now_from_seconds(train_start),
            finished_at=finished_at,
            actual_resource=actual_resource,
            baseline_eval=baseline_eval,
            trained_eval=trained_eval,
            run_duration_sec=max(0, int(round(time.time() - process_start))),
        )
        write_json(save_dir / "report.json", report)


def evaluate_model(model_path, data, batch, imgsz, device, project_dir, run_name):
    result = {
        "modelPath": str(model_path),
        "runDir": str(Path(project_dir) / run_name),
        "metrics": {},
        "speed": {},
        "maps": [],
        "names": [],
        "confusionMatrix": [],
        "error": None,
    }
    try:
        metrics = YOLO(model_path).val(
            data=data,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
            verbose=False,
        )
        result["metrics"] = to_builtin(getattr(metrics, "results_dict", {}))
        result["speed"] = to_builtin(getattr(metrics, "speed", {}))
        result["maps"] = to_builtin(getattr(metrics, "maps", []))
        result["names"] = normalize_names(getattr(metrics, "names", []))
        result.update(normalized_metric_summary(result["metrics"]))
        confusion = getattr(metrics, "confusion_matrix", None)
        if confusion is not None and hasattr(confusion, "matrix"):
            result["confusionMatrix"] = to_builtin(confusion.matrix)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def build_report(args, save_dir, csv_path, started_at, finished_at, actual_resource, baseline_eval, trained_eval,
                 run_duration_sec):
    series = read_series(csv_path)
    report = {
        "schemaVersion": 1,
        "task": "closed_loop",
        "model": args.model,
        "data": args.data,
        "project": args.project,
        "name": args.name,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "durationSec": actual_resource.get("durationSec", run_duration_sec),
        "runDurationSec": run_duration_sec,
        "actualResource": actual_resource,
        "baselineMetrics": baseline_eval,
        "trainedMetrics": trained_eval,
        "comparison": compare_metrics(baseline_eval, trained_eval),
        "series": series,
        "artifacts": collect_artifacts(save_dir),
    }
    return to_builtin(report)


def compare_metrics(before, after):
    before_metrics = (before or {}).get("metrics", {})
    after_metrics = (after or {}).get("metrics", {})
    comparison = {}
    for label in ("precision", "recall", "map50", "map5095", "fitness"):
        b = float(metric_lookup((before or {}).get(label), label, before_metrics) or 0)
        a = float(metric_lookup((after or {}).get(label), label, after_metrics) or 0)
        comparison[label] = {
            "before": round(b, 5),
            "after": round(a, 5),
            "delta": round(a - b, 5),
            "deltaPct": round(((a - b) / b) * 100, 2) if b else None,
        }
    before_speed = (before or {}).get("speed", {})
    after_speed = (after or {}).get("speed", {})
    comparison["speedInferenceMs"] = {
        "before": round(float(before_speed.get("inference", 0) or 0), 4),
        "after": round(float(after_speed.get("inference", 0) or 0), 4),
        "delta": round(float(after_speed.get("inference", 0) or 0) - float(before_speed.get("inference", 0) or 0), 4),
        "lowerIsBetter": True,
    }
    return comparison


def collect_artifacts(save_dir):
    names = [
        "trainning_data.csv",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    artifacts = []
    for name in names:
        path = save_dir / name
        if path.is_file():
            artifacts.append(str(path.relative_to(save_dir)))
    for sub in ("baseline_eval", "trained_eval"):
        sub_dir = save_dir / sub
        for name in [
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "F1_curve.png",
            "BoxPR_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "BoxF1_curve.png",
            "results.png",
            "val_batch0_labels.jpg",
            "val_batch0_pred.jpg",
        ]:
            path = sub_dir / name
            if path.is_file():
                artifacts.append(str(path.relative_to(save_dir)))
    weights_dir = save_dir / "weights"
    for name in ("best.pt", "last.pt"):
        path = weights_dir / name
        if path.is_file():
            artifacts.append(str(path.relative_to(save_dir)))
    return artifacts


def read_series(csv_path):
    series = []
    if not csv_path.is_file():
        return series
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            series.append({
                "epoch": int(float(row.get("epoch", 0) or 0)) + 1,
                "trainBoxLoss": to_number(row.get("train/box_loss")),
                "trainClsLoss": to_number(row.get("train/cls_loss")),
                "trainDflLoss": to_number(row.get("train/dfl_loss")),
                "precision": to_number(csv_lookup(row, "precision")),
                "recall": to_number(csv_lookup(row, "recall")),
                "map50": to_number(csv_lookup(row, "map50")),
                "map5095": to_number(csv_lookup(row, "map5095")),
                "valBoxLoss": to_number(row.get("val/box_loss")),
                "valClsLoss": to_number(row.get("val/cls_loss")),
                "valDflLoss": to_number(row.get("val/dfl_loss")),
                "epochSpendTime": to_number(row.get("epoch_spend_time")),
                "totalSpendTime": to_number(row.get("total_spend_time")),
                "remainingTime": to_number(row.get("remaining_time")),
                "resourceCpu": to_number(row.get("resource_cpu")),
                "resourceMem": to_number(row.get("resource_mem")),
                "resourceGpu": to_number(row.get("resource_gpu")),
                "resourceGpuMemUsedMb": to_number(row.get("resource_gpu_mem_used_mb")),
                "resourceGpuMemTotalMb": to_number(row.get("resource_gpu_mem_total_mb")),
                "resourceDisk": to_number(row.get("resource_disk")),
            })
    return series


def capture_resource_snapshot():
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    gpu = capture_gpu_snapshot()
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


def capture_gpu_snapshot():
    result = {
        "gpu": 0,
        "gpuName": "N/A",
        "gpuMemUsedMb": 0,
        "gpuMemTotalMb": 0,
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
        line = stdout.splitlines()[0] if stdout else ""
        if line:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                result["gpu"] = to_int(parts[0])
                result["gpuMemUsedMb"] = to_int(parts[1])
                result["gpuMemTotalMb"] = to_int(parts[2])
                result["gpuName"] = parts[3]
    except Exception:
        pass
    return result


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def utc_now_from_seconds(seconds):
    return datetime.utcfromtimestamp(seconds).isoformat(timespec="seconds") + "Z"


def _metric(trainer, key):
    try:
        if hasattr(trainer, "label_loss_items"):
            items = trainer.label_loss_items(trainer.tloss if hasattr(trainer, "tloss") else [])
            if isinstance(items, dict) and key in items:
                return items[key]
    except Exception:
        pass
    return 0


def _lr(trainer, idx):
    try:
        groups = trainer.optimizer.param_groups
        if idx < len(groups):
            return groups[idx].get("lr", 0)
    except Exception:
        pass
    return 0


def write_csv(csv_path, row, state):
    fieldnames = list(row.keys())
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header and not state["header_written"]:
            writer.writeheader()
            state["header_written"] = True
        writer.writerow(row)
        f.flush()


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def to_number(value):
    try:
        if value is None or value == "":
            return 0
        return round(float(value), 5)
    except Exception:
        return 0


METRIC_ALIASES = {
    "precision": ["precision", "metrics/precision(B)", "metrics/precision", "mp", "P"],
    "recall": ["recall", "metrics/recall(B)", "metrics/recall", "mr", "R"],
    "map50": ["map50", "mAP50", "metrics/mAP50(B)", "metrics/mAP50", "metrics/map50"],
    "map5095": ["map5095", "mAP50-95", "metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/map5095"],
    "fitness": ["fitness"],
}


def first_number(*values):
    for value in values:
        if value is None or value == "":
            continue
        try:
            return round(float(value), 5)
        except Exception:
            continue
    return 0


def metric_lookup(primary, label=None, metrics=None):
    if isinstance(primary, dict):
        metrics = primary
        primary = None
    values = [primary]
    for name in METRIC_ALIASES.get(label, [label]):
        if metrics and name in metrics:
            values.append(metrics.get(name))
    return first_number(*values)


def csv_lookup(row, label):
    for name in METRIC_ALIASES.get(label, [label]):
        if name in row:
            return row.get(name)
    return 0


def normalized_metric_summary(metrics):
    return {
        "precision": metric_lookup(None, "precision", metrics),
        "recall": metric_lookup(None, "recall", metrics),
        "map50": metric_lookup(None, "map50", metrics),
        "map5095": metric_lookup(None, "map5095", metrics),
        "fitness": metric_lookup(None, "fitness", metrics),
    }


def to_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0


def normalize_names(names):
    if isinstance(names, dict):
        return [names[key] for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


if __name__ == "__main__":
    main()
