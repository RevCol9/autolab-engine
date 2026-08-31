"""训练报告：评估指标、CSV 序列、report.json 生成。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

METRIC_ALIASES = {
    "precision": ["precision", "metrics/precision(B)", "metrics/precision", "mp", "P"],
    "recall": ["recall", "metrics/recall(B)", "metrics/recall", "mr", "R"],
    "map50": ["map50", "mAP50", "metrics/mAP50(B)", "metrics/mAP50", "metrics/map50"],
    "map5095": ["map5095", "mAP50-95", "metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/map5095"],
    "fitness": ["fitness"],
}


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


def build_report(
    config,
    save_dir,
    csv_path,
    started_at,
    finished_at,
    actual_resource,
    baseline_eval,
    trained_eval,
    run_duration_sec,
):
    series = read_series(csv_path)
    report = {
        "schemaVersion": 1,
        "task": "closed_loop",
        "model": config.get("model"),
        "data": config.get("data"),
        "project": config.get("project"),
        "name": config.get("name"),
        "epochs": config.get("epochs"),
        "batch": config.get("batch"),
        "imgsz": config.get("imgsz"),
        "device": config.get("device"),
        "trainConfig": str(save_dir / "train_config.yaml"),
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
        "train_config.yaml",
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


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def utc_now_from_seconds(seconds):
    return datetime.utcfromtimestamp(seconds).isoformat(timespec="seconds") + "Z"


def trainer_metric(trainer, key):
    try:
        if hasattr(trainer, "label_loss_items"):
            items = trainer.label_loss_items(trainer.tloss if hasattr(trainer, "tloss") else [])
            if isinstance(items, dict) and key in items:
                return items[key]
    except Exception:
        pass
    return 0


def trainer_lr(trainer, idx):
    try:
        groups = trainer.optimizer.param_groups
        if idx < len(groups):
            return groups[idx].get("lr", 0)
    except Exception:
        pass
    return 0


def to_number(value):
    try:
        if value is None or value == "":
            return 0
        return round(float(value), 5)
    except Exception:
        return 0


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
