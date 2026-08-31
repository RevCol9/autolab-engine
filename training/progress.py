"""从闭环训练产物解析进度，供 HTTP 轮询。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from training.paths import train_save_dir


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_epoch_row(row: Dict[str, str]) -> Dict[str, Any]:
    """将 ``trainning_data.csv`` 单行转为 API 友好结构。"""
    epoch_idx = _to_int(row.get("epoch"))
    display_epoch = (epoch_idx + 1) if epoch_idx is not None else None
    return {
        "epoch": epoch_idx,
        "epoch_display": display_epoch,
        "train_box_loss": _to_float(row.get("train/box_loss")),
        "train_cls_loss": _to_float(row.get("train/cls_loss")),
        "train_dfl_loss": _to_float(row.get("train/dfl_loss")),
        "precision": _to_float(row.get("metrics/precision(B)")),
        "recall": _to_float(row.get("metrics/recall(B)")),
        "map50": _to_float(row.get("metrics/mAP50(B)")),
        "map50_95": _to_float(row.get("metrics/mAP50-95(B)")),
        "epoch_spend_sec": _to_float(row.get("epoch_spend_time")),
        "total_spend_sec": _to_float(row.get("total_spend_time")),
        "remaining_sec": _to_float(row.get("remaining_time")),
        "resource": {
            "cpu": _to_float(row.get("resource_cpu")),
            "mem": _to_float(row.get("resource_mem")),
            "gpu": _to_float(row.get("resource_gpu")),
            "gpu_mem_used_mb": _to_int(row.get("resource_gpu_mem_used_mb")),
            "gpu_mem_total_mb": _to_int(row.get("resource_gpu_mem_total_mb")),
        },
    }


def read_job_progress(
    project_id: str,
    task_id: str,
    train_num: str,
    *,
    total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """读取任务目录下 CSV 最新 epoch；训练尚未写盘时返回空进度。"""
    save_dir = train_save_dir(project_id, task_id, train_num)
    csv_path = save_dir / "trainning_data.csv"
    log_path = save_dir / "train.log"
    rows = _read_csv_rows(csv_path)
    latest = parse_epoch_row(rows[-1]) if rows else None
    completed = latest.get("epoch") if latest else None
    completed_display = latest.get("epoch_display") if latest else 0

    progress_pct = None
    if total_epochs and total_epochs > 0 and completed_display is not None:
        progress_pct = round(min(100.0, completed_display / total_epochs * 100.0), 2)

    return {
        "save_dir": str(save_dir),
        "csv": str(csv_path) if csv_path.is_file() else None,
        "log": str(log_path) if log_path.is_file() else None,
        "epochs_completed": completed_display or 0,
        "epochs_total": total_epochs,
        "progress_pct": progress_pct,
        "latest": latest,
        "history_count": len(rows),
    }
