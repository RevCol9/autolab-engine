"""检测闭环训练调度（从 yolo_detect.YoloDetector.train 移植，去掉 Django）。"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Optional

from training.data_yaml import prepare_data_yaml_for_job
from training.paths import (
    CLOSED_LOOP_TRAIN_SCRIPT,
    YOLO_PYTHON,
    baseline_pt_from_last_train,
    train_save_dir,
)

logger = logging.getLogger(__name__)


def resolve_model_path(param: Dict[str, Any]) -> str:
    """解析起始权重：续训用 last_train 基线，否则用 param.model。"""
    if param.get("is_continue") is True and param.get("last_train"):
        return baseline_pt_from_last_train(str(param["last_train"]))
    model = param.get("model") or "yolov8n.pt"
    text = str(model).strip()
    # 短名如 Yolov8n.pt → 小写；绝对路径保持原样
    if "/" not in text and "\\" not in text:
        return text.lower()
    return text


def build_closed_loop_cmd(param: Dict[str, Any], *, device: str = "0") -> list[str]:
    """组装 closed_loop_train.py 命令行（与原 yolo_detect.train 一致）。"""
    project_id = str(param["projectId"])
    task_id = str(param["taskId"])
    train_num = str(param["trainNum"])

    data_yaml = prepare_data_yaml_for_job(param)
    save_path = train_save_dir(project_id, task_id, train_num)
    save_path.mkdir(parents=True, exist_ok=True)
    train_parent = str(save_path.parent)
    train_name = save_path.name
    model_path = resolve_model_path(param)

    return [
        YOLO_PYTHON,
        str(CLOSED_LOOP_TRAIN_SCRIPT),
        "--model",
        model_path,
        "--data",
        str(data_yaml),
        "--project",
        train_parent,
        "--name",
        train_name,
        "--epochs",
        str(int(param["epochs"])),
        "--batch",
        str(int(param["batch_size"])),
        "--imgsz",
        str(int(param["image_size"])),
        "--device",
        str(device),
        "--save-dir",
        str(save_path),
    ]


def run_detection_train(
    param: Dict[str, Any],
    *,
    device: str = "0",
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """同步执行闭环检测训练（阻塞到结束）。

    Returns:
        {"train_status": "finished", "weight_path": ".../weights/best.pt", "save_dir": "..."}
    """
    cmd = build_closed_loop_cmd(param, device=device)
    save_dir = train_save_dir(str(param["projectId"]), str(param["taskId"]), str(param["trainNum"]))
    logger.info("closed_loop_train cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, cwd=cwd or "/tmp")
    if proc.returncode != 0:
        raise RuntimeError(f"closed_loop_train failed, exit={proc.returncode}")
    weight = save_dir / "weights" / "best.pt"
    return {
        "train_status": "finished",
        "weight_path": str(weight),
        "save_dir": str(save_dir),
        "report": str(save_dir / "report.json"),
        "csv": str(save_dir / "trainning_data.csv"),
    }


def popen_detection_train(
    param: Dict[str, Any],
    *,
    device: str = "0",
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    """异步启动训练子进程，由 JobManager 管理。"""
    cmd = build_closed_loop_cmd(param, device=device)
    logger.info("closed_loop_train popen: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=cwd or "/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
