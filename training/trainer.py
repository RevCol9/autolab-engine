"""检测闭环训练调度（从 yolo_detect.YoloDetector.train 移植，去掉 Django）。"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from training.data_yaml import prepare_data_yaml_for_job
from training.paths import (
    CLOSED_LOOP_TRAIN_SCRIPT,
    YOLO_PYTHON,
    baseline_pt_from_last_train,
    train_save_dir,
)

logger = logging.getLogger(__name__)


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def resolve_model_path(param: Dict[str, Any]) -> str:
    """解析起始权重：续训用 last_train 基线，否则用 param.model。"""
    if _truthy(param.get("is_continue")):
        last_train = param.get("last_train")
        if not last_train:
            raise ValueError("is_continue=true 时必须提供 last_train")
        return baseline_pt_from_last_train(str(last_train))
    model = param.get("model") or "yolov8n.pt"
    text = str(model).strip()
    # 短名如 Yolov8n.pt → 小写；绝对/相对路径保持大小写
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


def collect_train_result(save_dir: Path) -> Dict[str, Any]:
    """训练结束后校验产物；缺 best.pt 视为失败。"""
    weight = save_dir / "weights" / "best.pt"
    if not weight.is_file():
        raise RuntimeError(f"trained weights not found: {weight}")
    return {
        "train_status": "finished",
        "weight_path": str(weight),
        "save_dir": str(save_dir),
        "report": str(save_dir / "report.json"),
        "csv": str(save_dir / "trainning_data.csv"),
    }


def run_detection_train(
    param: Dict[str, Any],
    *,
    device: str = "0",
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """同步执行闭环检测训练（阻塞到结束）。"""
    cmd = build_closed_loop_cmd(param, device=device)
    save_dir = train_save_dir(str(param["projectId"]), str(param["taskId"]), str(param["trainNum"]))
    logger.info("closed_loop_train cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, cwd=cwd or "/tmp", start_new_session=True)
    if proc.returncode != 0:
        raise RuntimeError(f"closed_loop_train failed, exit={proc.returncode}")
    return collect_train_result(save_dir)


def popen_detection_train(
    param: Dict[str, Any],
    *,
    device: str = "0",
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    """异步启动训练子进程，由 JobManager 管理。

    不使用 stdout=PIPE：父进程若不 drain 会在缓冲写满后死锁子进程。
    start_new_session=True：stop 时可杀整个进程组（含 ultralytics worker）。
    """
    cmd = build_closed_loop_cmd(param, device=device)
    logger.info("closed_loop_train popen: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=cwd or "/tmp",
        stdout=None,
        stderr=None,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def kill_process_group(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    """终止子进程及其进程组。"""
    pid = proc.pid
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
    try:
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        logger.warning("kill process group timed out pid=%s", pid)
