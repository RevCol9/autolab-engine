"""闭环训练调度：写 train_config.yaml 并 Popen closed_loop_train 子进程。"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from training.hparams import write_job_train_config
from training.paths import CLOSED_LOOP_TRAIN_SCRIPT, YOLO_PYTHON, train_save_dir
from training.run_artifacts import TRAINING_METRICS_CSV, reset_run_artifacts

logger = logging.getLogger(__name__)


def build_closed_loop_cmd(
    param: Dict[str, Any],
    *,
    task: str,
    device: Optional[str] = None,
) -> list[str]:
    config_path = write_job_train_config(param, task=task, device=device)
    return [
        YOLO_PYTHON,
        str(CLOSED_LOOP_TRAIN_SCRIPT),
        "--config",
        str(config_path),
    ]


def collect_train_result(save_dir: Path) -> Dict[str, Any]:
    weight = save_dir / "weights" / "best.pt"
    if not weight.is_file():
        raise RuntimeError(f"trained weights not found: {weight}")
    config_path = save_dir / "train_config.yaml"
    return {
        "train_status": "finished",
        "weight_path": str(weight),
        "save_dir": str(save_dir),
        "config": str(config_path) if config_path.is_file() else None,
        "report": str(save_dir / "report.json"),
        "csv": str(save_dir / TRAINING_METRICS_CSV),
        "log": str(save_dir / "train.log"),
    }


def popen_train(
    param: Dict[str, Any],
    *,
    task: str = "detection",
    device: Optional[str] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    cmd = build_closed_loop_cmd(param, task=task, device=device)
    save_dir = train_save_dir(str(param["projectId"]), str(param["taskId"]), str(param["trainNum"]))
    log_path = save_dir / "train.log"
    reset_run_artifacts(save_dir)
    logger.info("closed_loop_train popen task=%s: %s | log=%s", task, " ".join(cmd), log_path)
    with open(log_path, "w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd or "/tmp",
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return proc


# 兼容旧调用名
def popen_detection_train(
    param: Dict[str, Any],
    *,
    device: Optional[str] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    return popen_train(param, task="detection", device=device, cwd=cwd)

def kill_process_group(proc: subprocess.Popen, timeout: float = 15.0) -> bool:
    """终止子进程组；仅在确认进程已退出后返回 True。"""
    pid = proc.pid
    if pid is None:
        return proc.poll() is not None
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.poll() is not None
    except Exception as exc:
        logger.debug("killpg SIGTERM failed pid=%s: %s", pid, exc)
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception as fallback_exc:
            logger.warning("send SIGTERM failed pid=%s: %s", pid, fallback_exc)
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("SIGTERM timed out pid=%s; escalating to SIGKILL", pid)
    except Exception as exc:
        logger.warning("wait after SIGTERM failed pid=%s: %s", pid, exc)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return proc.poll() is not None
    except Exception as exc:
        logger.debug("killpg SIGKILL failed pid=%s: %s", pid, exc)
        try:
            proc.kill()
        except Exception as fallback_exc:
            logger.warning("process kill failed pid=%s: %s", pid, fallback_exc)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("kill process group timed out pid=%s", pid)
    except Exception as exc:
        logger.warning("final wait failed pid=%s: %s", pid, exc)
    return proc.poll() is not None
