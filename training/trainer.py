"""检测闭环训练调度。"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from training.hparams import write_job_train_config
from training.paths import CLOSED_LOOP_TRAIN_SCRIPT, YOLO_PYTHON, train_save_dir

logger = logging.getLogger(__name__)


def build_closed_loop_cmd(param: Dict[str, Any], *, device: str = "0") -> list[str]:
    """写入 train_config.yaml 并组装 closed_loop_train 子进程命令。"""
    config_path = write_job_train_config(param, device=device)
    return [
        YOLO_PYTHON,
        str(CLOSED_LOOP_TRAIN_SCRIPT),
        "--config",
        str(config_path),
    ]


def collect_train_result(save_dir: Path) -> Dict[str, Any]:
    """训练结束后校验产物；缺 best.pt 视为失败。"""
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
        "csv": str(save_dir / "trainning_data.csv"),
        "log": str(save_dir / "train.log"),
    }


def popen_detection_train(
    param: Dict[str, Any],
    *,
    device: str = "0",
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    """异步启动训练子进程；stdout/stderr 追加到 save_dir/train.log。"""
    cmd = build_closed_loop_cmd(param, device=device)
    save_dir = train_save_dir(str(param["projectId"]), str(param["taskId"]), str(param["trainNum"]))
    log_path = save_dir / "train.log"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "a", encoding="utf-8")
    logger.info("closed_loop_train popen: %s | log=%s", " ".join(cmd), log_path)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or "/tmp",
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    log_fp.close()  # 子进程已继承 fd，父进程关闭副本避免泄漏
    return proc


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
