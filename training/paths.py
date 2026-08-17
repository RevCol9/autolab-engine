"""训练路径配置（从 niii_training_algorithm/niii_paths.py 移植）。

可通过环境变量覆盖；与 Java boot-vision 的 storage / 基线 PT 约定保持一致。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# 训练 storage 根，如 .../storage
STORAGE_ROOT = Path(
    os.environ.get("NIII_TRAINING_STORAGE", "/niii_machine_version/AI_trainning_platform/storage")
)

# 本训练模块所在仓库根（默认 autolab-engine）
PROJECT_ROOT = Path(
    os.environ.get(
        "NIII_ALGORITHM_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
)

# 闭环训练 subprocess 使用的 Python（mamba/venv，需装 ultralytics 8.3+）
YOLO_PYTHON = os.environ.get(
    "NIII_YOLO_PYTHON", "/root/.local/share/mamba/envs/yolo/bin/python"
)

CLOSED_LOOP_TRAIN_SCRIPT = Path(__file__).resolve().parent / "closed_loop_train.py"

# Java 传入 last_train=storage/algorithms/.../models/baseline 时的平台根
PLATFORM_ROOT = Path(
    os.environ.get("NIII_TRAINING_PLATFORM_ROOT", str(STORAGE_ROOT.parent))
)


def safe_id(name: str, value: object) -> str:
    """单层目录名，禁止路径穿越。"""
    text = str(value or "").strip()
    if not text or not _SAFE_ID.match(text):
        raise ValueError(f"非法 {name}: {value!r}（仅允许字母数字 . _ -）")
    return text


def _ensure_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"路径越界: {path} 不在 {root_resolved} 内") from exc
    return resolved


def baseline_pt_from_last_train(last_train: str) -> str:
    """Resolve baseline weights path from Java last_train field."""
    rel = last_train.replace("\\", "/").strip().lstrip("/")
    if not rel:
        raise ValueError("last_train 为空")
    if ".." in rel.split("/"):
        raise ValueError(f"非法 last_train: {last_train!r}")
    if rel.startswith("storage/"):
        rel = rel[len("storage/") :]
    storage = (PLATFORM_ROOT / "storage").resolve()
    candidate = storage / rel / "weights" / "best.pt"
    return str(_ensure_under(candidate, storage))


def classes_txt_path(project_id: str, task_id: str) -> Path:
    pid = safe_id("projectId", project_id)
    tid = safe_id("taskId", task_id)
    path = STORAGE_ROOT / pid / tid / "classes.txt"
    return _ensure_under(path, STORAGE_ROOT)


def classes_yaml_path(project_id: str, task_id: str) -> Path:
    pid = safe_id("projectId", project_id)
    tid = safe_id("taskId", task_id)
    path = STORAGE_ROOT / pid / tid / "classes.yaml"
    return _ensure_under(path, STORAGE_ROOT)


def train_save_dir(project_id: str, task_id: str, train_num: str) -> Path:
    pid = safe_id("projectId", project_id)
    tid = safe_id("taskId", task_id)
    tnum = safe_id("trainNum", train_num)
    path = STORAGE_ROOT / pid / tid / tnum
    return _ensure_under(path, STORAGE_ROOT)
