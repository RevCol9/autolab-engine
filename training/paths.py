"""训练路径配置（从 niii_training_algorithm/niii_paths.py 移植）。

可通过环境变量覆盖；与 Java boot-vision 的 storage / 基线 PT 约定保持一致。
"""

from __future__ import annotations

import os
from pathlib import Path

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


def baseline_pt_from_last_train(last_train: str) -> str:
    """Resolve baseline weights path from Java last_train field."""
    rel = last_train.replace("\\", "/").lstrip("/")
    if rel.startswith("storage/"):
        rel = rel[len("storage/") :]
    return str(PLATFORM_ROOT / "storage" / rel / "weights" / "best.pt")


def classes_txt_path(project_id: str, task_id: str) -> Path:
    return STORAGE_ROOT / project_id / task_id / "classes.txt"


def classes_yaml_path(project_id: str, task_id: str) -> Path:
    return STORAGE_ROOT / project_id / task_id / "classes.yaml"


def train_save_dir(project_id: str, task_id: str, train_num: str) -> Path:
    return STORAGE_ROOT / project_id / task_id / train_num
