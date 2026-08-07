"""Deployment path configuration (override via environment variables)."""
import os
from pathlib import Path

# 训练 storage 根，如 .../storage
STORAGE_ROOT = Path(
    os.environ.get("NIII_TRAINING_STORAGE", "/niii_machine_version/AI_trainning_platform/storage")
)

# Django 项目根（boot-vison-python-master）
PROJECT_ROOT = Path(
    os.environ.get("NIII_ALGORITHM_ROOT", "/niii_machine_version/AI_trainning_platform/boot-vison-python-master")
)

# 闭环训练 subprocess 使用的 Python（mamba/venv）
YOLO_PYTHON = os.environ.get(
    "NIII_YOLO_PYTHON", "/root/.local/share/mamba/envs/yolo/bin/python"
)

CLOSED_LOOP_TRAIN_SCRIPT = PROJECT_ROOT / "algorithm_model" / "yolo" / "closed_loop_train.py"

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
