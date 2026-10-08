"""训练路径配置与加密模型地址约束。"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# 训练 storage 根，如 .../storage
STORAGE_ROOT = Path(
    os.environ.get("NIII_TRAINING_STORAGE", "/niii_machine_version/AI_trainning_platform/storage")
)

# 闭环训练 subprocess 使用的 Python（mamba/venv，需装 ultralytics 8.4.113）
YOLO_PYTHON = os.environ.get(
    "NIII_YOLO_PYTHON", "/root/.local/share/mamba/envs/yolo/bin/python"
)

CLOSED_LOOP_TRAIN_SCRIPT = Path(__file__).resolve().parent / "closed_loop_train.py"

# Java 传入 last_train=storage/algorithms/.../models/baseline 时的平台根
PLATFORM_ROOT = Path(
    os.environ.get("NIII_TRAINING_PLATFORM_ROOT", str(STORAGE_ROOT.parent))
)


def _configured_model_roots() -> tuple[Path, ...]:
    raw = os.environ.get("NIII_TRAINING_MODEL_ROOTS", "").strip()
    values = [part.strip() for part in raw.split(os.pathsep) if part.strip()]
    if not values:
        values = [str(PLATFORM_ROOT)]
    return tuple(Path(value).expanduser().resolve() for value in values)


TRAINING_MODEL_ROOTS = _configured_model_roots()


def safe_id(name: str, value: object) -> str:
    """单层目录名，禁止路径穿越。"""
    text = str(value or "").strip()
    if not text or text in {".", ".."} or not _SAFE_ID.fullmatch(text):
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


def _ensure_under_model_roots(path: Path) -> Path:
    resolved = path.resolve()
    for root in TRAINING_MODEL_ROOTS:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    roots = ", ".join(str(root) for root in TRAINING_MODEL_ROOTS)
    raise ValueError(f"预训练模型路径越界: {resolved}；允许的根目录: {roots}")


def _validate_pretrained_file(path: Path) -> Path:
    resolved = _ensure_under_model_roots(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"预训练模型不存在: {resolved}")
    if resolved.suffix != ".niii-model":
        raise ValueError(f"预训练模型必须是 .niii-model 加密文件: {resolved}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"预训练模型文件为空: {resolved}")
    try:
        with resolved.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise ValueError(f"预训练模型不可读: {resolved}: {exc}") from exc
    return resolved


def resolve_pretrained_model_path(value: object) -> str:
    """解析后端传入的预训练权重文件或模型目录。"""
    text = str(value or "").strip()
    if not text:
        raise ValueError("pretrained_model_path 不能为空")
    if "://" in text:
        raise ValueError("pretrained_model_path 仅支持共享文件系统路径，不支持 URL")

    requested = Path(text).expanduser()
    if requested.is_absolute():
        candidate = _ensure_under_model_roots(requested)
    else:
        matches = [
            _ensure_under(root / requested, root)
            for root in TRAINING_MODEL_ROOTS
            if (root / requested).exists()
        ]
        if not matches:
            roots = ", ".join(str(root) for root in TRAINING_MODEL_ROOTS)
            raise FileNotFoundError(
                f"预训练模型地址不存在: {requested}；已检查: {roots}"
            )
        if len(matches) > 1:
            paths = ", ".join(str(path) for path in matches)
            raise ValueError(f"相对模型地址在多个允许根目录中存在，无法消歧: {paths}")
        candidate = matches[0]
    if candidate.is_file():
        return str(_validate_pretrained_file(candidate))
    if not candidate.exists():
        raise FileNotFoundError(f"预训练模型地址不存在: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"预训练模型地址不是文件或目录: {candidate}")

    for relative in (
        Path("weights") / "best.niii-model",
        Path("best.niii-model"),
    ):
        model_file = candidate / relative
        if model_file.is_file():
            return str(_validate_pretrained_file(model_file))

    model_files = sorted(path for path in candidate.glob("*.niii-model") if path.is_file())
    if not model_files:
        raise FileNotFoundError(
            "模型目录中未找到 weights/best.niii-model、best.niii-model "
            f"或直属 .niii-model 文件: {candidate}"
        )
    if len(model_files) > 1:
        names = ", ".join(path.name for path in model_files)
        raise ValueError(f"模型目录包含多个 .niii-model 文件，无法确定加载目标: {names}")
    return str(_validate_pretrained_file(model_files[0]))


def baseline_model_from_last_train(last_train: str) -> str:
    """Resolve encrypted baseline weights from the Java last_train field."""
    rel = last_train.replace("\\", "/").strip().lstrip("/")
    if not rel:
        raise ValueError("last_train 为空")
    if ".." in rel.split("/"):
        raise ValueError(f"非法 last_train: {last_train!r}")
    if rel.startswith("storage/"):
        rel = rel[len("storage/") :]
    storage = STORAGE_ROOT.resolve()
    candidate = storage / rel / "weights" / "best.niii-model"
    return str(_validate_pretrained_file(_ensure_under(candidate, storage)))


def classes_txt_path(project_id: str, task_id: str) -> Path:
    pid = safe_id("projectId", project_id)
    tid = safe_id("taskId", task_id)
    path = STORAGE_ROOT / pid / tid / "classes.txt"
    return _ensure_under(path, STORAGE_ROOT)


def train_save_dir(project_id: str, task_id: str, train_num: str) -> Path:
    pid = safe_id("projectId", project_id)
    tid = safe_id("taskId", task_id)
    tnum = safe_id("trainNum", train_num)
    path = STORAGE_ROOT / pid / tid / tnum
    return _ensure_under(path, STORAGE_ROOT)
