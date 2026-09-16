"""YOLO 训练数据的启动前格式校验。"""

from __future__ import annotations

import math
from pathlib import Path


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def validate_image_dir(images_dir: Path) -> int:
    """确认图像目录存在且至少包含一张 Ultralytics 可读图像。"""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"训练图像目录不存在: {images_dir}")
    count = sum(
        1
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if count == 0:
        raise ValueError(f"训练图像目录为空: {images_dir}")
    return count


def _parse_class_id(token: str, *, path: Path, line_number: int) -> int:
    try:
        class_id = int(token)
    except ValueError as exc:
        raise ValueError(f"标签类别须为非负整数: {path}:{line_number}") from exc
    if class_id < 0:
        raise ValueError(f"标签类别须为非负整数: {path}:{line_number}")
    return class_id


def _parse_normalized_coordinates(
    tokens: list[str],
    *,
    path: Path,
    line_number: int,
) -> list[float]:
    try:
        coords = [float(value) for value in tokens]
    except ValueError as exc:
        raise ValueError(f"标签坐标须为数字: {path}:{line_number}") from exc
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coords):
        raise ValueError(f"标签坐标须位于 [0, 1]: {path}:{line_number}")
    return coords


def _validate_detection_line(parts: list[str], *, path: Path, line_number: int) -> None:
    if len(parts) != 5:
        raise ValueError(f"检测标签须为 5 列（class cx cy w h）: {path}:{line_number}")
    _parse_class_id(parts[0], path=path, line_number=line_number)
    coords = _parse_normalized_coordinates(parts[1:], path=path, line_number=line_number)
    if coords[2] <= 0.0 or coords[3] <= 0.0:
        raise ValueError(f"检测标签 w/h 须大于 0: {path}:{line_number}")


def _validate_segmentation_line(parts: list[str], *, path: Path, line_number: int) -> None:
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        raise ValueError(
            f"分割标签须为 class 加至少 3 个坐标点: {path}:{line_number}"
        )
    _parse_class_id(parts[0], path=path, line_number=line_number)
    _parse_normalized_coordinates(parts[1:], path=path, line_number=line_number)


def validate_yolo_labels(labels_dir: Path, *, task: str) -> int:
    """递归校验全部非空 YOLO 标签行，并返回标注实例数。"""
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"训练标签目录不存在: {labels_dir}")
    txt_files = sorted(labels_dir.rglob("*.txt"))
    if not txt_files:
        raise ValueError(f"训练标签目录中没有 .txt 文件: {labels_dir}")

    validator = {
        "detection": _validate_detection_line,
        "segmentation": _validate_segmentation_line,
    }.get(task)
    if validator is None:
        raise ValueError(f"不支持的标签任务: {task}")

    annotations = 0
    for path in txt_files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                validator(text.split(), path=path, line_number=line_number)
                annotations += 1
    if annotations == 0:
        raise ValueError(f"训练标签中没有有效标注实例: {labels_dir}")
    return annotations


__all__ = ["IMAGE_SUFFIXES", "validate_image_dir", "validate_yolo_labels"]
