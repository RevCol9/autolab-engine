"""训练数据集布局、校验与 Ultralytics YAML 生成。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import yaml

from training.paths import STORAGE_ROOT, classes_txt_path, safe_id

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
SUPPORTED_TASKS = frozenset({"detection", "segmentation"})


def read_class_names(classes_txt: Path) -> tuple[str, ...]:
    if not classes_txt.is_file():
        raise FileNotFoundError(f"classes.txt 不存在: {classes_txt}")
    names = tuple(
        line.strip()
        for line in classes_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not names:
        raise ValueError(f"classes.txt 不能为空: {classes_txt}")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"classes.txt 存在重复类别: {duplicates}")
    return names


def validate_image_dir(images_dir: Path) -> int:
    """确认图像目录存在且至少包含一张 Ultralytics 支持的图像。"""
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


def _parse_class_id(
    token: str,
    *,
    path: Path,
    line_number: int,
    class_count: int,
) -> int:
    try:
        class_id = int(token)
    except ValueError as exc:
        raise ValueError(f"标签类别须为整数: {path}:{line_number}") from exc
    if class_id < 0 or class_id >= class_count:
        raise ValueError(
            f"标签类别 {class_id} 超出 classes.txt 范围 0..{class_count - 1}: "
            f"{path}:{line_number}"
        )
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


def _validate_label_line(
    parts: list[str],
    *,
    task: str,
    path: Path,
    line_number: int,
    class_count: int,
) -> None:
    if task == "detection":
        if len(parts) != 5:
            raise ValueError(f"检测标签须为 5 列（class cx cy w h）: {path}:{line_number}")
    elif len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        raise ValueError(f"分割标签须为 class 加至少 3 个坐标点: {path}:{line_number}")

    _parse_class_id(
        parts[0],
        path=path,
        line_number=line_number,
        class_count=class_count,
    )
    coords = _parse_normalized_coordinates(parts[1:], path=path, line_number=line_number)
    if task == "detection" and (coords[2] <= 0.0 or coords[3] <= 0.0):
        raise ValueError(f"检测标签 w/h 须大于 0: {path}:{line_number}")


def validate_yolo_labels(labels_dir: Path, *, task: str, class_count: int) -> int:
    """递归校验全部非空 YOLO 标签行，并返回标注实例数。"""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的标签任务: {task}")
    if class_count <= 0:
        raise ValueError("class_count 须为正整数")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"训练标签目录不存在: {labels_dir}")
    txt_files = sorted(labels_dir.rglob("*.txt"))
    if not txt_files:
        raise ValueError(f"训练标签目录中没有 .txt 文件: {labels_dir}")

    annotations = 0
    for path in txt_files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                _validate_label_line(
                    text.split(),
                    task=task,
                    path=path,
                    line_number=line_number,
                    class_count=class_count,
                )
                annotations += 1
    if annotations == 0:
        raise ValueError(f"训练标签中没有有效标注实例: {labels_dir}")
    return annotations


@dataclass(frozen=True)
class TrainingDataset:
    project_id: str
    task_id: str
    train_num: str
    root: Path
    images_dir: Path
    labels_dir: Path
    classes_txt: Path
    data_yaml: Path
    images_relative: str

    @classmethod
    def from_job(cls, param: Mapping[str, Any]) -> "TrainingDataset":
        project_id = safe_id("projectId", param["projectId"])
        task_id = safe_id("taskId", param["taskId"])
        train_num = safe_id("trainNum", param.get("trainNum") or "train1")
        class_file = classes_txt_path(project_id, task_id)
        if project_id == "algorithms":
            root = STORAGE_ROOT / "algorithms" / task_id
            images_relative = f"{train_num}/images"
            run_root = root / train_num
        else:
            root = STORAGE_ROOT / project_id
            images_relative = "images"
            run_root = root
        return cls(
            project_id=project_id,
            task_id=task_id,
            train_num=train_num,
            root=root,
            images_dir=run_root / "images",
            labels_dir=run_root / "labels",
            classes_txt=class_file,
            data_yaml=class_file.with_suffix(".yaml"),
            images_relative=images_relative,
        )

    def prepare(self, *, task: str) -> Path:
        """先完整校验数据契约，再原子写出 Ultralytics data YAML。"""
        names = read_class_names(self.classes_txt)
        validate_image_dir(self.images_dir)
        validate_yolo_labels(self.labels_dir, task=task, class_count=len(names))
        data = {
            "path": str(self.root),
            "train": self.images_relative,
            "val": self.images_relative,
            "names": {index: name for index, name in enumerate(names)},
        }
        self.data_yaml.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.data_yaml.with_name(f".{self.data_yaml.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
            temporary.replace(self.data_yaml)
        finally:
            temporary.unlink(missing_ok=True)
        return self.data_yaml


__all__ = [
    "IMAGE_SUFFIXES",
    "TrainingDataset",
    "read_class_names",
    "validate_image_dir",
    "validate_yolo_labels",
]
