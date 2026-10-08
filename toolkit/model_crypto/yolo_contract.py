"""Shared names for the supported YOLO architecture and task contract."""

from __future__ import annotations

import re
from dataclasses import dataclass

FAMILY_FOLDERS = {
    "yolov8": "v8",
    "yolo11": "11",
    "yolo26": "26",
}
SCALES = frozenset("nsmlx")
TASK_SUFFIXES = {"detect": "", "segment": "-seg"}
TRAINING_TO_CONTAINER_TASK = {
    "detection": "detect",
    "segmentation": "segment",
}

_CONTROLLED_ARCHITECTURE = re.compile(
    r"^(yolov8|yolo11|yolo26)([nsmlx])(-seg)?\.yaml$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ControlledArchitecture:
    name: str
    family: str
    scale: str
    task: str


def container_task_for_training(task: str) -> str:
    normalized = str(task).strip().lower()
    try:
        return TRAINING_TO_CONTAINER_TASK[normalized]
    except KeyError as exc:
        raise ValueError(f"不支持的训练任务: {task!r}") from exc


def normalize_container_task(task: str) -> str:
    normalized = str(task).strip().lower()
    if normalized in TASK_SUFFIXES:
        return normalized
    return container_task_for_training(normalized)


def parse_controlled_architecture(
    model_name: str,
    *,
    expected_task: str | None = None,
) -> ControlledArchitecture:
    normalized = str(model_name).strip().lower()
    match = _CONTROLLED_ARCHITECTURE.fullmatch(normalized)
    if match is None:
        raise ValueError("只允许受控 YOLOv8/YOLO11/YOLO26 架构 YAML")
    family, scale, segment_suffix = match.groups()
    task = "segment" if segment_suffix else "detect"
    if expected_task is not None and task != expected_task:
        raise ValueError(
            f"模型架构与任务不匹配: {model_name!r} / {expected_task!r}"
        )
    return ControlledArchitecture(
        name=normalized,
        family=family.lower(),
        scale=scale.lower(),
        task=task,
    )


__all__ = [
    "ControlledArchitecture",
    "FAMILY_FOLDERS",
    "SCALES",
    "TASK_SUFFIXES",
    "TRAINING_TO_CONTAINER_TASK",
    "container_task_for_training",
    "normalize_container_task",
    "parse_controlled_architecture",
]
