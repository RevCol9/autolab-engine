"""训练后端注册表。"""

from training.backends.base import TrainBackend
from training.backends.detection import BACKEND as DETECTION_BACKEND
from training.backends.segmentation import BACKEND as SEGMENTATION_BACKEND

_BACKENDS: dict[str, TrainBackend] = {
    DETECTION_BACKEND.task: DETECTION_BACKEND,
    SEGMENTATION_BACKEND.task: SEGMENTATION_BACKEND,
}


def get_backend(task: str) -> TrainBackend:
    key = (task or "detection").strip().lower()
    backend = _BACKENDS.get(key)
    if backend is None:
        raise ValueError(f"未知训练任务: {task!r}，已注册: {sorted(_BACKENDS)}")
    return backend


def registered_tasks() -> list[str]:
    return sorted(_BACKENDS)
