"""Fail-closed, pickle-free runtime loading for authenticated YOLO models."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, SegmentationModel
from ultralytics.utils import ROOT, YAML

from toolkit.model_crypto.container_v1 import ModelContainer, read_model_container
from toolkit.model_crypto.envelope import load_kek

_SCALES = frozenset("nsmlx")
_FAMILIES = (
    ("yolov8", "v8"),
    ("yolo11", "11"),
    ("yolo26", "26"),
)
_TASKS = {
    "detect": ("", DetectionModel),
    "segment": ("-seg", SegmentationModel),
}


class EncryptedYOLO(YOLO):
    """YOLO inference facade with plaintext model writers disabled."""

    def train(self, *args, **kwargs):
        raise RuntimeError("EncryptedYOLO 不允许调用默认 train()；请使用加密训练 adapter")

    def save(self, *args, **kwargs):
        raise RuntimeError("EncryptedYOLO 不允许保存明文 checkpoint")

    def export(self, *args, **kwargs):
        raise RuntimeError("EncryptedYOLO 不允许导出明文模型")

    def tune(self, *args, **kwargs):
        raise RuntimeError("EncryptedYOLO 不允许调用默认 tune()")

    def benchmark(self, *args, **kwargs):
        raise RuntimeError("EncryptedYOLO 不允许调用默认 benchmark()")

    def load(self, *args, **kwargs):
        raise RuntimeError("EncryptedYOLO 不允许加载明文 checkpoint")


def _trusted_configuration(
    container: ModelContainer,
    expected_task: str,
) -> tuple[Path, dict[str, Any], type[DetectionModel]]:
    if expected_task not in _TASKS:
        raise ValueError(f"不支持的 YOLO 任务: {expected_task}")
    if container.task != expected_task:
        raise ValueError(f"加密模型任务不匹配: 期望 {expected_task}，实际 {container.task}")

    nc = container.model_yaml.get("nc")
    if type(nc) is not int or not 1 <= nc <= 1024:
        raise ValueError("加密模型类别数量 nc 非法")
    if set(container.model_names) != set(range(nc)):
        raise ValueError("加密模型类别 ID 必须恰好为 0..nc-1")
    forbidden_label_chars = frozenset('/\\<>:"|?*')
    for name in container.model_names.values():
        if (
            len(name) > 128
            or name in {".", ".."}
            or name.strip() != name
            or name.endswith(".")
            or any(char in forbidden_label_chars or ord(char) < 32 for char in name)
        ):
            raise ValueError("加密模型类别名称包含不安全的路径字符")

    scale = container.model_yaml.get("scale")
    if not isinstance(scale, str) or scale not in _SCALES:
        raise ValueError(f"不支持的 YOLO 模型规模: {scale}")
    suffix, model_type = _TASKS[expected_task]
    source_yaml = dict(container.model_yaml)
    source_filename = source_yaml.get("yaml_file", "")
    if not isinstance(source_filename, str) or len(source_filename) > 1024:
        raise ValueError("加密模型 yaml_file 元数据非法")
    for family, folder in _FAMILIES:
        trusted_path = ROOT / "cfg" / "models" / folder / f"{family}{suffix}.yaml"
        if not trusted_path.is_file():
            continue
        trusted_yaml = YAML.load(trusted_path)
        trusted_yaml.update(
            nc=nc,
            scale=scale,
            yaml_file=f"{family}{scale}{suffix}.yaml",
            channels=3,
        )
        # Build-host paths are descriptive metadata, never executable runtime paths.
        source_yaml["yaml_file"] = trusted_yaml["yaml_file"]
        if source_yaml == trusted_yaml:
            return trusted_path, trusted_yaml, model_type
    raise ValueError("加密模型架构与受控 YOLOv8/YOLO11/YOLO26 模板不一致")


def load_yolo_container(
    path: str | Path,
    device: str | torch.device,
    expected_task: str,
    *,
    key_dir: str | Path | None = None,
    allow_cpu_for_tests: bool = False,
) -> YOLO:
    """Authenticate a .niii-model and load its tensors without any plaintext file.

    CPU is available only with an explicit development-test opt-in.
    Only bundled YOLOv8, YOLO11, and YOLO26 detect/segment architectures are accepted.
    """
    path = Path(path)
    if path.suffix != ".niii-model":
        raise ValueError("运行时只接受 .niii-model 模型文件")

    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError(f"不支持的模型设备: {target_device}")
    if target_device.type == "cpu" and not allow_cpu_for_tests:
        raise RuntimeError("首版加密模型运行时必须使用 CUDA；CPU 仅限显式测试模式")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 设备不可用，不回退到 CPU")

    kek, key_id = load_kek(key_dir=key_dir, model_path=path)
    container = read_model_container(path, kek=kek)
    if container.key_id != key_id:
        raise ValueError(f"加密模型 key_id 不匹配: {container.key_id}")
    trusted_path, trusted_yaml, model_type = _trusted_configuration(container, expected_task)

    # An absolute package YAML path avoids cwd shadowing or implicit .pt downloads.
    yolo = EncryptedYOLO(str(trusted_path), task=expected_task, verbose=False)
    if trusted_yaml["scale"] == "n" and trusted_yaml["nc"] == yolo.model.yaml["nc"]:
        network = yolo.model
        network.yaml = deepcopy(trusted_yaml)
    else:
        previous_args = yolo.model.args
        yolo.model = None
        network = model_type(cfg=deepcopy(trusted_yaml), verbose=False)
        network.args = previous_args
    network.task = expected_task
    network.names = dict(container.model_names)

    expected_state = network.state_dict()
    if set(container.state_dict) != set(expected_state):
        raise ValueError("加密模型 state_dict 键与架构不匹配")
    for name, tensor in container.state_dict.items():
        reference = expected_state[name]
        if tensor.shape != reference.shape:
            raise ValueError(f"加密模型 tensor shape 不匹配: {name}")
        if reference.is_floating_point():
            if not tensor.is_floating_point():
                raise ValueError(f"加密模型 tensor dtype 不匹配: {name}")
        elif tensor.dtype != reference.dtype:
            raise ValueError(f"加密模型 tensor dtype 不匹配: {name}")
    network.load_state_dict(container.state_dict, strict=True, assign=True)
    yolo.model = network
    yolo.model_name = str(path)
    yolo.overrides["model"] = str(path)
    yolo.to(target_device)
    return yolo


__all__ = ["EncryptedYOLO", "load_yolo_container"]
