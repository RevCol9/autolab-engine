"""计算设备字符串的统一解析规则。"""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Optional


_DEVICE_INDEX_RE = re.compile(r"(?:cuda:)?(\d+)$", re.IGNORECASE)
_NON_CUDA_DEVICES = frozenset({"cpu", "mps"})


def parse_device_index(device: str | int | None) -> str:
    """解析单设备标识；当前调度器明确不支持多卡训练字符串。"""
    text = str(device if device is not None else "0").strip().lower() or "0"
    if "," in text:
        raise ValueError("当前训练调度仅支持单设备，不支持多 GPU device 列表")
    if text == "cuda":
        return "0"
    if text in _NON_CUDA_DEVICES:
        return text
    match = _DEVICE_INDEX_RE.fullmatch(text)
    if match:
        return match.group(1)
    raise ValueError(f"不支持的 device: {device!r}；请使用 0、cuda:0、cpu 或 mps")


def is_cuda_device(device: str | int | None) -> bool:
    """设备是否指向 CUDA；同时复用单设备格式校验。"""
    return parse_device_index(device) not in _NON_CUDA_DEVICES


def physical_cuda_device(
    device: str | int | None,
    *,
    visible_devices: Optional[str] = None,
) -> str:
    """把进程内 CUDA 逻辑序号映射为跨进程一致的物理设备标识。"""
    index = parse_device_index(device)
    if index in _NON_CUDA_DEVICES:
        return index

    raw = visible_devices
    if raw is None:
        import os

        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    text = str(raw or "").strip()
    if not text:
        return index
    if text in {"-1", "NoDevFiles"}:
        raise ValueError("CUDA_VISIBLE_DEVICES 未暴露可用 GPU")

    devices = [part.strip() for part in text.split(",") if part.strip()]
    logical_index = int(index)
    if logical_index >= len(devices):
        raise ValueError(
            f"device={device!r} 超出 CUDA_VISIBLE_DEVICES={text!r} 的逻辑设备范围"
        )
    return devices[logical_index]


def device_lock_key(
    device: str | int | None,
    *,
    visible_devices: Optional[str] = None,
) -> str:
    """返回可安全用于锁文件名的物理设备键。"""
    physical = physical_cuda_device(device, visible_devices=visible_devices)
    if physical in _NON_CUDA_DEVICES or physical.isdigit():
        return physical
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", physical).strip("-._")[:32] or "device"
    digest = sha256(physical.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


__all__ = [
    "device_lock_key",
    "is_cuda_device",
    "parse_device_index",
    "physical_cuda_device",
]
