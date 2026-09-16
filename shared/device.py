"""计算设备字符串的统一解析规则。"""

from __future__ import annotations

import re


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


__all__ = ["parse_device_index"]
