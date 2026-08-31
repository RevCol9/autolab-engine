"""YAML 配置加载与浅层合并（后写覆盖先写）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须为 mapping: {path}")
    return data


def merge_yaml_dicts(*parts: Mapping[str, Any]) -> Dict[str, Any]:
    """合并多个 mapping；同名顶层键由后者覆盖。"""
    merged: Dict[str, Any] = {}
    for part in parts:
        merged.update(dict(part))
    return merged


def load_merged_yaml(paths: Iterable[Path]) -> Dict[str, Any]:
    files = [Path(p).expanduser().resolve() for p in paths]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"配置文件不存在: {', '.join(missing)}")
    return merge_yaml_dicts(*(load_yaml_file(p) for p in files))
