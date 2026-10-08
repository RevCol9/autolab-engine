#!/usr/bin/env python3
"""Re-encrypt one `.niii-model` with the active KEK, keeping the source intact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from toolkit.model_crypto.container_v1 import (
    ModelContainer,
    read_model_container,
    read_model_key_id,
    write_model_container,
)
from toolkit.model_crypto.envelope import load_kek


def _same_container(left: ModelContainer, right: ModelContainer) -> bool:
    if (
        left.task != right.task
        or left.model_yaml != right.model_yaml
        or left.model_names != right.model_names
        or set(left.state_dict) != set(right.state_dict)
    ):
        return False
    for name, original in left.state_dict.items():
        restored = right.state_dict[name]
        if original.dtype != restored.dtype or original.shape != restored.shape:
            return False
        if not torch.equal(
            original.contiguous().reshape(-1).view(torch.uint8),
            restored.contiguous().reshape(-1).view(torch.uint8),
        ):
            return False
    return True


def rotate_model_key(
    source: str | Path,
    destination: str | Path,
    *,
    key_dir: str | Path,
) -> Path:
    """Authenticate, decrypt in memory, and write a separately versioned artifact."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if source.suffix != ".niii-model" or not source.is_file():
        raise ValueError(f"源文件必须是已有 .niii-model: {source}")
    if destination.suffix != ".niii-model":
        raise ValueError("目标文件必须以 .niii-model 结尾")
    if source == destination:
        raise ValueError("轮换不能原地覆盖源工件")
    if destination.exists():
        raise FileExistsError(f"轮换目标已存在，拒绝覆盖: {destination}")

    source_key_id = read_model_key_id(source)
    source_kek, loaded_source_id = load_kek(
        key_dir=key_dir,
        key_id=source_key_id,
    )
    active_kek, active_key_id = load_kek(key_dir=key_dir)
    if loaded_source_id != source_key_id:
        raise RuntimeError("源模型 KEK 标识解析不一致")
    if active_key_id == source_key_id:
        raise ValueError(f"源模型已经使用 active KEK: {active_key_id}")

    original = read_model_container(source, kek=source_kek)
    result = write_model_container(
        destination,
        state_dict=original.state_dict,
        model_yaml=original.model_yaml,
        model_names=original.model_names,
        task=original.task,
        kek=active_kek,
        key_id=active_key_id,
    )
    try:
        verified = read_model_container(result, kek=active_kek)
        if verified.key_id != active_key_id or not _same_container(original, verified):
            raise RuntimeError("轮换后模型逐 tensor 验证失败")
    except Exception:
        result.unlink(missing_ok=True)
        raise
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 .niii-model 在内存中重新加密为 active KEK 的新工件"
    )
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--keys", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = rotate_model_key(args.src, args.dst, key_dir=args.keys)
    print(f"轮换完成: {result} (key_id={read_model_key_id(result)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
