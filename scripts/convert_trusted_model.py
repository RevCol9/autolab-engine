#!/usr/bin/env python3
"""Convert one trusted YOLO ``.pt`` checkpoint on an isolated build host."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from toolkit.model_crypto.container_v1 import (
    read_model_container,
    write_model_container,
)
from toolkit.model_crypto.envelope import load_kek


def convert_trusted_pt(
    source: str | Path,
    destination: str | Path,
    *,
    task: str,
    key_dir: str | Path | None = None,
) -> Path:
    """Convert a reviewed local checkpoint without creating plaintext output."""
    source = Path(source)
    destination = Path(destination)
    if source.suffix.lower() != ".pt" or not source.is_file():
        raise ValueError(f"输入必须是可信的本地 .pt 文件: {source}")
    if destination.suffix != ".niii-model":
        raise ValueError("输出文件必须以 .niii-model 结尾")
    if source.resolve() == destination.resolve():
        raise ValueError("输入与输出不能是同一文件")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("输入不是 YOLO checkpoint 字典")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None or not hasattr(model, "state_dict"):
        raise ValueError("输入不包含 YOLO model/ema")
    model_task = getattr(model, "task", None)
    if model_task and model_task != task:
        raise ValueError(f"模型任务 {model_task!r} 与目标任务 {task!r} 不一致")
    yaml_config = getattr(model, "yaml", None)
    names = getattr(model, "names", None)
    if not isinstance(yaml_config, dict) or not isinstance(names, (dict, list, tuple)):
        raise ValueError("YOLO checkpoint 缺少 model.yaml 或 model.names")
    if not isinstance(names, dict):
        names = dict(enumerate(names))

    kek, key_id = load_kek(key_dir=key_dir)
    result = write_model_container(
        destination,
        state_dict=model.state_dict(),
        model_yaml=yaml_config,
        model_names=names,
        task=task,
        kek=kek,
        key_id=key_id,
    )
    verified = read_model_container(result, kek=kek)
    if verified.task != task or set(verified.state_dict) != set(model.state_dict()):
        raise RuntimeError("加密模型写后验证失败")
    for name, original in model.state_dict().items():
        original_bytes = original.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        restored = verified.state_dict[name]
        restored_bytes = restored.reshape(-1).view(torch.uint8)
        if original.dtype != restored.dtype or not torch.equal(
            original_bytes,
            restored_bytes,
        ):
            raise RuntimeError(f"加密模型写后验证失败: {name}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="隔离构建机：将可信 YOLO .pt 转换为 .niii-model"
    )
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--trust-source-pt", action="store_true")
    args = parser.parse_args(argv)
    if not args.trust_source_pt:
        parser.error("必须显式指定 --trust-source-pt")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = convert_trusted_pt(
        args.src,
        args.dst,
        task=args.task,
        key_dir=args.keys,
    )
    print(f"已生成并验证加密模型: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
