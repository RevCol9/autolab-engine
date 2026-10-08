#!/usr/bin/env python3
"""Validate one encrypted YOLO task on a real CUDA device.

This is an acceptance program, not a CPU-compatible unit test. Runtime input
validation, missing CUDA, failed evaluation, missing encrypted checkpoints, or
plaintext model artifacts all produce a non-zero exit status and a JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import ultralytics

from toolkit.model_crypto import load_yolo_container
from training.encrypted_trainer import create_encrypted_trainer

_TASKS = {
    "detect": "detection",
    "segment": "segmentation",
}
_FORBIDDEN_MODEL_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_cuda_device(value: str) -> torch.device:
    text = str(value).strip().lower()
    if text == "cuda":
        text = "cuda:0"
    elif text.isdigit():
        text = f"cuda:{text}"
    device = torch.device(text)
    if device.type != "cuda" or device.index is None:
        raise ValueError("真机验收只接受 cuda:N 或数字形式的单 CUDA 设备")
    return device


def _require_cuda(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；真机验收不允许跳过或回退 CPU")
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA 设备越界: {device}，当前仅检测到 {torch.cuda.device_count()} 张卡"
        )


def _require_file(path: Path, label: str, suffix: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}不存在: {resolved}")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ValueError(f"{label}必须是 {suffix} 文件: {resolved}")
    return resolved


def _prepare_run_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"run-dir 不是目录: {resolved}")
    if resolved.is_dir() and any(resolved.iterdir()):
        raise ValueError(f"run-dir 必须不存在或为空，避免污染验收证据: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sizeBytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_to_json_value(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scan_plaintext_artifacts(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        str(path.resolve())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _FORBIDDEN_MODEL_SUFFIXES
    )


def _cuda_memory(device: torch.device) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "freeBytes": int(free_bytes),
        "totalBytes": int(total_bytes),
        "allocatedBytes": int(torch.cuda.memory_allocated(device)),
        "reservedBytes": int(torch.cuda.memory_reserved(device)),
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _result_summary(result: Any) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    return {
        "boxCount": 0 if boxes is None else len(boxes),
        "maskCount": 0 if masks is None else len(masks.data),
        "speedMs": _to_json_value(getattr(result, "speed", {})),
    }


def _run_validation(args: argparse.Namespace, report: dict[str, Any]) -> None:
    device = _normalize_cuda_device(args.device)
    _require_cuda(device)
    torch.cuda.set_device(device)

    model_path = _require_file(args.model, "加密模型", ".niii-model")
    data_path = _require_file(args.data, "数据配置", ".yaml")
    image_path = _require_file(args.image, "推理样例")
    key_dir = args.keys.expanduser().resolve() if args.keys else None
    if key_dir is not None and not key_dir.is_dir():
        raise FileNotFoundError(f"密钥目录不存在: {key_dir}")
    run_dir = _prepare_run_dir(args.run_dir)

    properties = torch.cuda.get_device_properties(device)
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchCuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "device": str(device),
        "deviceName": properties.name,
        "computeCapability": f"{properties.major}.{properties.minor}",
        "totalVramBytes": int(properties.total_memory),
        "cudnn": torch.backends.cudnn.version(),
    }
    report["inputs"] = {
        "task": args.task,
        "model": _artifact_record(model_path),
        "data": str(data_path),
        "image": str(image_path),
        "runDir": str(run_dir),
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    model = load_yolo_container(
        model_path,
        device,
        args.task,
        key_dir=key_dir,
    )
    torch.cuda.synchronize(device)
    report["load"] = {
        "durationSec": round(time.perf_counter() - started, 6),
        "memory": _cuda_memory(device),
        "parameterDevice": str(next(model.model.parameters()).device),
    }

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=str(device),
        save=False,
        verbose=False,
    )
    torch.cuda.synchronize(device)
    if not prediction:
        raise RuntimeError("CUDA 推理没有返回结果")
    report["prediction"] = {
        "durationSec": round(time.perf_counter() - started, 6),
        "memory": _cuda_memory(device),
        "result": _result_summary(prediction[0]),
    }

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    baseline = model.val(
        data=str(data_path),
        batch=args.batch,
        imgsz=args.imgsz,
        device=str(device),
        project=str(run_dir),
        name="baseline_eval",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    torch.cuda.synchronize(device)
    report["baselineEvaluation"] = {
        "durationSec": round(time.perf_counter() - started, 6),
        "memory": _cuda_memory(device),
        "metrics": _to_json_value(getattr(baseline, "results_dict", {})),
    }

    if not args.skip_training:
        torch.cuda.reset_peak_memory_stats(device)
        trainer = create_encrypted_trainer(
            model.model,
            task=_TASKS[args.task],
            key_dir=key_dir,
            overrides={
                "model": str(model_path),
                "data": str(data_path),
                "project": str(run_dir.parent),
                "name": run_dir.name,
                "exist_ok": True,
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "device": str(device),
                "workers": args.workers,
                "amp": args.amp,
                "plots": False,
                "save": True,
                "save_period": -1,
                "pretrained": False,
                "val": True,
                "verbose": False,
            },
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        trainer.train()
        torch.cuda.synchronize(device)

        artifacts: dict[str, Any] = {}
        for name in ("best", "last"):
            path = trainer.wdir / f"{name}.niii-model"
            if not path.is_file():
                raise RuntimeError(f"CUDA 训练未生成 {path.name}")
            restored = load_yolo_container(
                path,
                device,
                args.task,
                key_dir=key_dir,
            )
            parameter_device = str(next(restored.model.parameters()).device)
            if not parameter_device.startswith("cuda"):
                raise RuntimeError(f"{path.name} 未重新加载到 CUDA: {parameter_device}")
            artifacts[name] = {
                **_artifact_record(path),
                "parameterDevice": parameter_device,
            }
        report["training"] = {
            "durationSec": round(time.perf_counter() - started, 6),
            "epochs": args.epochs,
            "memory": _cuda_memory(device),
            "metrics": _to_json_value(getattr(trainer, "metrics", {})),
            "artifacts": artifacts,
        }

    forbidden = _scan_plaintext_artifacts(run_dir)
    report["artifactScan"] = {
        "scope": str(run_dir),
        "forbiddenSuffixes": sorted(_FORBIDDEN_MODEL_SUFFIXES),
        "matches": forbidden,
        "eventMonitoring": False,
        "note": "这是结束扫描；全过程文件事件证据由后续独立监控程序提供。",
    }
    if forbidden:
        raise RuntimeError(f"验收目录发现明文模型工件: {forbidden}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在真实 CUDA 上验收一个 `.niii-model` YOLO 自动标注任务"
    )
    parser.add_argument("--task", choices=tuple(_TASKS), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="默认写入 RUN_DIR 同级的 RUN_DIR-cuda-validation-report.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0 or args.workers < 0:
        raise SystemExit("epochs/batch/imgsz 必须为正数，workers 不能为负数")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("conf 必须位于 [0, 1]")

    report_path = args.report or args.run_dir.with_name(
        f"{args.run_dir.name}-cuda-validation-report.json"
    )
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "startedAt": _utc_now(),
    }
    try:
        _run_validation(args, report)
    except Exception as exc:
        report.update(
            status="failed",
            finishedAt=_utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        _write_report(report_path, report)
        print(f"CUDA 验收失败: {exc}", file=sys.stderr)
        print(f"报告: {report_path.resolve()}", file=sys.stderr)
        return 1

    report.update(status="passed", finishedAt=_utc_now())
    _write_report(report_path, report)
    print(f"CUDA 验收通过，报告: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
