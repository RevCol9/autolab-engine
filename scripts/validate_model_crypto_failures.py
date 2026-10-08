#!/usr/bin/env python3
"""Table-driven failure validation for `.niii-model` key rotation."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.model_crypto import ModelCryptoError, load_yolo_container
from toolkit.model_crypto import container_v1
from toolkit.model_crypto.container_v1 import (
    MAGIC,
    ModelContainer,
    read_model_container,
    read_model_key_id,
    write_model_container,
)
from toolkit.model_crypto.envelope import load_kek

CASE_NAMES = (
    "wrong_key",
    "tampered_ciphertext",
    "interrupted_write",
    "disk_full",
    "read_only_directory",
    "cuda_failure",
    "concurrent_write",
)
_FORBIDDEN_SUFFIXES = {".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}


class _SkipCase(RuntimeError):
    pass


@dataclass(frozen=True)
class _Context:
    source: Path
    task: str
    key_dir: Path
    work_dir: Path
    source_key_id: str
    source_kek: bytes
    active_key_id: str
    active_kek: bytes
    source_container: ModelContainer
    source_sha256: str
    device: str
    include_cuda: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_dir(context: _Context, name: str) -> Path:
    path = context.work_dir / name
    path.mkdir()
    return path


def _assert_source_unchanged(context: _Context) -> None:
    if _sha256(context.source) != context.source_sha256:
        raise AssertionError("故障场景修改了源加密模型")
    read_model_container(context.source, kek=context.source_kek)


def _wrong_key(context: _Context) -> str:
    try:
        read_model_container(context.source, kek=os.urandom(32))
    except ModelCryptoError as exc:
        if exc.code == ModelCryptoError.AUTH_FAILED:
            return exc.code
        raise AssertionError(f"错误密钥返回了错误代码: {exc.code}") from exc
    raise AssertionError("错误密钥未被拒绝")


def _tampered_ciphertext(context: _Context) -> str:
    directory = _case_dir(context, "tampered_ciphertext")
    tampered = directory / "tampered.niii-model"
    data = bytearray(context.source.read_bytes())
    data[-1] ^= 1
    tampered.write_bytes(data)
    try:
        read_model_container(tampered, kek=context.source_kek)
    except ModelCryptoError as exc:
        if exc.code == ModelCryptoError.AUTH_FAILED:
            return exc.code
        raise AssertionError(f"篡改返回了错误代码: {exc.code}") from exc
    raise AssertionError("篡改密文未被拒绝")


def _interrupted_write(context: _Context) -> str:
    directory = _case_dir(context, "interrupted_write")
    target = directory / "target.niii-model"
    ready = directory / "child.ready"
    shutil.copy2(context.source, target)
    original_hash = _sha256(target)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_interrupt_child",
        "--src",
        str(context.source),
        "--dst",
        str(target),
        "--keys",
        str(context.key_dir),
        "--ready",
        str(ready),
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 20
    try:
        while not ready.is_file():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"强杀子进程提前退出: {process.returncode}: {stdout}{stderr}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("等待强杀注入点超时")
            time.sleep(0.05)
        process.kill()
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    if _sha256(target) != original_hash:
        raise AssertionError("强杀破坏了旧加密工件")
    read_model_container(target, kek=context.source_kek)
    temporary_files = tuple(directory.glob(f".{target.name}.*.tmp"))
    if not temporary_files:
        raise AssertionError("强杀测试未留下预期的密文临时文件")
    if any(path.read_bytes()[: len(MAGIC)] != MAGIC for path in temporary_files):
        raise AssertionError("强杀留下了非加密临时文件")
    return "old artifact authenticated after process kill"


def _disk_full(context: _Context) -> str:
    directory = _case_dir(context, "disk_full")
    target = directory / "target.niii-model"
    shutil.copy2(context.source, target)
    original_hash = _sha256(target)
    with patch.object(
        container_v1.os,
        "fsync",
        side_effect=OSError(errno.ENOSPC, "injected disk full"),
    ):
        try:
            write_model_container(
                target,
                state_dict=context.source_container.state_dict,
                model_yaml=context.source_container.model_yaml,
                model_names=context.source_container.model_names,
                task=context.source_container.task,
                kek=context.active_kek,
                key_id=context.active_key_id,
                replace_existing=True,
            )
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
        else:
            raise AssertionError("磁盘写满注入未中止写入")
    if _sha256(target) != original_hash:
        raise AssertionError("磁盘写满破坏了旧加密工件")
    read_model_container(target, kek=context.source_kek)
    return "old artifact authenticated after ENOSPC"


def _read_only_directory(context: _Context) -> str:
    if os.name == "nt":
        raise _SkipCase("Windows 不提供可靠的 POSIX 只读目录语义")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise _SkipCase("root 可绕过目录写权限，需使用非 root 账户验收")
    directory = _case_dir(context, "read_only_directory")
    target = directory / "target.niii-model"
    shutil.copy2(context.source, target)
    original_hash = _sha256(target)
    directory.chmod(0o500)
    try:
        try:
            write_model_container(
                target,
                state_dict=context.source_container.state_dict,
                model_yaml=context.source_container.model_yaml,
                model_names=context.source_container.model_names,
                task=context.source_container.task,
                kek=context.active_kek,
                key_id=context.active_key_id,
                replace_existing=True,
            )
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
                raise
        else:
            raise AssertionError("只读目录仍允许写入模型")
    finally:
        directory.chmod(0o700)
    if _sha256(target) != original_hash:
        raise AssertionError("只读目录场景破坏了旧加密工件")
    read_model_container(target, kek=context.source_kek)
    return "old artifact authenticated after read-only failure"


def _cuda_failure(context: _Context) -> str:
    if not context.include_cuda:
        raise _SkipCase("调用方显式跳过 CUDA 故障场景")
    try:
        load_yolo_container(
            context.source,
            context.device,
            context.task,
            key_dir=context.key_dir,
        )
    except ModelCryptoError as exc:
        if exc.code == ModelCryptoError.DEVICE_UNAVAILABLE:
            return exc.code
        raise AssertionError(f"CUDA 故障返回了错误代码: {exc.code}") from exc
    raise AssertionError("无效 CUDA 设备未被拒绝")


def _concurrent_write(context: _Context) -> str:
    directory = _case_dir(context, "concurrent_write")
    destination = directory / "rotated.niii-model"
    script = Path(__file__).with_name("rotate_model_key.py")
    command = [
        sys.executable,
        str(script),
        "--src",
        str(context.source),
        "--dst",
        str(destination),
        "--keys",
        str(context.key_dir),
    ]
    processes = [
        subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=60) for process in processes]
    return_codes = [process.returncode for process in processes]
    if return_codes.count(0) != 1:
        details = [
            {"returnCode": code, "stdout": out, "stderr": err}
            for code, (out, err) in zip(return_codes, results)
        ]
        raise AssertionError(f"并发写入应恰好一个成功: {details}")
    if read_model_key_id(destination) != context.active_key_id:
        raise AssertionError("并发写入结果未使用 active KEK")
    read_model_container(destination, kek=context.active_kek)
    return "exactly one writer published an authenticated artifact"


_CASES = {
    "wrong_key": _wrong_key,
    "tampered_ciphertext": _tampered_ciphertext,
    "interrupted_write": _interrupted_write,
    "disk_full": _disk_full,
    "read_only_directory": _read_only_directory,
    "cuda_failure": _cuda_failure,
    "concurrent_write": _concurrent_write,
}


def _scan_forbidden_artifacts(work_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(work_dir))
        for path in work_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _FORBIDDEN_SUFFIXES
    )


def _publish_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_failure_matrix(
    model: str | Path,
    *,
    task: str,
    key_dir: str | Path,
    work_dir: str | Path,
    device: str = "cuda:999999",
    include_cuda: bool = True,
) -> dict[str, object]:
    """Run every M2 fault case and publish one machine-readable report."""
    source = Path(model).expanduser().resolve()
    keys = Path(key_dir).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve()
    if source.suffix != ".niii-model" or not source.is_file():
        raise ValueError(f"模型必须是已有 .niii-model: {source}")
    if task not in {"detect", "segment"}:
        raise ValueError("task 仅支持 detect 或 segment")
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"故障验收目录必须为空: {work}")
    work.mkdir(parents=True, exist_ok=True)

    source_key_id = read_model_key_id(source)
    source_kek, _ = load_kek(key_dir=keys, key_id=source_key_id)
    active_kek, active_key_id = load_kek(key_dir=keys)
    if active_key_id == source_key_id:
        raise ValueError("请先 rotate-kek；故障验收要求 active KEK 与源模型 KEK 不同")
    source_container = read_model_container(source, kek=source_kek)
    if source_container.task != task:
        raise ValueError(
            f"模型任务不匹配: 期望 {task}，实际 {source_container.task}"
        )
    context = _Context(
        source=source,
        task=task,
        key_dir=keys,
        work_dir=work,
        source_key_id=source_key_id,
        source_kek=source_kek,
        active_key_id=active_key_id,
        active_kek=active_kek,
        source_container=source_container,
        source_sha256=_sha256(source),
        device=device,
        include_cuda=include_cuda,
    )
    report: dict[str, object] = {
        "schemaVersion": 1,
        "status": "running",
        "startedAt": _utc_now(),
        "model": {
            "path": str(source),
            "sha256": context.source_sha256,
            "keyId": source_key_id,
        },
        "activeKeyId": active_key_id,
        "cases": {},
    }
    case_results: dict[str, dict[str, str]] = {}
    for name in CASE_NAMES:
        try:
            detail = _CASES[name](context)
            _assert_source_unchanged(context)
            case_results[name] = {"status": "passed", "detail": detail}
        except _SkipCase as exc:
            case_results[name] = {"status": "skipped", "detail": str(exc)}
        except Exception as exc:
            case_results[name] = {
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
            }
    matches = _scan_forbidden_artifacts(work)
    report["cases"] = case_results
    report["artifactScan"] = {
        "scope": str(work),
        "forbiddenSuffixes": sorted(_FORBIDDEN_SUFFIXES),
        "matches": matches,
    }
    report["finishedAt"] = _utc_now()
    report["status"] = (
        "passed"
        if not matches and all(item["status"] != "failed" for item in case_results.values())
        else "failed"
    )
    _publish_report(work / "failure-report.json", report)
    return report


def _interrupt_child(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args(argv)

    source = Path(args.src)
    target = Path(args.dst)
    source_key_id = read_model_key_id(source)
    source_kek, _ = load_kek(key_dir=args.keys, key_id=source_key_id)
    active_kek, active_key_id = load_kek(key_dir=args.keys)
    container = read_model_container(source, kek=source_kek)

    def pause_before_publish(_source: object, _destination: object) -> None:
        Path(args.ready).write_text("ready\n", encoding="utf-8")
        while True:
            time.sleep(1)

    with patch.object(container_v1.os, "replace", side_effect=pause_before_publish):
        write_model_container(
            target,
            state_dict=container.state_dict,
            model_yaml=container.model_yaml,
            model_names=container.model_names,
            task=container.task,
            kek=active_kek,
            key_id=active_key_id,
            replace_existing=True,
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行模型加密 M2 故障注入矩阵")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda:999999")
    parser.add_argument("--skip-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_failure_matrix(
        args.model,
        task=args.task,
        key_dir=args.keys,
        work_dir=args.work_dir,
        device=args.device,
        include_cuda=not args.skip_cuda,
    )
    report_path = Path(args.work_dir).expanduser().resolve() / "failure-report.json"
    print(f"故障注入 {report['status']}，报告: {report_path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_interrupt_child":
        raise SystemExit(_interrupt_child(sys.argv[2:]))
    raise SystemExit(main())
