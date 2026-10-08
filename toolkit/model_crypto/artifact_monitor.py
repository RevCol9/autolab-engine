"""Process-external Linux inotify evidence for plaintext model artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import selectors
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)
DEFAULT_MAX_EVENT_LOG_BYTES = 256 * 1024 * 1024
MAX_RECORDED_FORBIDDEN_PATHS = 1_000
MAX_RECORDED_INTEGRITY_FAILURES = 100

_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_EXCL_UNLINK = 0x04000000
_IN_ISDIR = 0x40000000

_WATCH_MASK = (
    _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_EXCL_UNLINK
)
_EVENT_STRUCT = struct.Struct("=iIII")
_EVENT_NAMES = (
    (_IN_ATTRIB, "ATTRIB"),
    (_IN_CLOSE_WRITE, "CLOSE_WRITE"),
    (_IN_MOVED_FROM, "MOVED_FROM"),
    (_IN_MOVED_TO, "MOVED_TO"),
    (_IN_CREATE, "CREATE"),
    (_IN_DELETE, "DELETE"),
    (_IN_DELETE_SELF, "DELETE_SELF"),
    (_IN_MOVE_SELF, "MOVE_SELF"),
    (_IN_UNMOUNT, "UNMOUNT"),
    (_IN_Q_OVERFLOW, "Q_OVERFLOW"),
    (_IN_IGNORED, "IGNORED"),
    (_IN_ISDIR, "ISDIR"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _normalize_suffixes(values: Iterable[str]) -> tuple[str, ...]:
    normalized = set()
    for value in values:
        suffix = str(value).strip().lower()
        if not suffix:
            raise ValueError("禁止后缀不能为空")
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if any(char in suffix for char in "/\\\x00"):
            raise ValueError(f"禁止后缀非法: {value!r}")
        normalized.add(suffix)
    if not normalized:
        raise ValueError("至少需要一个禁止模型后缀")
    return tuple(sorted(normalized))


def _is_forbidden_path(path: Path, suffixes: tuple[str, ...]) -> bool:
    """Match final and common temporary forms such as ``best.pt.tmp``."""
    name = path.name.lower()
    return any(name.endswith(suffix) or f"{suffix}." in name for suffix in suffixes)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _event_names(mask: int) -> list[str]:
    return [name for bit, name in _EVENT_NAMES if mask & bit]


def _decode_inotify_buffer(data: bytes) -> list[tuple[int, int, int, str]]:
    events = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < _EVENT_STRUCT.size:
            raise RuntimeError("inotify 返回了截断的事件头")
        watch_descriptor, mask, cookie, name_length = _EVENT_STRUCT.unpack_from(
            data,
            offset,
        )
        offset += _EVENT_STRUCT.size
        end = offset + name_length
        if end > len(data):
            raise RuntimeError("inotify 返回了截断的事件名称")
        raw_name = data[offset:end].split(b"\x00", 1)[0]
        name = os.fsdecode(raw_name)
        events.append((watch_descriptor, mask, cookie, name))
        offset = end
    return events


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有监控工件: {path}")
        # Publish without an overwrite race. os.replace() could silently replace
        # evidence created by another monitoring session after the exists check.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArtifactMonitorConfig:
    """Validated configuration for one independent monitor process."""

    roots: tuple[Path, ...]
    event_log: Path
    summary: Path
    forbidden_suffixes: tuple[str, ...] = DEFAULT_FORBIDDEN_SUFFIXES
    ready_file: Path | None = None
    stop_file: Path | None = None
    lock_file: Path | None = None
    parent_pid: int | None = None
    poll_interval_sec: float = 0.25
    max_event_log_bytes: int = DEFAULT_MAX_EVENT_LOG_BYTES

    @classmethod
    def build(
        cls,
        *,
        roots: Iterable[str | Path],
        event_log: str | Path,
        summary: str | Path,
        forbidden_suffixes: Iterable[str] = DEFAULT_FORBIDDEN_SUFFIXES,
        ready_file: str | Path | None = None,
        stop_file: str | Path | None = None,
        lock_file: str | Path | None = None,
        parent_pid: int | None = None,
        poll_interval_sec: float = 0.25,
        max_event_log_bytes: int = DEFAULT_MAX_EVENT_LOG_BYTES,
    ) -> ArtifactMonitorConfig:
        resolved_roots = tuple(
            dict.fromkeys(Path(root).expanduser().resolve() for root in roots)
        )
        if not resolved_roots:
            raise ValueError("至少需要一个监控根目录")
        for root in resolved_roots:
            if not root.is_dir():
                raise FileNotFoundError(f"监控根目录不存在或不是目录: {root}")
        if not 0.01 <= poll_interval_sec <= 60.0:
            raise ValueError("poll_interval_sec 必须位于 [0.01, 60]")
        if not 1024 * 1024 <= max_event_log_bytes <= 16 * 1024**3:
            raise ValueError("max_event_log_bytes 必须位于 [1 MiB, 16 GiB]")
        if parent_pid is not None and parent_pid <= 1:
            raise ValueError("parent_pid 必须是大于 1 的进程 ID")

        resolved_event_log = Path(event_log).expanduser().resolve()
        resolved_summary = Path(summary).expanduser().resolve()
        resolved_ready = Path(ready_file).expanduser().resolve() if ready_file else None
        resolved_stop = Path(stop_file).expanduser().resolve() if stop_file else None
        resolved_lock = Path(lock_file).expanduser().resolve() if lock_file else None
        evidence_paths = tuple(
            path
            for path in (resolved_event_log, resolved_summary, resolved_ready)
            if path
        )
        control_paths = tuple(
            path
            for path in (*evidence_paths, resolved_stop, resolved_lock)
            if path is not None
        )
        if len(set(control_paths)) != len(control_paths):
            raise ValueError("事件、汇总、ready、stop 和 lock 路径必须彼此不同")
        for control_path in control_paths:
            if any(_is_within(control_path, root) for root in resolved_roots):
                raise ValueError(
                    f"监控输出和控制文件必须位于监控根目录之外: {control_path}"
                )
        for evidence_path in evidence_paths:
            if evidence_path.exists():
                raise FileExistsError(f"拒绝覆盖已有监控工件: {evidence_path}")
        if resolved_stop is not None and resolved_stop.exists():
            raise FileExistsError(f"停止文件已存在，无法开始新监控: {resolved_stop}")

        return cls(
            roots=resolved_roots,
            event_log=resolved_event_log,
            summary=resolved_summary,
            forbidden_suffixes=_normalize_suffixes(forbidden_suffixes),
            ready_file=resolved_ready,
            stop_file=resolved_stop,
            lock_file=resolved_lock,
            parent_pid=parent_pid,
            poll_interval_sec=float(poll_interval_sec),
            max_event_log_bytes=int(max_event_log_bytes),
        )


@dataclass(frozen=True)
class ArtifactMonitorResult:
    status: str
    summary: Path
    event_log: Path
    forbidden_event_count: int
    integrity_failure_count: int
    integrity_failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class _EvidenceWriter:
    def __init__(self, path: Path, max_bytes: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x", encoding="utf-8")
        self._sequence = 0
        self._max_bytes = max_bytes
        self._bytes_written = 0
        self._last_sync = time.monotonic()

    def write(self, kind: str, *, force: bool = False, **fields: Any) -> None:
        self._sequence += 1
        record = {
            "schemaVersion": 1,
            "sequence": self._sequence,
            "observedAt": _utc_now(),
            "monotonicNs": time.monotonic_ns(),
            "kind": kind,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
        encoded_size = len(line.encode("utf-8"))
        if not force and self._bytes_written + encoded_size > self._max_bytes:
            self._sequence -= 1
            raise RuntimeError(
                f"事件证据超过容量上限 {self._max_bytes} bytes"
            )
        self._stream.write(line)
        self._stream.flush()
        self._bytes_written += encoded_size
        if force or time.monotonic() - self._last_sync >= 0.25:
            self.sync()

    def sync(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._last_sync = time.monotonic()

    def close(self) -> None:
        if not self._stream.closed:
            self.sync()
            self._stream.close()


class LinuxInotifyArtifactMonitor:
    """Recursively watch filesystem mutations and produce durable evidence."""

    def __init__(self, config: ArtifactMonitorConfig):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("Linux inotify 监控只能在 Linux 上运行")
        self.config = config
        self._stop_requested = threading.Event()
        self._stop_reason = "requested"
        self._fd = -1
        self._lock_stream: Any | None = None
        self._wd_to_path: dict[int, Path] = {}
        self._path_to_wd: dict[Path, int] = {}
        self._writer: _EvidenceWriter | None = None
        self._filesystem_event_count = 0
        self._forbidden_event_count = 0
        self._forbidden_paths: set[str] = set()
        self._forbidden_paths_truncated = False
        self._integrity_failure_count = 0
        self._integrity_failures: list[str] = []
        self._started_at = _utc_now()

    def request_stop(self, reason: str = "requested") -> None:
        self._stop_reason = reason
        self._stop_requested.set()

    def _acquire_process_lock(self) -> None:
        if self.config.lock_file is None:
            return
        import fcntl

        self.config.lock_file.parent.mkdir(parents=True, exist_ok=True)
        stream = self.config.lock_file.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise RuntimeError("已有模型工件监控进程持有单实例锁") from None
        self._lock_stream = stream

    def _parent_exited(self) -> bool:
        parent_pid = self.config.parent_pid
        return parent_pid is not None and os.getppid() != parent_pid

    def _open_inotify(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add_watch = libc.inotify_add_watch
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        self._libc = libc
        self._add_watch_call = add_watch
        self._fd = init(os.O_NONBLOCK | os.O_CLOEXEC)
        if self._fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    def _add_watch(self, directory: Path) -> None:
        directory = directory.resolve()
        if directory in self._path_to_wd or directory.is_symlink():
            return
        descriptor = self._add_watch_call(
            self._fd,
            os.fsencode(directory),
            _WATCH_MASK,
        )
        if descriptor < 0:
            error = ctypes.get_errno()
            raise OSError(
                error,
                f"inotify_add_watch({directory}): {os.strerror(error)}",
            )
        old_path = self._wd_to_path.get(descriptor)
        if old_path is not None:
            self._path_to_wd.pop(old_path, None)
        self._wd_to_path[descriptor] = directory
        self._path_to_wd[directory] = descriptor

    def _add_tree(self, root: Path) -> None:
        for directory, subdirectories, _files in os.walk(root, followlinks=False):
            current = Path(directory)
            subdirectories[:] = [
                name for name in subdirectories if not (current / name).is_symlink()
            ]
            self._add_watch(current)

    def _record_forbidden(self, path: Path, *, phase: str, event: str) -> None:
        self._forbidden_event_count += 1
        path_text = str(path)
        if len(self._forbidden_paths) < MAX_RECORDED_FORBIDDEN_PATHS:
            self._forbidden_paths.add(path_text)
        elif path_text not in self._forbidden_paths:
            self._forbidden_paths_truncated = True
        assert self._writer is not None
        self._writer.write(
            "forbidden_artifact",
            phase=phase,
            event=event,
            path=str(path),
        )

    def _record_integrity_failure(self, message: str) -> None:
        self._integrity_failure_count += 1
        if len(self._integrity_failures) < MAX_RECORDED_INTEGRITY_FAILURES:
            self._integrity_failures.append(message)
        assert self._writer is not None
        self._writer.write("monitor_integrity_failure", message=message)

    def _scan_tree(self, root: Path, *, phase: str) -> None:
        try:
            paths = root.rglob("*")
            for path in paths:
                if path.is_file() and _is_forbidden_path(
                    path,
                    self.config.forbidden_suffixes,
                ):
                    self._record_forbidden(path.resolve(), phase=phase, event="SCAN")
        except OSError as exc:
            failure = f"{phase} 扫描失败: {root}: {exc}"
            self._record_integrity_failure(failure)

    def _write_ready_file(self) -> None:
        if self.config.ready_file is None:
            return
        payload = {
            "schemaVersion": 1,
            "pid": os.getpid(),
            "startedAt": self._started_at,
            "roots": [str(root) for root in self.config.roots],
            "eventLog": str(self.config.event_log),
        }
        _write_json_atomic(self.config.ready_file, payload)

    def _handle_event(self, descriptor: int, mask: int, cookie: int, name: str) -> None:
        assert self._writer is not None
        if mask & _IN_Q_OVERFLOW:
            failure = "inotify 事件队列溢出，监控证据不完整"
            self._record_integrity_failure(failure)
            return

        watched_directory = self._wd_to_path.get(descriptor)
        if watched_directory is None:
            failure = f"收到未知 watch descriptor: {descriptor}"
            self._record_integrity_failure(failure)
            return

        path = watched_directory / name if name else watched_directory
        is_directory = bool(mask & _IN_ISDIR)
        names = _event_names(mask)
        self._filesystem_event_count += 1
        forbidden = not is_directory and _is_forbidden_path(
            path,
            self.config.forbidden_suffixes,
        )
        self._writer.write(
            "filesystem_event",
            path=str(path),
            events=names,
            mask=mask,
            cookie=cookie,
            isDirectory=is_directory,
            forbidden=forbidden,
        )
        if forbidden:
            self._record_forbidden(path, phase="event", event="|".join(names))

        if is_directory and mask & (_IN_CREATE | _IN_MOVED_TO):
            try:
                if path.is_dir():
                    self._add_tree(path)
                    self._scan_tree(path, phase="dynamic")
            except OSError as exc:
                failure = f"新增目录监听失败: {path}: {exc}"
                self._record_integrity_failure(failure)

        if mask & (_IN_DELETE_SELF | _IN_MOVE_SELF | _IN_UNMOUNT):
            if path in self.config.roots:
                failure = f"监控根目录丢失或卸载: {path}"
                self._record_integrity_failure(failure)

        if mask & _IN_IGNORED:
            removed = self._wd_to_path.pop(descriptor, None)
            if removed is not None:
                self._path_to_wd.pop(removed, None)

    def _read_events(self) -> None:
        while True:
            try:
                data = os.read(self._fd, 1024 * 1024)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise
            if not data:
                return
            for event in _decode_inotify_buffer(data):
                self._handle_event(*event)

    def _stop_file_requested(self) -> bool:
        stop_file = self.config.stop_file
        return stop_file is not None and stop_file.exists()

    def _build_summary(self, *, error: str | None = None) -> dict[str, Any]:
        event_log_sha256 = _sha256_file(self.config.event_log)
        failed = bool(self._forbidden_event_count or self._integrity_failure_count or error)
        return {
            "schemaVersion": 1,
            "status": "failed" if failed else "passed",
            "startedAt": self._started_at,
            "finishedAt": _utc_now(),
            "stopReason": self._stop_reason,
            "roots": [str(root) for root in self.config.roots],
            "eventLog": {
                "path": str(self.config.event_log),
                "sha256": event_log_sha256,
                "sizeBytes": self.config.event_log.stat().st_size,
            },
            "forbiddenSuffixes": list(self.config.forbidden_suffixes),
            "filesystemEventCount": self._filesystem_event_count,
            "forbiddenEventCount": self._forbidden_event_count,
            "forbiddenPaths": sorted(self._forbidden_paths),
            "forbiddenPathsTruncated": self._forbidden_paths_truncated,
            "integrityFailureCount": self._integrity_failure_count,
            "integrityFailures": list(self._integrity_failures),
            "integrityFailuresTruncated": (
                self._integrity_failure_count > len(self._integrity_failures)
            ),
            "error": error,
        }

    def run(self) -> ArtifactMonitorResult:
        error: str | None = None
        self._writer = _EvidenceWriter(
            self.config.event_log,
            self.config.max_event_log_bytes,
        )
        try:
            self._acquire_process_lock()
            self._open_inotify()
            for root in self.config.roots:
                self._add_tree(root)
            self._writer.write(
                "monitor_started",
                pid=os.getpid(),
                roots=[str(root) for root in self.config.roots],
                forbiddenSuffixes=list(self.config.forbidden_suffixes),
            )
            for root in self.config.roots:
                self._scan_tree(root, phase="initial")
            self._writer.sync()
            self._write_ready_file()

            with selectors.DefaultSelector() as selector:
                selector.register(self._fd, selectors.EVENT_READ)
                while not self._stop_requested.is_set():
                    if self._parent_exited():
                        self._stop_reason = "parent_exit"
                        self._record_integrity_failure(
                            "启动监控的父进程已退出，证据链提前终止"
                        )
                        break
                    if self._stop_file_requested():
                        self._stop_reason = "stop_file"
                        break
                    if selector.select(self.config.poll_interval_sec):
                        self._read_events()
                self._read_events()
            if self._stop_reason in {"SIGINT", "SIGTERM"}:
                self._record_integrity_failure(
                    f"监控收到非受控停止信号: {self._stop_reason}"
                )
            for root in self.config.roots:
                self._scan_tree(root, phase="final")
        except Exception as exc:  # report monitor failures instead of losing evidence
            error = f"{type(exc).__name__}: {exc}"
            self._integrity_failure_count += 1
            if len(self._integrity_failures) < MAX_RECORDED_INTEGRITY_FAILURES:
                self._integrity_failures.append(error)
            self._writer.write("monitor_error", error=error, force=True)
        finally:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            self._writer.write(
                "monitor_stopped",
                force=True,
                stopReason=self._stop_reason,
                forbiddenEventCount=self._forbidden_event_count,
                integrityFailureCount=self._integrity_failure_count,
            )
            self._writer.close()
            if self._lock_stream is not None:
                self._lock_stream.close()
                self._lock_stream = None

        summary = self._build_summary(error=error)
        _write_json_atomic(self.config.summary, summary)
        return ArtifactMonitorResult(
            status=str(summary["status"]),
            summary=self.config.summary,
            event_log=self.config.event_log,
            forbidden_event_count=self._forbidden_event_count,
            integrity_failure_count=self._integrity_failure_count,
            integrity_failures=tuple(self._integrity_failures),
        )


def install_signal_handlers(monitor: LinuxInotifyArtifactMonitor) -> None:
    """Request a clean evidence flush on SIGINT or SIGTERM."""

    def handle_signal(signum, _frame):
        monitor.request_stop(signal.Signals(signum).name)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


__all__ = [
    "ArtifactMonitorConfig",
    "ArtifactMonitorResult",
    "DEFAULT_FORBIDDEN_SUFFIXES",
    "DEFAULT_MAX_EVENT_LOG_BYTES",
    "LinuxInotifyArtifactMonitor",
    "install_signal_handlers",
]
