"""Web lifecycle manager for the independent inotify evidence process."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.paths import PLATFORM_ROOT, STORAGE_ROOT, safe_id

REPO_ROOT = Path(__file__).resolve().parent.parent
MONITOR_SCRIPT = REPO_ROOT / "scripts" / "monitor_model_artifacts.py"


def _configured_path(environment_name: str, default: Path) -> Path:
    value = os.environ.get(environment_name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


DEFAULT_EVIDENCE_ROOT = _configured_path(
    "NIII_MODEL_MONITOR_EVIDENCE_ROOT",
    PLATFORM_ROOT / "model_crypto_evidence",
)
DEFAULT_TEMP_ROOT = _configured_path(
    "NIII_MODEL_MONITOR_TEMP_ROOT",
    PLATFORM_ROOT / "model_crypto_tmp",
)


class ArtifactMonitorBusyError(RuntimeError):
    """A monitor process is already active."""


class ArtifactMonitorUnavailableError(RuntimeError):
    """The target platform cannot start the monitor."""


@dataclass
class _MonitorSession:
    session_id: str
    project_id: str
    task_id: str
    train_num: str
    run_root: Path
    temp_root: Path
    evidence_root: Path
    event_log: Path
    summary: Path
    ready_file: Path
    stop_file: Path
    process_log: Path
    process: subprocess.Popen[Any]
    started_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"监控 JSON 证据不可读: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"监控 JSON 证据必须是 object: {path}")
    return value


class ArtifactMonitorService:
    """Start and control one process-external monitor for the training API."""

    def __init__(
        self,
        *,
        storage_root: Path = STORAGE_ROOT,
        evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
        temp_root: Path = DEFAULT_TEMP_ROOT,
        python_executable: str = sys.executable,
        monitor_script: Path = MONITOR_SCRIPT,
        ready_timeout_sec: float = 10.0,
        stop_timeout_sec: float = 10.0,
    ) -> None:
        self.storage_root = storage_root.expanduser().resolve()
        self.evidence_root = evidence_root.expanduser().resolve()
        self.temp_root = temp_root.expanduser().resolve()
        self.python_executable = python_executable
        self.monitor_script = monitor_script.expanduser().resolve()
        self.ready_timeout_sec = ready_timeout_sec
        self.stop_timeout_sec = stop_timeout_sec
        self._lock = threading.Lock()
        self._session: _MonitorSession | None = None

    def _run_root(self, project_id: str, task_id: str, train_num: str) -> Path:
        parts = (
            safe_id("projectId", project_id),
            safe_id("taskId", task_id),
            safe_id("trainNum", train_num),
        )
        run_root = self.storage_root.joinpath(*parts).resolve()
        if not _is_within(run_root, self.storage_root):
            raise ValueError(f"训练目录越界: {run_root}")
        if not run_root.is_dir():
            raise FileNotFoundError(f"训练目录不存在: {run_root}")
        return run_root

    @staticmethod
    def _session_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}-{secrets.token_hex(4)}"

    @staticmethod
    def _process_exit_code(session: _MonitorSession) -> int | None:
        return session.process.poll()

    @staticmethod
    def _temporary_environment(session: _MonitorSession) -> dict[str, str]:
        tmp_dir = str(session.temp_root / "tmp")
        return {
            "TMPDIR": tmp_dir,
            "TMP": tmp_dir,
            "TEMP": tmp_dir,
            "TORCH_HOME": str(session.temp_root / "torch-home"),
            "XDG_CACHE_HOME": str(session.temp_root / "xdg-cache"),
            "MPLCONFIGDIR": str(session.temp_root / "matplotlib"),
        }

    def _snapshot_locked(self, session: _MonitorSession) -> dict[str, Any]:
        exit_code = self._process_exit_code(session)
        summary = _read_json(session.summary)
        if exit_code is None:
            status = "running" if session.ready_file.is_file() else "starting"
        elif summary is not None:
            status = str(summary.get("status") or "error")
        else:
            status = "error"
        temporary_environment = self._temporary_environment(session)
        return {
            "status": status,
            "sessionId": session.session_id,
            "projectId": session.project_id,
            "taskId": session.task_id,
            "trainNum": session.train_num,
            "pid": session.process.pid,
            "monitorExitCode": exit_code,
            "startedAt": session.started_at,
            "roots": [str(session.run_root), str(session.temp_root)],
            "tempRoot": str(session.temp_root),
            "tempDir": temporary_environment["TMPDIR"],
            "environment": temporary_environment,
            "evidenceDir": str(session.evidence_root),
            "eventLog": str(session.event_log),
            "summaryPath": str(session.summary),
            "processLog": str(session.process_log),
            "summary": summary,
        }

    @staticmethod
    def _terminate(process: subprocess.Popen[Any], timeout_sec: float) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            process.kill()
        process.wait(timeout=5)

    def start(
        self,
        *,
        project_id: str,
        task_id: str,
        train_num: str,
    ) -> dict[str, Any]:
        if not sys.platform.startswith("linux"):
            raise ArtifactMonitorUnavailableError("Web 文件事件监控仅支持 Linux inotify")
        if not self.monitor_script.is_file():
            raise ArtifactMonitorUnavailableError(
                f"监控脚本不存在: {self.monitor_script}"
            )

        with self._lock:
            if (
                self._session is not None
                and self._process_exit_code(self._session) is None
            ):
                raise ArtifactMonitorBusyError(
                    f"已有监控会话在运行: {self._session.session_id}"
                )

            run_root = self._run_root(project_id, task_id, train_num)
            session_id = self._session_id()
            evidence_root = self.evidence_root / session_id
            temp_root = self.temp_root / session_id
            for controlled_root in (evidence_root, temp_root):
                if _is_within(controlled_root, run_root):
                    raise ValueError(
                        f"监控证据和临时目录不能位于训练目录内: {controlled_root}"
                    )
            evidence_root.mkdir(parents=True, exist_ok=False)
            temp_root.mkdir(parents=True, exist_ok=False)
            for directory_name in (
                "tmp",
                "torch-home",
                "xdg-cache",
                "matplotlib",
            ):
                (temp_root / directory_name).mkdir()

            event_log = evidence_root / "events.jsonl"
            summary = evidence_root / "summary.json"
            ready_file = evidence_root / "ready.json"
            stop_file = evidence_root / "stop"
            process_log = evidence_root / "monitor.log"
            command = [
                self.python_executable,
                str(self.monitor_script),
                "--root",
                str(run_root),
                "--root",
                str(temp_root),
                "--events",
                str(event_log),
                "--summary",
                str(summary),
                "--ready-file",
                str(ready_file),
                "--stop-file",
                str(stop_file),
            ]
            with process_log.open("x", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )

            session = _MonitorSession(
                session_id=session_id,
                project_id=project_id,
                task_id=task_id,
                train_num=train_num,
                run_root=run_root,
                temp_root=temp_root,
                evidence_root=evidence_root,
                event_log=event_log,
                summary=summary,
                ready_file=ready_file,
                stop_file=stop_file,
                process_log=process_log,
                process=process,
                started_at=_utc_now(),
            )
            self._session = session

            deadline = time.monotonic() + self.ready_timeout_sec
            while not ready_file.is_file():
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"监控进程启动失败，exit_code={exit_code}，日志: {process_log}"
                    )
                if time.monotonic() >= deadline:
                    self._terminate(process, self.stop_timeout_sec)
                    raise TimeoutError(
                        f"等待监控 ready 超时，日志: {process_log}"
                    )
                time.sleep(0.05)
            return self._snapshot_locked(session)

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._session is None:
                return {"status": "idle"}
            return self._snapshot_locked(self._session)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None:
                return {"status": "idle"}
            if session.process.poll() is None:
                try:
                    session.stop_file.touch(exist_ok=False)
                except FileExistsError:
                    pass
                try:
                    session.process.wait(timeout=self.stop_timeout_sec)
                except subprocess.TimeoutExpired:
                    self._terminate(session.process, self.stop_timeout_sec)
            return self._snapshot_locked(session)

    def events(self, *, after_sequence: int = 0, limit: int = 200) -> dict[str, Any]:
        if after_sequence < 0:
            raise ValueError("after_sequence 不能小于 0")
        if not 1 <= limit <= 1000:
            raise ValueError("limit 必须位于 [1, 1000]")
        with self._lock:
            session = self._session
            if session is None:
                raise LookupError("当前没有监控会话")
            event_log = session.event_log
            session_id = session.session_id

        records: list[dict[str, Any]] = []
        if event_log.is_file():
            try:
                with event_log.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"事件证据第 {line_number} 行损坏: {event_log}"
                            ) from exc
                        sequence = int(record.get("sequence") or 0)
                        if sequence <= after_sequence:
                            continue
                        records.append(record)
                        if len(records) >= limit:
                            break
            except OSError as exc:
                raise RuntimeError(f"事件证据不可读: {event_log}: {exc}") from exc
        return {
            "status": "ok",
            "sessionId": session_id,
            "afterSequence": after_sequence,
            "count": len(records),
            "events": records,
        }

    def process_log(self, *, tail: int = 200) -> dict[str, Any]:
        if not 1 <= tail <= 2000:
            raise ValueError("tail 必须位于 [1, 2000]")
        with self._lock:
            session = self._session
            if session is None:
                raise LookupError("当前没有监控会话")
            process_log = session.process_log
            session_id = session.session_id

        lines: list[str] = []
        if process_log.is_file():
            try:
                with process_log.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    size = stream.tell()
                    stream.seek(max(0, size - 1024 * 1024))
                    content = stream.read().decode("utf-8", errors="replace")
                lines = content.splitlines()[-tail:]
            except OSError as exc:
                raise RuntimeError(f"监控进程日志不可读: {process_log}: {exc}") from exc
        return {
            "status": "ok",
            "sessionId": session_id,
            "tail": tail,
            "count": len(lines),
            "lines": lines,
        }

    def environment_for(self, run_root: Path) -> dict[str, str]:
        """Bind a matching training subprocess to the watched temporary directory."""
        resolved_run_root = run_root.expanduser().resolve()
        with self._lock:
            session = self._session
            if session is None or session.run_root != resolved_run_root:
                return {}
            if session.process.poll() is not None or not session.ready_file.is_file():
                raise RuntimeError(
                    f"训练对应的文件监控会话不健康: {session.session_id}"
                )
            return self._temporary_environment(session)


MONITOR_SERVICE = ArtifactMonitorService()


__all__ = [
    "ArtifactMonitorBusyError",
    "ArtifactMonitorService",
    "ArtifactMonitorUnavailableError",
    "MONITOR_SERVICE",
]
