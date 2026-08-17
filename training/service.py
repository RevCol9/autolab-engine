"""训练任务管理：单任务 start/stop（替代原 Django TrainThread 强杀模型）。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from training.trainer import (
    collect_train_result,
    kill_process_group,
    popen_detection_train,
    run_detection_train,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainJob:
    job_id: str
    param: Dict[str, Any]
    status: str = "pending"  # pending|running|finished|failed|stopped
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    pid: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    _proc: Any = field(default=None, repr=False)


class JobManager:
    """进程级单例：同一时刻只允许一个训练任务（与原平台全局线程语义接近）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[TrainJob] = None
        self._counter = 0

    def current(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._reap_if_exited()
            if self._job is None:
                return None
            return self._snapshot(self._job)

    def _reap_if_exited(self) -> None:
        job = self._job
        if job is None or job.status != "running" or job._proc is None:
            return
        code = job._proc.poll()
        if code is None:
            return
        if job.finished_at is None:
            job.finished_at = time.time()
        if code == 0 and job.status == "running":
            try:
                from training.paths import train_save_dir

                save_dir = train_save_dir(
                    str(job.param["projectId"]),
                    str(job.param["taskId"]),
                    str(job.param["trainNum"]),
                )
                job.result = collect_train_result(save_dir)
                job.status = "finished"
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
        elif job.status == "running":
            job.status = "failed"
            job.error = job.error or f"exit_code={code}"

    def _is_busy_locked(self) -> bool:
        self._reap_if_exited()
        job = self._job
        if job is None:
            return False
        if job.status == "pending":
            return True
        if job.status == "running":
            if job._proc is not None and job._proc.poll() is not None:
                return False
            return True
        return False

    def start(self, param: Dict[str, Any], *, sync: bool = False, device: str = "0") -> Dict[str, Any]:
        with self._lock:
            if self._is_busy_locked():
                job = self._job
                raise RuntimeError(f"已有训练任务在跑: {job.job_id if job else '?'} (pid={job.pid if job else None})")

            self._counter += 1
            job_id = f"train-{self._counter}-{int(time.time())}"
            job = TrainJob(job_id=job_id, param=dict(param), status="pending")
            self._job = job

        if sync:
            try:
                with self._lock:
                    if job.status == "stopped":
                        return self._snapshot(job)
                    job.status = "running"
                    job.started_at = time.time()
                result = run_detection_train(param, device=device)
                with self._lock:
                    if job.status == "stopped":
                        return self._snapshot(job)
                    job.status = "finished"
                    job.result = result
                    job.finished_at = time.time()
                return self._snapshot(job)
            except Exception as exc:
                with self._lock:
                    if job.status != "stopped":
                        job.status = "failed"
                        job.error = str(exc)
                        job.finished_at = time.time()
                raise

        try:
            proc = popen_detection_train(param, device=device)
        except Exception as exc:
            with self._lock:
                if job.status != "stopped":
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = time.time()
            raise

        kill_after_start = False
        with self._lock:
            if job.status == "stopped":
                kill_after_start = True
            else:
                job.status = "running"
                job.started_at = time.time()
                job.pid = proc.pid
                job._proc = proc

        if kill_after_start:
            kill_process_group(proc)
            with self._lock:
                job.pid = proc.pid
                job._proc = proc
                job.finished_at = time.time()
            return self._snapshot(job)

        def _wait() -> None:
            code = proc.wait()
            with self._lock:
                job.finished_at = time.time()
                if job.status == "stopped":
                    return
                if code == 0:
                    from training.paths import train_save_dir

                    try:
                        save_dir = train_save_dir(
                            str(param["projectId"]), str(param["taskId"]), str(param["trainNum"])
                        )
                        job.result = collect_train_result(save_dir)
                        job.status = "finished"
                    except Exception as exc:
                        job.status = "failed"
                        job.error = str(exc)
                else:
                    job.status = "failed"
                    job.error = f"exit_code={code}"

        threading.Thread(target=_wait, daemon=True).start()
        return self._snapshot(job)

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None:
                return {"stopped": False, "message": "无运行中的训练任务", "job": None}

            if job.status == "pending":
                job.status = "stopped"
                job.finished_at = time.time()
                return {"stopped": True, "job_id": job.job_id, "pid": job.pid, "job": self._snapshot(job)}

            if job.status != "running":
                return {"stopped": False, "message": "无运行中的训练任务", "job": self._snapshot(job)}

            proc = job._proc
            job.status = "stopped"
            job.finished_at = time.time()
            pid = job.pid
            snapshot = self._snapshot(job)

        if proc is not None:
            try:
                kill_process_group(proc)
            except Exception as exc:
                logger.warning("stop train failed: %s", exc)
        return {"stopped": True, "job_id": job.job_id, "pid": pid, "job": snapshot}

    @staticmethod
    def _snapshot(job: TrainJob) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "pid": job.pid,
            "param": {
                "projectId": job.param.get("projectId"),
                "taskId": job.param.get("taskId"),
                "trainNum": job.param.get("trainNum"),
                "epochs": job.param.get("epochs"),
                "model": job.param.get("model"),
            },
            "error": job.error,
            "result": job.result,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }


MANAGER = JobManager()
