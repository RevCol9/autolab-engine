"""训练任务管理：单任务 start/stop，跨进程 GPU 锁与进度快照。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from shared.gpu_lock import GpuDeviceLock
from training.paths import train_save_dir
from training.progress import read_job_progress
from training.trainer import collect_train_result, kill_process_group, popen_detection_train

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
    device: str = "0"
    _proc: Any = field(default=None, repr=False)
    _gpu_lock: Optional[GpuDeviceLock] = field(default=None, repr=False)


class JobManager:
    """同一时刻只允许一个训练任务；持锁期间阻塞推理进程获取 GPU。"""

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

    def progress(self) -> Dict[str, Any]:
        with self._lock:
            self._reap_if_exited()
            job = self._job
            if job is None:
                return {"status": "idle"}
            payload = job.param
            prog = read_job_progress(
                str(payload["projectId"]),
                str(payload["taskId"]),
                str(payload["trainNum"]),
                total_epochs=int(payload.get("epochs") or 0) or None,
            )
            snap = self._snapshot(job)
            snap["progress"] = prog
            return snap

    def _release_gpu_lock(self, job: TrainJob) -> None:
        lock = job._gpu_lock
        job._gpu_lock = None
        if lock is not None:
            try:
                lock.release()
            except Exception as exc:
                logger.warning("release gpu lock job=%s: %s", job.job_id, exc)

    def _reap_if_exited(self) -> None:
        job = self._job
        if job is None or job.status != "running" or job._proc is None:
            return
        code = job._proc.poll()
        if code is None:
            return
        self._complete_job(job, code)

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

    def start(self, param: Dict[str, Any], *, device: str = "0") -> Dict[str, Any]:
        with self._lock:
            if self._is_busy_locked():
                job = self._job
                raise RuntimeError(
                    f"已有训练任务在跑: {job.job_id if job else '?'} (pid={job.pid if job else None})"
                )

            self._counter += 1
            job_id = f"train-{self._counter}-{int(time.time())}"
            job = TrainJob(job_id=job_id, param=dict(param), status="pending", device=str(device or "0"))
            self._job = job

        gpu_lock = GpuDeviceLock(job.device)
        if not gpu_lock.acquire(blocking=False):
            with self._lock:
                job.status = "failed"
                job.error = "GPU 正被推理占用，无法启动训练"
                job.finished_at = time.time()
            raise RuntimeError("GPU 正被推理占用，无法启动训练")
        job._gpu_lock = gpu_lock

        try:
            proc = popen_detection_train(param, device=device)
        except Exception as exc:
            self._release_gpu_lock(job)
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
            self._release_gpu_lock(job)
            with self._lock:
                job.pid = proc.pid
                job._proc = proc
                job.finished_at = time.time()
            return self._snapshot(job)

        threading.Thread(target=self._wait_for_proc, args=(job, proc, param), daemon=True).start()
        return self._snapshot(job)

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None:
                return {"stopped": False, "message": "无运行中的训练任务", "job": None}

            if job.status == "pending":
                job.status = "stopped"
                job.finished_at = time.time()
                self._release_gpu_lock(job)
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
        self._release_gpu_lock(job)
        return {"stopped": True, "job_id": job.job_id, "pid": pid, "job": snapshot}

    def _wait_for_proc(self, job: TrainJob, proc: Any, param: Dict[str, Any]) -> None:
        code = proc.wait()
        with self._lock:
            self._complete_job(job, code, param=param)
            self._release_gpu_lock(job)

    def _complete_job(self, job: TrainJob, code: int, *, param: Optional[Dict[str, Any]] = None) -> None:
        if job.finished_at is None:
            job.finished_at = time.time()
        if job.status == "stopped":
            return
        if code != 0:
            job.status = "failed"
            job.error = job.error or f"exit_code={code}"
            return

        payload = param or job.param
        try:
            save_dir = train_save_dir(
                str(payload["projectId"]),
                str(payload["taskId"]),
                str(payload["trainNum"]),
            )
            job.result = collect_train_result(save_dir)
            job.status = "finished"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)

    @staticmethod
    def _snapshot(job: TrainJob) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "pid": job.pid,
            "device": job.device,
            "param": {
                "projectId": job.param.get("projectId"),
                "taskId": job.param.get("taskId"),
                "trainNum": job.param.get("trainNum"),
                "epochs": job.param.get("epochs"),
                "batch_size": job.param.get("batch_size"),
                "image_size": job.param.get("image_size"),
                "model": job.param.get("model"),
            },
            "error": job.error,
            "result": job.result,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }


MANAGER = JobManager()
