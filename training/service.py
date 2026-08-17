"""训练任务管理：单任务 start/stop（替代原 Django TrainThread 强杀模型）。"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from training.trainer import popen_detection_train, run_detection_train

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
            if self._job is None:
                return None
            return self._snapshot(self._job)

    def start(self, param: Dict[str, Any], *, sync: bool = False, device: str = "0") -> Dict[str, Any]:
        with self._lock:
            if self._job and self._job.status == "running":
                raise RuntimeError(f"已有训练任务在跑: {self._job.job_id} (pid={self._job.pid})")
            self._counter += 1
            job_id = f"train-{self._counter}-{int(time.time())}"
            job = TrainJob(job_id=job_id, param=dict(param), status="running", started_at=time.time())
            self._job = job

        if sync:
            try:
                result = run_detection_train(param, device=device)
                with self._lock:
                    job.status = "finished"
                    job.result = result
                    job.finished_at = time.time()
                return self._snapshot(job)
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = time.time()
                raise

        proc = popen_detection_train(param, device=device)
        with self._lock:
            job.pid = proc.pid
            job._proc = proc

        def _wait() -> None:
            assert proc is not None
            code = proc.wait()
            with self._lock:
                job.finished_at = time.time()
                if job.status == "stopped":
                    return
                if code == 0:
                    from training.paths import train_save_dir

                    save_dir = train_save_dir(
                        str(param["projectId"]), str(param["taskId"]), str(param["trainNum"])
                    )
                    job.status = "finished"
                    job.result = {
                        "train_status": "finished",
                        "weight_path": str(save_dir / "weights" / "best.pt"),
                        "save_dir": str(save_dir),
                        "report": str(save_dir / "report.json"),
                        "csv": str(save_dir / "trainning_data.csv"),
                    }
                else:
                    job.status = "failed"
                    job.error = f"exit_code={code}"

        threading.Thread(target=_wait, daemon=True).start()
        return self._snapshot(job)

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None or job.status != "running" or job._proc is None:
                return {"status": "idle", "message": "无运行中的训练任务"}
            proc = job._proc
            job.status = "stopped"
            job.finished_at = time.time()
            pid = job.pid

        try:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        except Exception as exc:
            logger.warning("stop train failed: %s", exc)
        return {"status": "stopped", "job_id": job.job_id, "pid": pid}

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
