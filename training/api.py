"""训练 HTTP API。"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.gpu_lock import GpuDeviceLock, parse_device_index
from training.paths import STORAGE_ROOT, YOLO_PYTHON
from training.service import MANAGER

logger = logging.getLogger(__name__)

app = FastAPI(title="autolab-training", version="0.2.0")


class TrainDetectionBody(BaseModel):
    action: str = Field(..., description="start | stop")
    projectId: Optional[str] = None
    taskId: Optional[str] = None
    trainNum: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    image_size: Optional[int] = None
    model: Optional[str] = None
    is_continue: Optional[Any] = None
    last_train: Optional[str] = None
    device: Optional[str] = "0"

    class Config:
        extra = "allow"


def _probe_ultralytics() -> Dict[str, Any]:
    try:
        import ultralytics

        return {"ok": True, "version": getattr(ultralytics, "__version__", None)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _probe_storage() -> Dict[str, Any]:
    try:
        root = STORAGE_ROOT.resolve()
        root.mkdir(parents=True, exist_ok=True)
        test = root / ".write_probe"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        usage = shutil.disk_usage(root)
        return {
            "ok": True,
            "path": str(root),
            "free_gb": round(usage.free / (1024**3), 2),
        }
    except Exception as exc:
        return {"ok": False, "path": str(STORAGE_ROOT), "error": str(exc)}


def _probe_yolo_python() -> Dict[str, Any]:
    py = Path(YOLO_PYTHON)
    return {"path": str(py), "exists": py.is_file()}


@app.get("/api/health")
def health() -> Dict[str, Any]:
    cur = MANAGER.current()
    device = "0"
    if cur and cur.get("device"):
        device = str(cur["device"])
    gpu_busy = GpuDeviceLock(device).is_held_by_other()
    storage = _probe_storage()
    ultralytics = _probe_ultralytics()
    degraded = not storage.get("ok") or not ultralytics.get("ok")
    return {
        "status": "degraded" if degraded else "ok",
        "current": cur,
        "yolo_python": _probe_yolo_python(),
        "storage": storage,
        "ultralytics": ultralytics,
        "gpu": {"device": parse_device_index(device), "busy": gpu_busy},
    }


@app.get("/api/train/status")
def train_status() -> Dict[str, Any]:
    cur = MANAGER.current()
    return cur or {"status": "idle"}


@app.get("/api/train/progress")
def train_progress(
    projectId: Optional[str] = Query(None),
    taskId: Optional[str] = Query(None),
    trainNum: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """返回当前任务或指定三元组的训练进度（解析 trainning_data.csv 末行）。"""
    cur = MANAGER.current()
    if projectId and taskId and trainNum:
        from training.progress import read_job_progress

        return {
            "status": "ok",
            "progress": read_job_progress(projectId, taskId, trainNum),
        }
    if cur is None:
        return {"status": "idle"}
    return {"status": "ok", **MANAGER.progress()}


def _positive_int_field(name: str, value: Any) -> int:
    if value is None or value == "":
        raise HTTPException(status_code=400, detail=f"缺少字段: {name}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} 须为正整数") from exc
    if parsed <= 0:
        raise HTTPException(status_code=400, detail=f"{name} 须为正整数")
    return parsed


def _train_detection(body: TrainDetectionBody) -> Dict[str, Any]:
    action = (body.action or "").strip().lower()
    if action == "stop":
        result = MANAGER.stop()
        return {"status": "success", **result}

    if action != "start":
        raise HTTPException(status_code=400, detail="action 须为 start 或 stop")

    required_str = ["projectId", "taskId", "trainNum"]
    missing = [k for k in required_str if getattr(body, k, None) in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail=f"缺少字段: {missing}")

    body.epochs = _positive_int_field("epochs", body.epochs)
    body.batch_size = _positive_int_field("batch_size", body.batch_size)
    body.image_size = _positive_int_field("image_size", body.image_size)

    param: Dict[str, Any] = body.model_dump(exclude_none=True)
    param.pop("action", None)
    device = str(param.pop("device", "0") or "0")

    try:
        job = MANAGER.start(param, sync=False, device=device)
        return {"status": "success", "job": job}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("train start failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/{endpoint_name}/train/detection")
def train_detection(endpoint_name: str, body: TrainDetectionBody) -> Dict[str, Any]:
    if endpoint_name != "yolo_detector":
        raise HTTPException(
            status_code=400,
            detail=f"本模块仅支持 yolo_detector，收到 endpoint_name={endpoint_name}",
        )
    return _train_detection(body)
