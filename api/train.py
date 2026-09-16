"""训练 HTTP API（:21011，端口见 config/training/base.yaml）。

POST .../train/detection | .../train/segmentation：action=start|stop
POST .../dataset/clean：数据清洗（训练前）
GET /api/train/status、/api/train/progress、/api/dataset/clean/defaults
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from shared.device import is_cuda_device
from shared.gpu_lock import GpuDeviceLock, parse_device_index
from shared.openapi_docs import openapi_description
from training.data_clean_service import run_storage_data_clean
from training.backends import registered_tasks
from training.settings import default_training_device, resolve_training_device
from training.paths import STORAGE_ROOT, YOLO_PYTHON
from training.service import MANAGER

logger = logging.getLogger(__name__)

TrainTask = Literal["detection", "segmentation"]

app = FastAPI(
    title="autolab-training",
    version="0.3.0",
    description=openapi_description("train", summary="YOLO 闭环训练（检测 / 分割）"),
)


class TrainJobBody(BaseModel):
    action: str = Field(..., description="start | stop")
    projectId: Optional[str] = None
    taskId: Optional[str] = None
    trainNum: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    image_size: Optional[int] = None
    model: Optional[str] = None
    is_continue: Optional[bool] = None
    last_train: Optional[str] = None
    device: Optional[str] = None

    model_config = ConfigDict(extra="allow")


# 兼容旧 schema 名
TrainDetectionBody = TrainJobBody


class DataCleanThresholds(BaseModel):
    dark_brightness_lt: Optional[float] = Field(None, description="过暗：brightness 低于此值剔除")
    odd_aspect_ratio_lt: Optional[float] = Field(None, description="异常宽高比")
    low_information_entropy_lt: Optional[float] = Field(None, description="低信息量：entropy 低于此值剔除")
    blurry_blurriness_lt: Optional[float] = Field(None, description="模糊：blurriness 低于此值剔除")
    odd_size_lt: Optional[float] = Field(None, description="异常尺寸：size 低于此值剔除")


class DataCleanBody(BaseModel):
    projectId: str = Field(..., description="与训练一致，如 algorithms")
    taskId: str = Field(..., description="任务 ID，如 Helmet")
    trainNum: str = Field(..., description="训练批次目录名，如 train1")
    outputName: Optional[str] = Field("clean_output", description="输出子目录名，位于 trainNum 下")
    overwrite: Optional[bool] = Field(True, description="是否覆盖已有 clean_output")
    skipCleanvision: Optional[bool] = Field(False, description="true 时跳过全部图像质量过滤，仅做标签校验与导出")
    requireCleanvision: Optional[bool] = Field(True, description="true 且未安装 cleanvision 时返回 400")
    enabledFilters: Optional[list[str]] = Field(
        None,
        description=(
            "启用的过滤器；不传则全部启用。可选: dark, odd_aspect_ratio, "
            "low_information, blurry, odd_size, near_duplicates, exact_duplicates, "
            "missing_label"
        ),
    )
    thresholds: Optional[DataCleanThresholds] = Field(
        None,
        description="各过滤器阈值；未传字段使用算法默认值",
    )


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


@lru_cache(maxsize=1)
def _probe_yolo_python() -> Dict[str, Any]:
    configured = str(YOLO_PYTHON).strip()
    candidate = Path(configured).expanduser()
    executable = str(candidate) if candidate.is_file() else shutil.which(configured)
    if not executable:
        return {"ok": False, "path": configured, "exists": False}
    try:
        proc = subprocess.run(
            [
                executable,
                "-c",
                "import ultralytics; print(getattr(ultralytics, '__version__', 'unknown'))",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception as exc:
        return {"ok": False, "path": executable, "exists": True, "error": str(exc)}
    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout or "ultralytics import failed").strip()
        return {"ok": False, "path": executable, "exists": True, "error": error[-1000:]}
    return {
        "ok": True,
        "path": executable,
        "exists": True,
        "ultralytics_version": (proc.stdout or "").strip() or None,
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    cur = MANAGER.current()
    device = default_training_device()
    if cur and cur.get("device"):
        device = str(cur["device"])
    gpu_error = None
    try:
        gpu_device = parse_device_index(device)
        gpu_lock = GpuDeviceLock(device) if is_cuda_device(device) else None
        gpu_busy = gpu_lock.is_held_by_other() if gpu_lock is not None else False
        physical_device = gpu_lock.device_key if gpu_lock is not None else None
    except ValueError as exc:
        gpu_device = str(device)
        gpu_busy = False
        physical_device = None
        gpu_error = str(exc)
    storage = _probe_storage()
    ultralytics = _probe_ultralytics()
    yolo_python = _probe_yolo_python()
    degraded = (
        not storage.get("ok")
        or not ultralytics.get("ok")
        or not yolo_python.get("ok")
        or gpu_error is not None
    )
    return {
        "status": "degraded" if degraded else "ok",
        "current": cur,
        "train_tasks": registered_tasks(),
        "yolo_python": yolo_python,
        "storage": storage,
        "ultralytics": ultralytics,
        "gpu": {
            "device": gpu_device,
            "physical_device": physical_device,
            "busy": gpu_busy,
            "error": gpu_error,
        },
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


def _assert_yolo_detector(endpoint_name: str) -> None:
    if endpoint_name != "yolo_detector":
        raise HTTPException(
            status_code=400,
            detail=f"本模块仅支持 yolo_detector，收到 endpoint_name={endpoint_name}",
        )


def _run_train_job(body: TrainJobBody, *, task: TrainTask) -> Dict[str, Any]:
    action = (body.action or "").strip().lower()
    if action == "stop":
        result = MANAGER.stop()
        job_status = (result.get("job") or {}).get("status")
        response_status = "stopping" if job_status == "stopping" else "success"
        return {"status": response_status, **result}

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
    try:
        device_raw = param.pop("device", None)
        device = resolve_training_device(device_raw)
        job = MANAGER.start(param, task=task, device=device)
        return {"status": "success", "job": job}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("train start failed task=%s", task)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/{endpoint_name}/train/detection")
def train_detection(endpoint_name: str, body: TrainJobBody) -> Dict[str, Any]:
    _assert_yolo_detector(endpoint_name)
    return _run_train_job(body, task="detection")


@app.post("/api/v1/{endpoint_name}/train/segmentation")
def train_segmentation(endpoint_name: str, body: TrainJobBody) -> Dict[str, Any]:
    _assert_yolo_detector(endpoint_name)
    return _run_train_job(body, task="segmentation")


@app.get("/api/dataset/clean/defaults")
def dataset_clean_defaults() -> Dict[str, Any]:
    """返回前端可展示的默认阈值与可选过滤器列表。"""
    from toolkit.data_clean.config import DEFAULT_THRESHOLDS, FILTER_NAMES

    return {
        "filters": list(FILTER_NAMES),
        "defaultEnabledFilters": list(FILTER_NAMES),
        "defaultThresholds": DEFAULT_THRESHOLDS,
    }


@app.post("/api/v1/{endpoint_name}/dataset/clean")
def dataset_clean(endpoint_name: str, body: DataCleanBody) -> Dict[str, Any]:
    """
    对 storage 下 ``{projectId}/{taskId}/{trainNum}/`` 执行数据清洗。

    前端配置参数 → Java 原样转发 → 本接口执行 → 返回摘要；明细 CSV 写在 outputRoot 下。
    """
    _assert_yolo_detector(endpoint_name)

    thresholds = None
    if body.thresholds is not None:
        thresholds = {
            k: v
            for k, v in body.thresholds.model_dump(exclude_none=True).items()
            if v is not None
        }

    try:
        return run_storage_data_clean(
            body.projectId,
            body.taskId,
            body.trainNum,
            outputName=body.outputName,
            overwrite=body.overwrite,
            skipCleanvision=body.skipCleanvision,
            requireCleanvision=body.requireCleanvision,
            enabledFilters=body.enabledFilters,
            thresholds=thresholds,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("dataset clean failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
