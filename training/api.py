"""训练 HTTP API（兼容原 POST .../train/detection 的 action=start|stop 字段）。

可独立挂载到 FastAPI，或由后端直接调用 training.service.MANAGER。
默认端口建议与推理 :21010 分离，例如 :21011。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from training.service import MANAGER

app = FastAPI(title="autolab-training", version="0.1.0")


class TrainDetectionBody(BaseModel):
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
    device: Optional[str] = "0"

    class Config:
        extra = "allow"


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "current": MANAGER.current()}


@app.get("/api/train/status")
def train_status() -> Dict[str, Any]:
    cur = MANAGER.current()
    return cur or {"status": "idle"}


@app.post("/api/v1/yolo_detector/train/detection")
def train_detection(body: TrainDetectionBody) -> Dict[str, Any]:
    """兼容原 Django TrainDetectionView 路径与字段。"""
    action = (body.action or "").strip().lower()
    if action == "stop":
        return MANAGER.stop()

    if action != "start":
        raise HTTPException(status_code=400, detail="action 须为 start 或 stop")

    required = ["projectId", "taskId", "trainNum", "epochs", "batch_size", "image_size"]
    missing = [k for k in required if getattr(body, k, None) in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail=f"缺少字段: {missing}")

    param: Dict[str, Any] = body.model_dump(exclude_none=True)
    param.pop("action", None)
    device = str(param.pop("device", "0") or "0")
    if "model" in param and isinstance(param["model"], str):
        param["model"] = param["model"].lower()

    try:
        return {"status": "success", **MANAGER.start(param, sync=False, device=device)}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
