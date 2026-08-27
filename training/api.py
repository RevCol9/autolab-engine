"""训练 HTTP API。"""

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
    is_continue: Optional[Any] = None
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


def _train_detection(body: TrainDetectionBody) -> Dict[str, Any]:
    action = (body.action or "").strip().lower()
    if action == "stop":
        result = MANAGER.stop()
        return {"status": "success", **result}

    if action != "start":
        raise HTTPException(status_code=400, detail="action 须为 start 或 stop")

    required = ["projectId", "taskId", "trainNum", "epochs", "batch_size", "image_size"]
    missing = [k for k in required if getattr(body, k, None) in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail=f"缺少字段: {missing}")

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
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/{endpoint_name}/train/detection")
def train_detection(endpoint_name: str, body: TrainDetectionBody) -> Dict[str, Any]:
    if endpoint_name != "yolo_detector":
        raise HTTPException(
            status_code=400,
            detail=f"本模块仅支持 yolo_detector，收到 endpoint_name={endpoint_name}",
        )
    return _train_detection(body)
