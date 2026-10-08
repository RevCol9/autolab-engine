"""HTTP control and read-only evidence views for model artifact monitoring."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from training.artifact_monitor_service import (
    ArtifactMonitorBusyError,
    ArtifactMonitorUnavailableError,
    MONITOR_SERVICE,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/model-artifact-monitor",
    tags=["model-artifact-monitor"],
)


class ArtifactMonitorStartBody(BaseModel):
    projectId: str = Field(..., min_length=1, description="训练项目 ID")
    taskId: str = Field(..., min_length=1, description="训练任务 ID")
    trainNum: str = Field(..., min_length=1, description="训练批次目录名")

    model_config = ConfigDict(extra="forbid")


@router.post("/start", status_code=201)
def start_artifact_monitor(body: ArtifactMonitorStartBody) -> dict[str, Any]:
    """Start the independent inotify process and wait until it is ready."""
    try:
        return MONITOR_SERVICE.start(
            project_id=body.projectId,
            task_id=body.taskId,
            train_num=body.trainNum,
        )
    except ArtifactMonitorBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ArtifactMonitorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("artifact monitor start failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
def artifact_monitor_status() -> dict[str, Any]:
    """Return the current or most recent in-process session snapshot."""
    try:
        return MONITOR_SERVICE.status()
    except Exception as exc:
        logger.exception("artifact monitor status failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/events")
def artifact_monitor_events(
    after_sequence: int = Query(0, alias="afterSequence", ge=0),
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    """Read a bounded page of durable JSONL events."""
    try:
        return MONITOR_SERVICE.events(
            after_sequence=after_sequence,
            limit=limit,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("artifact monitor events failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/process-log")
def artifact_monitor_process_log(
    tail: int = Query(200, ge=1, le=2000),
) -> dict[str, Any]:
    """Read a bounded tail of the independent monitor process log."""
    try:
        return MONITOR_SERVICE.process_log(tail=tail)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("artifact monitor process log failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stop")
def stop_artifact_monitor() -> dict[str, Any]:
    """Request a clean stop and return the authenticated evidence summary."""
    try:
        return MONITOR_SERVICE.stop()
    except Exception as exc:
        logger.exception("artifact monitor stop failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


__all__ = ["ArtifactMonitorStartBody", "router"]
