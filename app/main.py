"""FastAPI 入口：模型加载与推理 API。"""

from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.engines.base import BaseEngine
from app.engines.yolo import YoloDetectEngine, YoloSegmentEngine
from app.logging_setup import setup_logging
from app.settings import ModelConfig, Settings, load_settings

SETTINGS: Settings = load_settings()
setup_logging(SETTINGS.log_level)
logger = logging.getLogger("autolab-engine")

app = FastAPI(title="autolab-engine", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engines: Dict[str, BaseEngine] = {}
_active_key: Optional[str] = None


def get_model_config(model_key: Optional[str]) -> ModelConfig:
    key = (model_key or SETTINGS.default_model or "").strip()
    if not SETTINGS.models:
        raise HTTPException(status_code=500, detail="config.yaml 未配置 models")
    if not key:
        return SETTINGS.models[0]
    for m in SETTINGS.models:
        if m.key == key:
            return m
    raise HTTPException(status_code=400, detail=f"未知 model_key: {key}")


def build_engine(config: ModelConfig) -> BaseEngine:
    task = (config.task or "detect").lower()
    if task == "detect":
        return YoloDetectEngine(config)
    if task == "segment":
        return YoloSegmentEngine(config)
    raise HTTPException(status_code=400, detail=f"不支持的 task: {task}")


def ensure_engine(model_key: Optional[str] = None) -> BaseEngine:
    global _active_key
    cfg = get_model_config(model_key)
    if cfg.key in _engines:
        _active_key = cfg.key
        return _engines[cfg.key]
    if not cfg.path:
        raise HTTPException(
            status_code=503,
            detail=f"模型 {cfg.key} 尚未配置权重路径（models[].path 为空）",
        )
    engine = build_engine(cfg)
    engine.load()
    _engines[cfg.key] = engine
    _active_key = cfg.key
    return engine


@app.on_event("startup")
def on_startup() -> None:
    logger.info("---------- autolab-engine ----------")
    logger.info("config  %s", SETTINGS.config_path)
    logger.info(
        "models  %s | default=%s",
        [m.key for m in SETTINGS.models],
        SETTINGS.default_model,
    )
    default = get_model_config(None)
    if default.path:
        try:
            ensure_engine(default.key)
            logger.info("ready   model=%s | http://0.0.0.0:%s", default.key, SETTINGS.port)
        except Exception as exc:
            logger.warning("startup skip load: %s", exc)
            logger.info("ready   (lazy load) | http://0.0.0.0:%s", SETTINGS.port)
    else:
        logger.warning("default model path empty; waiting for config.yaml weights path")
        logger.info("ready   (no weights yet) | http://0.0.0.0:%s", SETTINGS.port)
    logger.info("------------------------------------")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "default_model": SETTINGS.default_model,
        "active_model": _active_key,
        "loaded": list(_engines.keys()),
        "models": [
            {
                "key": m.key,
                "name": m.name,
                "task": m.task,
                "path_configured": bool(m.path),
            }
            for m in SETTINGS.models
        ],
    }


@app.get("/api/models")
def models() -> dict:
    return {
        "default_model": SETTINGS.default_model,
        "models": [
            {
                "key": m.key,
                "name": m.name,
                "task": m.task,
                "path": m.path,
                "device": m.device,
                "conf": m.conf,
                "iou": m.iou,
                "imgsz": m.imgsz,
            }
            for m in SETTINGS.models
        ],
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    model_key: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
) -> dict:
    t0 = time.perf_counter()
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空图片")
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析图片: {exc}") from exc

    cfg = get_model_config(model_key)
    logger.info(
        "req  %s | %s | %sx%s",
        cfg.key,
        getattr(image, "filename", None) or "-",
        img.width,
        img.height,
    )
    try:
        engine = ensure_engine(cfg.key)
        result = engine.predict(
            img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            coord_space=SETTINGS.coord_space,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("fail %s | %s", cfg.key, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total = time.perf_counter() - t0
    timings = result.get("timings") or {}
    timings["total"] = total
    result["timings"] = timings
    result["model_key"] = cfg.key
    result["model_name"] = cfg.name

    n_box = len(result.get("boxes") or [])
    logger.info("done %s | boxes=%s | %.2fs", cfg.key, n_box, total)
    return result
