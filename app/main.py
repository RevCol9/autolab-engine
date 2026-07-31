"""FastAPI 入口：模型加载与单图/批量推理 API。"""

from __future__ import annotations

import json
import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.box_format import SUPPORTED_BOX_FORMATS, apply_box_format
from app.engines.base import BaseEngine
from app.engines.yolo import YoloDetectEngine, YoloSegmentEngine
from app.logging_setup import setup_logging
from app.settings import ModelConfig, Settings, load_settings

SETTINGS: Settings = load_settings()
setup_logging(SETTINGS.log_level)
logger = logging.getLogger("autolab-engine")

app = FastAPI(title="autolab-engine", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engines: Dict[str, BaseEngine] = {}
_active_key: Optional[str] = None

# 单次批量上限，避免一次占满显存/超时
BATCH_MAX_IMAGES = 32


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


def _parse_image(raw: bytes, filename: str = "-") -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail=f"空图片: {filename}")
    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析图片 {filename}: {exc}") from exc


def _normalize_box_format(box_format: Optional[str]) -> str:
    fmt = (box_format or "xyxy").strip().lower()
    if fmt not in SUPPORTED_BOX_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 box_format: {box_format!r}，可选 {list(SUPPORTED_BOX_FORMATS)}",
        )
    return fmt


def _run_predict(
    img: Image.Image,
    cfg: ModelConfig,
    *,
    conf: Optional[float],
    iou: Optional[float],
    imgsz: Optional[int],
    box_format: str,
) -> Dict[str, Any]:
    engine = ensure_engine(cfg.key)
    result = engine.predict(
        img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        coord_space=SETTINGS.coord_space,
    )
    result["boxes"] = apply_box_format(
        result.get("boxes") or [],
        image_width=int(result.get("image_width") or img.width),
        image_height=int(result.get("image_height") or img.height),
        box_format=box_format,
        coord_space=SETTINGS.coord_space,
    )
    result["box_format"] = box_format
    result["model_key"] = cfg.key
    result["model_name"] = cfg.name
    return result


def _parse_image_ids(raw: Optional[str], n: int) -> List[str]:
    if raw is None or str(raw).strip() == "":
        return [str(i) for i in range(n)]
    text = str(raw).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) == n:
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == n:
        return parts
    raise HTTPException(
        status_code=400,
        detail=f"image_ids 数量需与 images 一致（期望 {n}，实际无法解析）",
    )


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
                "loaded": m.key in _engines,
            }
            for m in SETTINGS.models
        ],
    }


@app.get("/api/models")
def models() -> dict:
    """返回 config.yaml 中的模型清单"""
    return {
        "default_model": SETTINGS.default_model,
        "loaded": list(_engines.keys()),
        "box_formats": list(SUPPORTED_BOX_FORMATS),
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
                "max_det": m.max_det,
                "loaded": m.key in _engines,
            }
            for m in SETTINGS.models
        ],
    }


@app.post("/api/models/{model_key}/load")
def load_model(model_key: str) -> dict:
    """显式将指定模型加载到 GPU（半自动标注切换「当前基线 PT」时可预热）。"""
    t0 = time.perf_counter()
    cfg = get_model_config(model_key)
    try:
        ensure_engine(cfg.key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("load fail %s | %s", cfg.key, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    cost = time.perf_counter() - t0
    logger.info("load ok %s | %.2fs", cfg.key, cost)
    return {
        "status": "ok",
        "model_key": cfg.key,
        "model_name": cfg.name,
        "task": cfg.task,
        "device": cfg.device,
        "loaded": list(_engines.keys()),
        "timings": {"load": cost},
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    model_key: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    box_format: Optional[str] = Form("xyxy"),
) -> dict:
    """单图推理。对应平台「生成建议」。"""
    t0 = time.perf_counter()
    fmt = _normalize_box_format(box_format)
    raw = await image.read()
    filename = getattr(image, "filename", None) or "-"
    img = _parse_image(raw, filename)
    cfg = get_model_config(model_key)
    logger.info("req  %s | %s | %sx%s | fmt=%s", cfg.key, filename, img.width, img.height, fmt)

    try:
        result = _run_predict(img, cfg, conf=conf, iou=iou, imgsz=imgsz, box_format=fmt)
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
    logger.info("done %s | boxes=%s | %.2fs", cfg.key, len(result.get("boxes") or []), total)
    return result


@app.post("/api/predict/batch")
async def predict_batch(
    images: List[UploadFile] = File(...),
    image_ids: Optional[str] = Form(None),
    model_key: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    box_format: Optional[str] = Form("xyxy"),
) -> dict:
    """批量推理。同一 model_key；单次最多 BATCH_MAX_IMAGES 张。

    image_ids: JSON 数组或逗号分隔，数量须与 images 一致；省略则用 "0","1",...
    单张失败不中断整批，该条 ok=false 并带 error。
    """
    t0 = time.perf_counter()
    fmt = _normalize_box_format(box_format)
    n = len(images)
    if n == 0:
        raise HTTPException(status_code=400, detail="images 不能为空")
    if n > BATCH_MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"单次批量最多 {BATCH_MAX_IMAGES} 张，当前 {n}",
        )

    ids = _parse_image_ids(image_ids, n)
    cfg = get_model_config(model_key)
    logger.info("batch %s | n=%s | fmt=%s", cfg.key, n, fmt)

    # 预加载，避免第一张才报 path 错误
    try:
        ensure_engine(cfg.key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    results: List[Dict[str, Any]] = []
    ok_n = 0
    for i, (up, image_id) in enumerate(zip(images, ids)):
        filename = getattr(up, "filename", None) or image_id
        item: Dict[str, Any] = {"image_id": image_id, "index": i, "filename": filename}
        try:
            raw = await up.read()
            img = _parse_image(raw, filename)
            one = _run_predict(img, cfg, conf=conf, iou=iou, imgsz=imgsz, box_format=fmt)
            item.update(
                {
                    "ok": True,
                    "task": one.get("task"),
                    "image_width": one.get("image_width"),
                    "image_height": one.get("image_height"),
                    "boxes": one.get("boxes") or [],
                    "masks": one.get("masks") or [],
                    "timings": one.get("timings") or {},
                }
            )
            ok_n += 1
        except HTTPException as exc:
            item.update({"ok": False, "error": exc.detail, "boxes": [], "masks": []})
        except Exception as exc:
            logger.exception("batch item fail %s | %s | %s", cfg.key, image_id, exc)
            item.update({"ok": False, "error": str(exc), "boxes": [], "masks": []})
        results.append(item)

    total = time.perf_counter() - t0
    logger.info("batch done %s | ok=%s/%s | %.2fs", cfg.key, ok_n, n, total)
    return {
        "model_key": cfg.key,
        "model_name": cfg.name,
        "task": cfg.task,
        "box_format": fmt,
        "count": n,
        "ok_count": ok_n,
        "results": results,
        "timings": {"total": total},
    }
