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
from app.mask_format import SUPPORTED_MASK_FORMATS
from app.engines.base import BaseEngine
from app.engines.yolo import YoloDetectEngine, YoloSegmentEngine
from app.settings import ModelConfig, Settings, apply_runtime_env, load_settings
from app.test_ui import test_page

SETTINGS: Settings = load_settings()
apply_runtime_env(SETTINGS)
logger = logging.getLogger(__name__)

app = FastAPI(title="autolab-engine", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VLM_ENGINES = frozenset({"locateanything", "sam3"})
_engines: Dict[str, BaseEngine] = {}
_active_key: Optional[str] = None


def is_vlm_engine(cfg: ModelConfig) -> bool:
    return (cfg.engine or "yolo").lower() in VLM_ENGINES


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
    eng = (config.engine or "yolo").lower()
    if eng == "locateanything":
        from app.engines.locate_engine import LocateEngine

        return LocateEngine(config, SETTINGS)
    if eng == "sam3":
        from app.engines.sam3_engine import Sam3Engine

        return Sam3Engine(config, SETTINGS)
    task = (config.task or "detect").lower()
    if task == "detect":
        return YoloDetectEngine(config)
    if task == "segment":
        return YoloSegmentEngine(config)
    raise HTTPException(status_code=400, detail=f"不支持的 task: {task}")


def ensure_model_engine(model_key: Optional[str] = None) -> BaseEngine:
    global _active_key
    cfg = get_model_config(model_key)
    if cfg.key in _engines:
        _active_key = cfg.key
        return _engines[cfg.key]
    if not cfg.path:
        raise HTTPException(
            status_code=503,
            detail=f"模型 {cfg.key} 尚未配置路径（models[].path 为空）",
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


def _normalize_mask_format(mask_format: Optional[str]) -> str:
    fmt = (mask_format or "polygon_norm_pct").strip().lower()
    if fmt not in SUPPORTED_MASK_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 mask_format: {mask_format!r}，可选 {list(SUPPORTED_MASK_FORMATS)}",
        )
    return fmt


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
    task: Optional[str] = None,
    categories: str = "",
    phrase: str = "",
    sam3_threshold: Optional[float] = None,
    sam3_points: Optional[str] = None,
    sam3_boxes: Optional[str] = None,
    mask_format: str = "polygon_norm_pct",
) -> Dict[str, Any]:
    if is_vlm_engine(cfg):
        try:
            engine = ensure_model_engine(cfg.key)
            result = engine.predict(
                img,
                task=task,
                categories=categories,
                phrase=phrase,
                sam3_threshold=sam3_threshold,
                sam3_points=sam3_points,
                sam3_boxes=sam3_boxes,
                conf=conf,
                mask_format=mask_format,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        w = int(result.get("image_width") or img.width)
        h = int(result.get("image_height") or img.height)
        result["boxes"] = apply_box_format(
            result.get("boxes") or [],
            image_width=w,
            image_height=h,
            box_format=box_format,
            coord_space="pixel",
        )
        result["box_format"] = box_format
        result["model_key"] = cfg.key
        result["model_name"] = cfg.name
        result["engine"] = cfg.engine
        if "segments" not in result:
            result["segments"] = []
        return result

    engine = ensure_model_engine(cfg.key)
    is_segment = (cfg.task or "detect").lower() == "segment"
    if is_segment:
        result = engine.predict(
            img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            coord_space=SETTINGS.coord_space,
            mask_format=mask_format,
        )
    else:
        result = engine.predict(
            img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            coord_space=SETTINGS.coord_space,
        )
        result.setdefault("annotation_type", "box")
        result.setdefault("segments", [])
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
    result["engine"] = cfg.engine or "yolo"
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
    default = get_model_config(None)
    if default.path and not is_vlm_engine(default):
        try:
            ensure_model_engine(default.key)
        except Exception as exc:
            logger.warning("startup skip load default=%s: %s", default.key, exc)


@app.get("/test", include_in_schema=True)
def test_inference_page():
    """浏览器联调：选 model_key、上传图片、查看 JSON 与框预览。"""
    return test_page()


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
                "engine": m.engine,
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
        "mask_formats": list(SUPPORTED_MASK_FORMATS),
        "models": [
            {
                "key": m.key,
                "name": m.name,
                "task": m.task,
                "engine": m.engine,
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
    """显式将指定模型加载到 GPU。"""
    t0 = time.perf_counter()
    cfg = get_model_config(model_key)
    try:
        ensure_model_engine(cfg.key)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("load fail %s | %s", cfg.key, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    cost = time.perf_counter() - t0
    logger.debug("load ok %s | %.2fs", cfg.key, cost)
    return {
        "status": "ok",
        "model_key": cfg.key,
        "model_name": cfg.name,
        "task": cfg.task,
        "engine": cfg.engine,
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
    task: Optional[str] = Form(None),
    categories: str = Form(""),
    phrase: str = Form(""),
    sam3_threshold: Optional[float] = Form(None),
    sam3_points: Optional[str] = Form(None),
    sam3_boxes: Optional[str] = Form(None),
    mask_format: Optional[str] = Form("polygon_norm_pct"),
) -> dict:
    """单图推理。YOLO、LocateAnything 或 SAM3（同进程加载）。"""
    t0 = time.perf_counter()
    fmt = _normalize_box_format(box_format)
    mfmt = _normalize_mask_format(mask_format)
    raw = await image.read()
    filename = getattr(image, "filename", None) or "-"
    img = _parse_image(raw, filename)
    cfg = get_model_config(model_key)
    logger.debug(
        "predict %s | %s | %sx%s | fmt=%s",
        cfg.key,
        filename,
        img.width,
        img.height,
        fmt,
    )

    try:
        result = _run_predict(
            img,
            cfg,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            box_format=fmt,
            task=task,
            categories=categories,
            phrase=phrase,
            sam3_threshold=sam3_threshold,
            sam3_points=sam3_points,
            sam3_boxes=sam3_boxes,
            mask_format=mfmt,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("fail %s | %s", cfg.key, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total = time.perf_counter() - t0
    timings = result.get("timings") or {}
    timings["total"] = total
    result["timings"] = timings
    logger.debug(
        "predict done %s | boxes=%s | segments=%s | %.2fs",
        cfg.key,
        len(result.get("boxes") or []),
        len(result.get("segments") or []),
        total,
    )
    return result


@app.post("/api/predict/segment")
async def predict_segment(
    image: UploadFile = File(...),
    model_key: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    categories: str = Form(""),
    phrase: str = Form(""),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    box_format: Optional[str] = Form("xyxy"),
    mask_format: Optional[str] = Form("polygon_norm_pct"),
    sam3_threshold: Optional[float] = Form(None),
    sam3_points: Optional[str] = Form(None),
    sam3_boxes: Optional[str] = Form(None),
) -> dict:
    """分割专用：YOLO instance segment 或 SAM3（文本/正负点/框提示）。

    sam3_points JSON 示例: [{"x":120,"y":80,"label":1},{"x":200,"y":90,"label":0}]
    label>0 正点，0 负点。可与 sam3_boxes、phrase/categories 组合。
    """
    cfg = get_model_config(model_key)
    eng = (cfg.engine or "yolo").lower()
    task_eff = (task or "").strip().lower()
    if eng == "sam3" and not task_eff:
        if sam3_points or sam3_boxes:
            task_eff = "sam3_point"
        else:
            task_eff = "detect"
    elif (cfg.task or "").lower() == "segment" and not task_eff:
        task_eff = "segment"

    return await predict(
        image=image,
        model_key=model_key,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        box_format=box_format,
        task=task_eff or task,
        categories=categories,
        phrase=phrase,
        sam3_threshold=sam3_threshold,
        sam3_points=sam3_points,
        sam3_boxes=sam3_boxes,
        mask_format=mask_format,
    )


@app.post("/api/predict/batch")
async def predict_batch(
    images: List[UploadFile] = File(...),
    image_ids: Optional[str] = Form(None),
    model_key: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    box_format: Optional[str] = Form("xyxy"),
    mask_format: Optional[str] = Form("polygon_norm_pct"),
) -> dict:
    """批量推理。同一 model_key；单次最多 BATCH_MAX_IMAGES 张。

    image_ids: JSON 数组或逗号分隔，数量须与 images 一致；省略则用 "0","1",...
    单张失败不中断整批，该条 ok=false 并带 error。
    """
    t0 = time.perf_counter()
    fmt = _normalize_box_format(box_format)
    mfmt = _normalize_mask_format(mask_format)
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
    if is_vlm_engine(cfg):
        raise HTTPException(
            status_code=400,
            detail=f"模型 {cfg.key} 为 Locate/SAM3，请用单图 /api/predict 或 /api/predict/segment",
        )
    logger.debug("batch %s | n=%s | fmt=%s", cfg.key, n, fmt)

    try:
        ensure_model_engine(cfg.key)
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
            one = _run_predict(
                img,
                cfg,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                box_format=fmt,
                mask_format=mfmt,
            )
            item.update(
                {
                    "ok": True,
                    "task": one.get("task"),
                    "annotation_type": one.get("annotation_type"),
                    "image_width": one.get("image_width"),
                    "image_height": one.get("image_height"),
                    "boxes": one.get("boxes") or [],
                    "segments": one.get("segments") or [],
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
    logger.debug("batch done %s | ok=%s/%s | %.2fs", cfg.key, ok_n, n, total)
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
