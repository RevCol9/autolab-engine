"""推理 HTTP API（:21010）。

路由委托 annotation/predict_core 与 registry；配置来自 annotation/config.yaml。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from annotation import runtime_state
from annotation.bootstrap import SETTINGS
from annotation.box_format import SUPPORTED_BOX_FORMATS
from annotation.mask_format import SUPPORTED_MASK_FORMATS
from annotation.predict_core import (
    normalize_box_format,
    normalize_mask_format,
    parse_image,
    parse_image_ids,
    run_predict_locked,
)
from annotation.registry import (
    BATCH_MAX_IMAGES,
    GPU_LOAD_TIMEOUT,
    _engines,
    _gpu_lock,
    active_key,
    cross_gpu_session,
    ensure_model_engine,
    get_model_config,
    is_vlm_engine,
    set_active_key,
)
from annotation.test_ui import test_page
from shared.openapi_docs import openapi_description

# 供测试与外部脚本复用
from annotation.registry import ensure_model_engine, get_model_config  # noqa: F401

logger = logging.getLogger(__name__)

app = FastAPI(
    title="autolab-engine",
    version="0.3.1",
    description=openapi_description("infer", summary="推理 / 半自动标注"),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    default = get_model_config(None)
    if default.path and not is_vlm_engine(default):
        try:
            with _gpu_lock:
                with cross_gpu_session(default, timeout=GPU_LOAD_TIMEOUT):
                    ensure_model_engine(default.key)
        except HTTPException as exc:
            logger.warning("startup skip load default=%s: %s", default.key, exc.detail)
        except Exception as exc:
            logger.warning("startup skip load default=%s: %s", default.key, exc)


@app.get("/test", include_in_schema=True)
def test_inference_page():
    """浏览器联调：选 model_key、上传图片、查看 JSON 与框预览。"""
    return test_page()


@app.get("/api/health")
def health() -> dict:
    cuda = runtime_state.probe_cuda()
    weight = runtime_state.probe_default_weight(SETTINGS)
    gpu_locks = runtime_state.probe_gpu_locks(SETTINGS)
    degraded = not cuda.get("available") or not weight.get("ok")
    return {
        "status": "degraded" if degraded else "ok",
        "default_model": SETTINGS.default_model,
        "active_model": active_key(),
        "loaded": list(_engines.keys()),
        "cuda": cuda,
        "default_weight": weight,
        "gpu_locks": gpu_locks,
        "models": [
            runtime_state.model_entry(m, loaded=m.key in _engines)
            for m in SETTINGS.models
        ],
    }


@app.get("/api/models")
def models() -> dict:
    """返回 config.yaml 中的模型清单（不暴露权重绝对路径）。"""
    return {
        "default_model": SETTINGS.default_model,
        "loaded": list(_engines.keys()),
        "box_formats": list(SUPPORTED_BOX_FORMATS),
        "mask_formats": list(SUPPORTED_MASK_FORMATS),
        "models": [
            runtime_state.model_entry(m, loaded=m.key in _engines)
            for m in SETTINGS.models
        ],
    }


@app.get("/api/models/{model_key}/classes")
def model_classes(model_key: str) -> dict:
    """返回已加载或按需加载后的类别列表。"""
    cfg = get_model_config(model_key)
    with _gpu_lock:
        with cross_gpu_session(cfg, timeout=GPU_LOAD_TIMEOUT):
            engine = ensure_model_engine(cfg.key)
            return {
                "model_key": cfg.key,
                "classes": engine.classes(),
                "loaded": list(_engines.keys()),
            }


@app.post("/api/models/{model_key}/load")
def load_model(model_key: str) -> dict:
    """显式将指定模型加载到 GPU。"""
    t0 = time.perf_counter()
    cfg = get_model_config(model_key)
    try:
        with _gpu_lock:
            with cross_gpu_session(cfg, timeout=GPU_LOAD_TIMEOUT):
                engine = ensure_model_engine(cfg.key)
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
        "classes": engine.classes(),
        "loaded": list(_engines.keys()),
        "timings": {"load": cost},
    }


@app.post("/api/models/{model_key}/unload")
def unload_model(model_key: str) -> dict:
    """从 GPU 卸载指定模型，释放显存。"""
    cfg = get_model_config(model_key)
    with _gpu_lock:
        if cfg.key not in _engines:
            return {
                "status": "ok",
                "model_key": cfg.key,
                "unloaded": False,
                "message": "模型未在内存中",
                "loaded": list(_engines.keys()),
            }
        loaded, new_active = runtime_state.unload_engine(cfg.key, _engines, active_key())
        set_active_key(new_active)
    return {
        "status": "ok",
        "model_key": cfg.key,
        "unloaded": True,
        "loaded": loaded,
        "active_model": active_key(),
    }


@app.post("/api/models/unload_all")
def unload_all_models() -> dict:
    """卸载所有已加载模型。"""
    with _gpu_lock:
        count = len(_engines)
        runtime_state.unload_all_engines(_engines)
        set_active_key(None)
    return {"status": "ok", "unloaded_count": count, "loaded": []}


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
    fmt = normalize_box_format(box_format)
    mfmt = normalize_mask_format(mask_format)
    raw = await image.read()
    filename = getattr(image, "filename", None) or "-"
    img = parse_image(raw, filename)
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
        result = await asyncio.to_thread(
            run_predict_locked,
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
    """分割专用：YOLO instance segment 或 SAM3（文本/正负点/框提示）。"""
    cfg = get_model_config(model_key)
    eng = (cfg.engine or "yolo").lower()
    task_cfg = (cfg.task or "").lower()
    if eng not in {"sam3"} and task_cfg != "segment":
        raise HTTPException(
            status_code=400,
            detail=(
                f"模型 {cfg.key} 不支持分割接口（engine={eng}, task={task_cfg}）；"
                "请使用 task=segment 的 YOLO 或 engine=sam3"
            ),
        )

    task_eff = (task or "").strip().lower()
    if eng == "sam3" and not task_eff:
        task_eff = "sam3_point" if (sam3_points or sam3_boxes) else "detect"
    elif task_cfg == "segment" and not task_eff:
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
    """批量推理。同一 model_key；单次最多 BATCH_MAX_IMAGES 张。"""
    t0 = time.perf_counter()
    fmt = normalize_box_format(box_format)
    mfmt = normalize_mask_format(mask_format)
    n = len(images)
    if n == 0:
        raise HTTPException(status_code=400, detail="images 不能为空")
    if n > BATCH_MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"单次批量最多 {BATCH_MAX_IMAGES} 张，当前 {n}",
        )

    ids = parse_image_ids(image_ids, n)
    cfg = get_model_config(model_key)
    if is_vlm_engine(cfg):
        raise HTTPException(
            status_code=400,
            detail=f"模型 {cfg.key} 为 Locate/SAM3，请用单图 /api/predict 或 /api/predict/segment",
        )
    logger.debug("batch %s | n=%s | fmt=%s", cfg.key, n, fmt)

    try:
        with _gpu_lock:
            with cross_gpu_session(cfg, timeout=GPU_LOAD_TIMEOUT):
                ensure_model_engine(cfg.key)
    except HTTPException as exc:
        raise exc
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
            img = parse_image(raw, filename)
            one = await asyncio.to_thread(
                run_predict_locked,
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
                    "engine": one.get("engine") or cfg.engine or "yolo",
                    "mask_format": one.get("mask_format") or mfmt,
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
            item.update(
                {
                    "ok": False,
                    "error": exc.detail,
                    "engine": cfg.engine or "yolo",
                    "mask_format": mfmt,
                    "boxes": [],
                    "segments": [],
                    "masks": [],
                }
            )
        except Exception as exc:
            logger.exception("batch item fail %s | %s | %s", cfg.key, image_id, exc)
            item.update(
                {
                    "ok": False,
                    "error": str(exc),
                    "engine": cfg.engine or "yolo",
                    "mask_format": mfmt,
                    "boxes": [],
                    "segments": [],
                    "masks": [],
                }
            )
        results.append(item)

    total = time.perf_counter() - t0
    logger.debug("batch done %s | ok=%s/%s | %.2fs", cfg.key, ok_n, n, total)
    return {
        "model_key": cfg.key,
        "model_name": cfg.name,
        "task": cfg.task,
        "engine": cfg.engine or "yolo",
        "box_format": fmt,
        "mask_format": mfmt,
        "count": n,
        "ok_count": ok_n,
        "results": results,
        "timings": {"total": total},
    }
