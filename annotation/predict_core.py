"""推理核心：单图预测、格式归一化、批量 ID 解析。"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from PIL import Image

from annotation.bootstrap import SETTINGS
from annotation.box_format import apply_box_format
from annotation.mask_format import SUPPORTED_MASK_FORMATS
from annotation.box_format import SUPPORTED_BOX_FORMATS
from annotation.registry import (
    _gpu_lock,
    cross_gpu_session,
    ensure_model_engine,
    is_vlm_engine,
)
from annotation.settings import ModelConfig


def parse_image(raw: bytes, filename: str = "-") -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail=f"空图片: {filename}")
    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析图片 {filename}: {exc}") from exc


def normalize_mask_format(mask_format: Optional[str]) -> str:
    fmt = (mask_format or "polygon_norm_pct").strip().lower()
    if fmt not in SUPPORTED_MASK_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 mask_format: {mask_format!r}，可选 {list(SUPPORTED_MASK_FORMATS)}",
        )
    return fmt


def normalize_box_format(box_format: Optional[str]) -> str:
    fmt = (box_format or "xyxy").strip().lower()
    if fmt not in SUPPORTED_BOX_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 box_format: {box_format!r}，可选 {list(SUPPORTED_BOX_FORMATS)}",
        )
    return fmt


def run_predict_unlocked(
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
        result["mask_format"] = mask_format
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
    result["mask_format"] = mask_format if is_segment else result.get("mask_format")
    result["model_key"] = cfg.key
    result["model_name"] = cfg.name
    result["engine"] = cfg.engine or "yolo"
    return result


def run_predict_locked(
    img: Image.Image,
    cfg: ModelConfig,
    **kwargs: Any,
) -> Dict[str, Any]:
    """进程内线程锁 + 跨进程 GPU 锁，串行化 load/predict。"""
    with _gpu_lock:
        with cross_gpu_session(cfg):
            return run_predict_unlocked(img, cfg, **kwargs)


def parse_image_ids(raw: Optional[str], n: int) -> List[str]:
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
