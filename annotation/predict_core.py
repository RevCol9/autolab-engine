"""推理核心：单图预测、格式归一化、批量 ID 解析。

HTTP 层（api/inference.py）负责读图与参数校验；本模块负责：
  1. run_predict_locked：线程锁 + GPU 文件锁后调用引擎
  2. run_predict_unlocked：按 engine 分发 YOLO / VLM，统一 box/mask 坐标格式
  3. parse_image / parse_image_ids：multipart 输入解析
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from PIL import Image

from annotation.bootstrap import SETTINGS
from annotation.box_format import apply_box_format
from annotation.box_format import SUPPORTED_BOX_FORMATS
from annotation.engines.base import BaseEngine
from annotation.mask_format import SUPPORTED_MASK_FORMATS
from annotation.registry import MODEL_RUNTIME
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


def validate_predict_options(
    *,
    conf: Optional[float],
    iou: Optional[float],
    imgsz: Optional[int],
    sam3_threshold: Optional[float] = None,
) -> None:
    """校验所有入口共用的数值推理参数。"""
    for name, value in (("conf", conf), ("iou", iou), ("sam3_threshold", sam3_threshold)):
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise HTTPException(status_code=400, detail=f"{name} 须位于 [0, 1]")
    if imgsz is not None and int(imgsz) <= 0:
        raise HTTPException(status_code=400, detail="imgsz 须为正整数")


def run_predict_unlocked(
    img: Image.Image,
    cfg: ModelConfig,
    engine: BaseEngine,
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
    if cfg.is_vlm:
        try:
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
    with MODEL_RUNTIME.use_engine(cfg) as (_, engine):
        return run_predict_unlocked(img, cfg, engine, **kwargs)


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
