"""实例/分割 mask → 平台多边形等结构化输出。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SUPPORTED_MASK_FORMATS = ("polygon_pixel", "polygon_norm_pct", "polygon_norm")

# polygon_norm：0~1 相对宽高；polygon_norm_pct：0~100（与检测 cxcywh_pct 一致）


def _scale_polygon(
    points: Sequence[Sequence[float]],
    *,
    image_width: int,
    image_height: int,
    mask_format: str,
) -> List[List[float]]:
    w = float(image_width) if image_width else 1.0
    h = float(image_height) if image_height else 1.0
    fmt = (mask_format or "polygon_pixel").strip().lower()
    out: List[List[float]] = []
    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        if fmt == "polygon_pixel":
            out.append([x, y])
        elif fmt == "polygon_norm":
            out.append([x / w, y / h])
        elif fmt == "polygon_norm_pct":
            out.append([x / w * 100.0, y / h * 100.0])
        else:
            raise ValueError(f"不支持的 mask_format: {mask_format!r}")
    return out


def mask_to_polygon_pixel(mask: np.ndarray, simplify_eps: float = 1.5) -> List[List[float]]:
    """二值 mask → 外轮廓多边形（像素坐标）。"""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("分割 polygon 需要 opencv-python-headless") from exc

    if mask.ndim != 2:
        raise ValueError("mask 必须为 H×W")
    binary = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1:
        return []
    if simplify_eps > 0:
        cnt = cv2.approxPolyDP(cnt, simplify_eps, True)
    return [[float(p[0][0]), float(p[0][1])] for p in cnt]


def build_segment_item(
    *,
    polygon_pixel: List[List[float]],
    image_width: int,
    image_height: int,
    mask_format: str,
    label: str,
    class_id: Optional[int] = None,
    score: Optional[float] = None,
    bbox: Optional[Dict[str, float]] = None,
    kind: str = "instance",
) -> Dict[str, Any]:
    poly = _scale_polygon(
        polygon_pixel,
        image_width=image_width,
        image_height=image_height,
        mask_format=mask_format,
    )
    if len(poly) < 3:
        return {}
    item: Dict[str, Any] = {
        "kind": kind,
        "label": label,
        "score": score,
        "polygon": poly,
    }
    if class_id is not None:
        item["class_id"] = class_id
    if bbox:
        item["bbox"] = bbox
    return item


def segments_from_bool_masks(
    masks: Sequence[np.ndarray],
    *,
    image_width: int,
    image_height: int,
    mask_format: str,
    labels: Optional[Sequence[str]] = None,
    scores: Optional[Sequence[Optional[float]]] = None,
    boxes_xyxy: Optional[Sequence[Sequence[float]]] = None,
    class_ids: Optional[Sequence[Optional[int]]] = None,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    n = len(masks)
    for i in range(n):
        poly_px = mask_to_polygon_pixel(masks[i])
        if not poly_px:
            continue
        label = (labels[i] if labels and i < len(labels) else None) or "object"
        score = scores[i] if scores and i < len(scores) else None
        cid = class_ids[i] if class_ids and i < len(class_ids) else None
        bbox = None
        if boxes_xyxy and i < len(boxes_xyxy):
            b = boxes_xyxy[i]
            bbox = {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}
        item = build_segment_item(
            polygon_pixel=poly_px,
            image_width=image_width,
            image_height=image_height,
            mask_format=mask_format,
            label=str(label),
            class_id=cid,
            score=float(score) if score is not None else None,
            bbox=bbox,
            kind="instance",
        )
        if item:
            segments.append(item)
    return segments


def segments_from_yolo_polys(
    polys: Sequence[np.ndarray],
    *,
    image_width: int,
    image_height: int,
    mask_format: str,
    labels: Sequence[str],
    class_ids: Sequence[int],
    scores: Sequence[float],
    boxes_xyxy: Sequence[Sequence[float]],
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for i, poly_arr in enumerate(polys):
        if poly_arr is None or len(poly_arr) < 3:
            continue
        poly_px = [[float(x), float(y)] for x, y in poly_arr]
        item = build_segment_item(
            polygon_pixel=poly_px,
            image_width=image_width,
            image_height=image_height,
            mask_format=mask_format,
            label=str(labels[i]) if i < len(labels) else "object",
            class_id=int(class_ids[i]) if i < len(class_ids) else None,
            score=float(scores[i]) if i < len(scores) else None,
            bbox={
                "x1": float(boxes_xyxy[i][0]),
                "y1": float(boxes_xyxy[i][1]),
                "x2": float(boxes_xyxy[i][2]),
                "y2": float(boxes_xyxy[i][3]),
            }
            if i < len(boxes_xyxy)
            else None,
            kind="instance",
        )
        if item:
            segments.append(item)
    return segments
