"""实例/分割 mask → 平台多边形等结构化输出。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SUPPORTED_MASK_FORMATS = ("polygon_pixel", "polygon_norm_pct", "polygon_norm")

# polygon_norm：0~1 相对宽高；polygon_norm_pct：0~100（与检测 cxcywh_pct 一致）


def _scale_xy(x: float, y: float, image_width: int, image_height: int, mask_format: str) -> List[float]:
    w = float(image_width) if image_width else 1.0
    h = float(image_height) if image_height else 1.0
    fmt = (mask_format or "polygon_pixel").strip().lower()
    if fmt == "polygon_pixel":
        return [x, y]
    if fmt == "polygon_norm":
        return [x / w, y / h]
    if fmt == "polygon_norm_pct":
        return [x / w * 100.0, y / h * 100.0]
    raise ValueError(f"不支持的 mask_format: {mask_format!r}")


def _scale_polygon(
    points: Sequence[Sequence[float]],
    *,
    image_width: int,
    image_height: int,
    mask_format: str,
) -> List[List[float]]:
    return [_scale_xy(float(pt[0]), float(pt[1]), image_width, image_height, mask_format) for pt in points]


def scale_bbox_xyxy(
    bbox: Optional[Dict[str, float]],
    *,
    image_width: int,
    image_height: int,
    mask_format: str,
) -> Optional[Dict[str, float]]:
    """将像素 xyxy bbox 转到与 polygon 相同的 mask_format。"""
    if not bbox:
        return None
    x1, y1 = _scale_xy(float(bbox["x1"]), float(bbox["y1"]), image_width, image_height, mask_format)
    x2, y2 = _scale_xy(float(bbox["x2"]), float(bbox["y2"]), image_width, image_height, mask_format)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def resize_mask_to_image(mask: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """将任意分辨率二值 mask 拉到原图尺寸，避免按错误 w/h 归一化。"""
    if mask.ndim != 2:
        raise ValueError("mask 必须为 H×W")
    mh, mw = mask.shape
    arr = np.asarray(mask)
    if arr.dtype == bool:
        binary = arr.astype(np.uint8) * 255
    else:
        mx = float(arr.max()) if arr.size else 0.0
        binary = ((arr > 0.5) if mx <= 1.0 else (arr > 0)).astype(np.uint8) * 255
    if mw == image_width and mh == image_height:
        return binary > 0
    import cv2

    resized = cv2.resize(binary, (int(image_width), int(image_height)), interpolation=cv2.INTER_NEAREST)
    return resized > 0


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
    drop_reason: Optional[str] = None,
) -> Dict[str, Any]:
    poly = _scale_polygon(
        polygon_pixel,
        image_width=image_width,
        image_height=image_height,
        mask_format=mask_format,
    )
    if len(poly) < 3:
        poly = []
        drop_reason = drop_reason or "empty_or_degenerate_mask"
    item: Dict[str, Any] = {
        "kind": kind,
        "label": label,
        "score": score,
        "polygon": poly,
    }
    if class_id is not None:
        item["class_id"] = class_id
    scaled_bbox = scale_bbox_xyxy(
        bbox,
        image_width=image_width,
        image_height=image_height,
        mask_format=mask_format,
    )
    if scaled_bbox:
        item["bbox"] = scaled_bbox
    if drop_reason:
        item["drop_reason"] = drop_reason
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
    n = max(len(masks), len(boxes_xyxy) if boxes_xyxy is not None else 0)
    for i in range(n):
        poly_px: List[List[float]] = []
        drop_reason = None
        if i < len(masks):
            mask = resize_mask_to_image(np.asarray(masks[i]), image_width, image_height)
            poly_px = mask_to_polygon_pixel(mask)
            if not poly_px:
                drop_reason = "empty_or_degenerate_mask"
        else:
            drop_reason = "missing_mask"
        label = (labels[i] if labels and i < len(labels) else None) or "object"
        score = scores[i] if scores and i < len(scores) else None
        cid = class_ids[i] if class_ids and i < len(class_ids) else None
        bbox = None
        if boxes_xyxy and i < len(boxes_xyxy):
            b = boxes_xyxy[i]
            bbox = {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}
        segments.append(
            build_segment_item(
                polygon_pixel=poly_px,
                image_width=image_width,
                image_height=image_height,
                mask_format=mask_format,
                label=str(label),
                class_id=cid,
                score=float(score) if score is not None else None,
                bbox=bbox,
                kind="instance",
                drop_reason=drop_reason,
            )
        )
    return segments


def segments_from_yolo_polys(
    polys: Sequence[Any],
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
    n = len(boxes_xyxy)
    for i in range(n):
        poly_arr = polys[i] if i < len(polys) else None
        drop_reason = None
        poly_px: List[List[float]] = []
        if poly_arr is None or len(poly_arr) < 3:
            drop_reason = "empty_or_degenerate_mask"
        else:
            poly_px = [[float(x), float(y)] for x, y in poly_arr]
        bbox = None
        if i < len(boxes_xyxy):
            bbox = {
                "x1": float(boxes_xyxy[i][0]),
                "y1": float(boxes_xyxy[i][1]),
                "x2": float(boxes_xyxy[i][2]),
                "y2": float(boxes_xyxy[i][3]),
            }
        segments.append(
            build_segment_item(
                polygon_pixel=poly_px,
                image_width=image_width,
                image_height=image_height,
                mask_format=mask_format,
                label=str(labels[i]) if i < len(labels) else "object",
                class_id=int(class_ids[i]) if i < len(class_ids) else None,
                score=float(scores[i]) if i < len(scores) else None,
                bbox=bbox,
                kind="instance",
                drop_reason=drop_reason,
            )
        )
    return segments
