"""检测框格式转换（平台半自动标注对齐）。"""

from __future__ import annotations

from typing import Any, Dict, List


SUPPORTED_BOX_FORMATS = ("xyxy", "cxcywh_pct")


def apply_box_format(
    boxes: List[Dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    box_format: str = "xyxy",
    coord_space: str = "pixel",
) -> List[Dict[str, Any]]:
    """将引擎输出的框转为约定格式。

    - xyxy：保持引擎输出（受 config inference.coord_space 影响：pixel 或 norm1000）
    - cxcywh_pct：转为相对原图的中心点宽高百分比（0~100），对齐平台标注列表展示
    """
    fmt = (box_format or "xyxy").strip().lower()
    if fmt not in SUPPORTED_BOX_FORMATS:
        raise ValueError(f"不支持的 box_format: {box_format!r}，可选 {SUPPORTED_BOX_FORMATS}")

    if fmt == "xyxy":
        return [
            {
                "label": b.get("label"),
                "class_id": b.get("class_id"),
                "score": b.get("score"),
                "x1": float(b["x1"]),
                "y1": float(b["y1"]),
                "x2": float(b["x2"]),
                "y2": float(b["y2"]),
            }
            for b in boxes or []
        ]

    w = float(image_width) if image_width else 1.0
    h = float(image_height) if image_height else 1.0
    space = (coord_space or "pixel").strip().lower()

    out: List[Dict[str, Any]] = []
    for b in boxes or []:
        x1 = float(b["x1"])
        y1 = float(b["y1"])
        x2 = float(b["x2"])
        y2 = float(b["y2"])
        if space == "norm1000":
            x1 = x1 / 1000.0 * w
            y1 = y1 / 1000.0 * h
            x2 = x2 / 1000.0 * w
            y2 = y2 / 1000.0 * h

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        out.append(
            {
                "label": b.get("label"),
                "class_id": b.get("class_id"),
                "score": b.get("score"),
                "cx": (x1 + x2) / 2.0 / w * 100.0,
                "cy": (y1 + y2) / 2.0 / h * 100.0,
                "w": bw / w * 100.0,
                "h": bh / h * 100.0,
                "unit": "percent",
            }
        )
    return out
