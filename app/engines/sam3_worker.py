"""SAM3 worker adapter for the annotation UI."""

from __future__ import annotations

import logging
import base64
import json
from io import BytesIO
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

SAM3_COLORS = [
    {"stroke": "#16a34a", "rgba": (22, 163, 74, 92), "rgb": (22, 163, 74)},
    {"stroke": "#2563eb", "rgba": (37, 99, 235, 92), "rgb": (37, 99, 235)},
    {"stroke": "#dc2626", "rgba": (220, 38, 38, 92), "rgb": (220, 38, 38)},
    {"stroke": "#d97706", "rgba": (217, 119, 6, 92), "rgb": (217, 119, 6)},
    {"stroke": "#7c3aed", "rgba": (124, 58, 237, 92), "rgb": (124, 58, 237)},
]


def _to_numpy_masks(masks):
    masks = masks.detach().cpu()
    if masks.ndim == 4:
        masks = masks[:, 0]
    return masks.numpy().astype(bool)


def _mask_overlay_data_url(image_size, masks) -> Optional[str]:
    if len(masks) == 0:
        return None

    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    for index, mask in enumerate(masks):
        color_spec = SAM3_COLORS[index % len(SAM3_COLORS)]
        color = color_spec["rgba"]
        color_layer = Image.new("RGBA", image_size, color)
        alpha = Image.fromarray((mask.astype(np.uint8) * color[3]), mode="L")
        overlay = Image.composite(color_layer, overlay, alpha)

        edge_alpha = alpha.filter(ImageFilter.FIND_EDGES).point(lambda v: 210 if v else 0)
        edge_layer = Image.new("RGBA", image_size, (*color_spec["rgb"], 230))
        overlay = Image.composite(edge_layer, overlay, edge_alpha)

    buf = BytesIO()
    overlay.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _require_float(item: dict, key: str, *, index: int, kind: str) -> float:
    if key not in item or item[key] is None:
        raise ValueError(f"SAM3 {kind}[{index}] 缺少字段 {key}")
    try:
        return float(item[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SAM3 {kind}[{index}].{key} 不是数字: {item[key]!r}") from exc


def parse_points(raw: Optional[str]) -> List[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SAM3 points JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("SAM3 points must be a JSON list.")

    points: List[dict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"SAM3 points[{index}] 必须是对象")
        x = _require_float(item, "x", index=index, kind="points")
        y = _require_float(item, "y", index=index, kind="points")
        try:
            label = int(item.get("label", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SAM3 points[{index}].label 非法: {item.get('label')!r}") from exc
        points.append({"x": x, "y": y, "label": 1 if label > 0 else 0})
    return points


def parse_boxes(raw: Optional[str]) -> List[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SAM3 boxes JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("SAM3 boxes must be a JSON list.")

    boxes: List[dict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"SAM3 boxes[{index}] 必须是对象")
        x1 = _require_float(item, "x1", index=index, kind="boxes")
        y1 = _require_float(item, "y1", index=index, kind="boxes")
        x2 = _require_float(item, "x2", index=index, kind="boxes")
        y2 = _require_float(item, "y2", index=index, kind="boxes")
        try:
            label = int(item.get("label", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SAM3 boxes[{index}].label 非法: {item.get('label')!r}") from exc
        boxes.append(
            {
                "x1": min(x1, x2),
                "y1": min(y1, y2),
                "x2": max(x1, x2),
                "y2": max(y1, y2),
                "label": 1 if label > 0 else 0,
            }
        )
    return boxes


class Sam3Worker:
    """Thin adapter around the local SAM3 image model."""

    def __init__(
        self,
        resource_dir: str,
        device: str = "cuda:0",
        threshold: float = 0.5,
    ):
        self.resource_dir = Path(resource_dir).expanduser().resolve()
        self.checkpoint = self.resource_dir / "sam3.pt"
        text = str(device or "cpu").strip()
        self.device = text if text.startswith("cuda") else "cpu"
        self.threshold = float(threshold)

        if not self.resource_dir.is_dir():
            raise FileNotFoundError(f"SAM3 resource directory not found: {self.resource_dir}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint}")
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM3 requested CUDA, but torch.cuda.is_available() is False.")

        sam3_repo = self.resource_dir.parent / "sam3"
        if sam3_repo.is_dir() and str(sam3_repo) not in sys.path:
            sys.path.insert(0, str(sam3_repo))

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        t0 = time.perf_counter()
        logger.debug("sam3  load checkpoint=%s | device=%s", self.checkpoint, self.device)
        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint),
            load_from_HF=False,
            device=self.device,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.threshold,
        )
        self.load_seconds = time.perf_counter() - t0
        logger.debug("sam3  done in %.1fs", self.load_seconds)

    def set_threshold(self, threshold: Optional[float]) -> None:
        if threshold is None:
            return
        self.threshold = float(threshold)
        if hasattr(self.processor, "set_confidence_threshold"):
            self.processor.set_confidence_threshold(self.threshold)

    def predict_text(
        self,
        image: Image.Image,
        prompt: str,
        label: Optional[str] = None,
        threshold: Optional[float] = None,
        points: Optional[List[dict]] = None,
        prompt_boxes: Optional[List[dict]] = None,
        mask_format: str = "polygon_norm_pct",
    ) -> dict:
        prompt = prompt.strip()
        points = points or []
        prompt_boxes = prompt_boxes or []
        if not prompt and not points and not prompt_boxes:
            raise ValueError("SAM3 needs a text prompt, interactive points, or boxes.")

        self.set_threshold(threshold)
        t0 = time.perf_counter()
        state = self.processor.set_image(image)
        if prompt:
            state = self.processor.set_text_prompt(state=state, prompt=prompt)
        elif "language_features" not in state["backbone_out"]:
            text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(text_outputs)
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        if points:
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = self.model._get_dummy_prompt()
            for point in points:
                nx = max(0.0, min(1.0, float(point["x"]) / image.width))
                ny = max(0.0, min(1.0, float(point["y"]) / image.height))
                point_tensor = torch.tensor([[[nx, ny]]], device=self.device, dtype=torch.float32)
                label_tensor = torch.tensor([[int(point["label"])]], device=self.device, dtype=torch.long)
                state["geometric_prompt"].append_points(point_tensor, label_tensor)

        if prompt_boxes:
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = self.model._get_dummy_prompt()
            for box in prompt_boxes:
                x1 = max(0.0, min(float(image.width), float(box["x1"])))
                y1 = max(0.0, min(float(image.height), float(box["y1"])))
                x2 = max(0.0, min(float(image.width), float(box["x2"])))
                y2 = max(0.0, min(float(image.height), float(box["y2"])))
                cx = ((x1 + x2) / 2.0) / image.width
                cy = ((y1 + y2) / 2.0) / image.height
                bw = max(1.0, x2 - x1) / image.width
                bh = max(1.0, y2 - y1) / image.height
                box_tensor = torch.tensor([cx, cy, bw, bh], device=self.device, dtype=torch.float32).view(1, 1, 4)
                label_tensor = torch.tensor([bool(box["label"])], device=self.device, dtype=torch.bool).view(1, 1)
                state["geometric_prompt"].append_boxes(box_tensor, label_tensor)

        if points or prompt_boxes:
            output = self.processor._forward_grounding(state)
        else:
            output = state

        masks = _to_numpy_masks(output["masks"])
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()

        label = label or prompt
        parsed_boxes: List[dict] = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(float(image.width), x1))
            x2 = max(0.0, min(float(image.width), x2))
            y1 = max(0.0, min(float(image.height), y1))
            y2 = max(0.0, min(float(image.height), y2))
            score = float(scores[index]) if index < len(scores) else None
            color = SAM3_COLORS[index % len(SAM3_COLORS)]["stroke"]
            parsed_boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "score": score,
                    "color": color,
                }
            )

        mask_areas = [int(mask.sum()) for mask in masks]
        elapsed = time.perf_counter() - t0

        from app.mask_format import segments_from_bool_masks

        box_list = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in parsed_boxes]
        label_list = [b.get("label") or label for b in parsed_boxes]
        score_list = [b.get("score") for b in parsed_boxes]
        segments = segments_from_bool_masks(
            masks,
            image_width=image.width,
            image_height=image.height,
            mask_format=mask_format,
            labels=label_list,
            scores=score_list,
            boxes_xyxy=box_list,
        )

        return {
            "answer": f"SAM3 prompt={prompt!r}, instances={len(parsed_boxes)}",
            "boxes": parsed_boxes,
            "points": [],
            "segments": segments,
            "annotation_type": "polygon",
            "segmentation_mode": "instance",
            "sam3": {
                "prompt": prompt,
                "checkpoint": str(self.checkpoint),
                "threshold": self.threshold,
                "mask_areas": mask_areas,
                "mask_overlay": _mask_overlay_data_url(image.size, masks),
                "points": points,
                "boxes_prompt": prompt_boxes,
                "scores": [float(v) for v in scores],
            },
            "timings": {
                "total_request": elapsed,
                "generate": elapsed,
            },
        }
