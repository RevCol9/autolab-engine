"""SAM3 引擎"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from app.engines.base import BaseEngine
from app.engines.locate_engine import _parse_categories
from app.engines.sam3_worker import Sam3Worker, parse_boxes, parse_points
from app.mask_format import SUPPORTED_MASK_FORMATS
from app.settings import ModelConfig, Settings


class Sam3Engine(BaseEngine):
    def __init__(self, config: ModelConfig, settings: Settings):
        super().__init__(config)
        self.settings = settings
        self.worker: Optional[Sam3Worker] = None

    def load(self) -> None:
        threshold = float(self.config.conf if self.config.conf else 0.25)
        self.worker = Sam3Worker(
            resource_dir=self.config.path,
            device=self.config.device,
            threshold=threshold,
        )

    def predict(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        if self.worker is None:
            raise RuntimeError("SAM3 未加载")

        task = str(kwargs.get("task") or "detect").strip().lower()
        categories = kwargs.get("categories") or ""
        phrase = str(kwargs.get("phrase") or "")
        threshold = kwargs.get("sam3_threshold")
        if threshold is None:
            threshold = kwargs.get("conf")
        points_json = kwargs.get("sam3_points")
        boxes_json = kwargs.get("sam3_boxes")
        mask_format = str(kwargs.get("mask_format") or "polygon_norm_pct").strip().lower()
        if mask_format not in SUPPORTED_MASK_FORMATS:
            raise ValueError(f"不支持的 mask_format: {mask_format!r}，可选 {SUPPORTED_MASK_FORMATS}")

        points = parse_points(points_json)
        prompt_boxes = parse_boxes(boxes_json)

        prompt = ""
        if task in ("sam3_point", "point", "interactive"):
            if not points and not prompt_boxes:
                raise ValueError("sam3_point 需要 sam3_points 和/或 sam3_boxes")
            cats = _parse_categories(categories)
            prompt = phrase.strip() or (cats[0] if cats else "")
        elif task == "sam3_text":
            cats = _parse_categories(categories)
            prompt = phrase.strip() or (cats[0] if cats else "")
            if not prompt and not points and not prompt_boxes:
                raise ValueError("SAM3 prompt 为空")
        elif task == "detect":
            cats = _parse_categories(categories)
            if not cats and not points and not prompt_boxes:
                raise ValueError("categories 为空且未提供交互点/框")
            prompt = cats[0] if cats else phrase.strip()
        elif task in {"ground_multi", "ground_single", "ground_text", "ground_gui"}:
            prompt = phrase.strip()
            if not prompt:
                raise ValueError("phrase 为空")
        else:
            raise ValueError(
                f"SAM3 不支持 task: {task}；可选 detect, sam3_text, sam3_point, ground_*"
            )

        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size

        result = self.worker.predict_text(
            image,
            prompt,
            label=prompt or "sam3",
            threshold=threshold,
            points=points,
            prompt_boxes=prompt_boxes,
            mask_format=mask_format,
        )
        segments = result.get("segments") or []
        return {
            "task": task,
            "annotation_type": "polygon",
            "segmentation_mode": "instance",
            "mask_format": mask_format,
            "image_width": w,
            "image_height": h,
            "boxes": result.get("boxes") or [],
            "segments": segments,
            "points": points,
            "masks": [],
            "answer": result.get("answer"),
            "sam3": result.get("sam3"),
            "timings": result.get("timings") or {},
        }
