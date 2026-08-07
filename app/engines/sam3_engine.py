"""SAM3 引擎"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from app.engines.base import BaseEngine
from app.engines.locate_engine import _parse_categories
from app.engines.sam3_worker import Sam3Worker, parse_boxes, parse_points
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

        points = parse_points(points_json)
        prompt_boxes = parse_boxes(boxes_json)

        if task == "sam3_text":
            cats = _parse_categories(categories)
            prompt = phrase.strip() or (cats[0] if cats else "")
            if not prompt and not points and not prompt_boxes:
                raise ValueError("SAM3 prompt 为空")
        elif task == "detect":
            cats = _parse_categories(categories)
            if not cats:
                raise ValueError("categories 为空")
            prompt = cats[0]
        elif task in {"ground_multi", "ground_single", "ground_text", "ground_gui", "point"}:
            prompt = phrase.strip()
            if not prompt:
                raise ValueError("phrase 为空")
        else:
            raise ValueError(f"SAM3 不支持 task: {task}")

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
        )
        return {
            "task": task,
            "image_width": w,
            "image_height": h,
            "boxes": result.get("boxes") or [],
            "points": result.get("points") or [],
            "masks": [],
            "answer": result.get("answer"),
            "sam3": result.get("sam3"),
            "timings": result.get("timings") or {},
        }
