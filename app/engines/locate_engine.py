"""LocateAnything 引擎"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from app.engines.base import BaseEngine
from app.engines.box_postprocess import (
    build_detect_prompt,
    build_ground_gui_prompt,
    build_ground_multi_prompt,
    build_ground_single_prompt,
    build_ground_text_prompt,
    build_point_prompt,
    refine_detections,
)
from app.engines.locate_worker import LocateAnythingWorker
from app.settings import ModelConfig, Settings

logger = logging.getLogger(__name__)


class LocateEngine(BaseEngine):
    def __init__(self, config: ModelConfig, settings: Settings):
        super().__init__(config)
        self.settings = settings
        self.worker: Optional[LocateAnythingWorker] = None

    def load(self) -> None:
        s = self.settings
        max_memory = s.max_memory
        if max_memory and "_all" in max_memory:
            import torch

            n = torch.cuda.device_count() if torch.cuda.is_available() else 1
            from app.settings import expand_max_memory

            max_memory = expand_max_memory(max_memory, n)
        self.worker = LocateAnythingWorker(
            model_path=self.config.path,
            device=self.config.device,
            dtype=s.locate_dtype,
            local_files_only=s.locate_local_files_only,
            device_map=s.device_map,
            max_memory=max_memory,
            in_token_limit=s.in_token_limit,
            max_image_side=s.max_image_side,
        )

    def predict(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        if self.worker is None:
            raise RuntimeError("LocateAnything 未加载")

        task = str(kwargs.get("task") or "ground_multi").strip().lower()
        categories = kwargs.get("categories") or ""
        phrase = str(kwargs.get("phrase") or "")
        output_type = str(kwargs.get("output_type") or "box")
        generation_mode = kwargs.get("generation_mode") or self.settings.generation_mode
        max_new_tokens = int(
            kwargs.get("max_new_tokens")
            if kwargs.get("max_new_tokens") is not None
            else self.settings.max_new_tokens
        )
        temperature = float(
            kwargs.get("temperature")
            if kwargs.get("temperature") is not None
            else self.settings.temperature
        )
        verbose = bool(kwargs.get("verbose", False))
        bilingual = self.settings.bilingual_prompt

        common = {
            "generation_mode": generation_mode,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "verbose": verbose,
        }

        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size

        if task == "detect":
            cats = _parse_categories(categories)
            if not cats:
                raise ValueError("categories 为空")
            result = self.worker.predict(image, build_detect_prompt(cats, bilingual), **common)
        elif task == "ground_multi":
            if not phrase.strip():
                raise ValueError("phrase 为空")
            result = self.worker.predict(
                image, build_ground_multi_prompt(phrase.strip(), bilingual), **common
            )
        elif task == "ground_single":
            if not phrase.strip():
                raise ValueError("phrase 为空")
            result = self.worker.predict(
                image, build_ground_single_prompt(phrase.strip(), bilingual), **common
            )
        elif task == "detect_text":
            result = self.worker.predict(image, "Detect all the text in box format.", **common)
        elif task == "ground_text":
            if not phrase.strip():
                raise ValueError("phrase 为空")
            result = self.worker.predict(
                image, build_ground_text_prompt(phrase.strip(), bilingual), **common
            )
        elif task == "ground_gui":
            if not phrase.strip():
                raise ValueError("phrase 为空")
            result = self.worker.predict(
                image,
                build_ground_gui_prompt(phrase.strip(), output_type=output_type, bilingual=bilingual),
                **common,
            )
        elif task == "point":
            if not phrase.strip():
                raise ValueError("phrase 为空")
            result = self.worker.predict(image, build_point_prompt(phrase.strip(), bilingual), **common)
        else:
            raise ValueError(f"不支持的 task: {task}")

        answer = result["answer"]
        raw_boxes = LocateAnythingWorker.parse_boxes(answer, width, height)
        points = LocateAnythingWorker.parse_points(answer, width, height)
        if task == "detect":
            preferred_label = ",".join(_parse_categories(categories)[:3])
        else:
            preferred_label = phrase.strip()

        s = self.settings
        refined = refine_detections(
            raw_boxes,
            width,
            height,
            answer=answer,
            max_new_tokens=max_new_tokens,
            enable=s.postprocess_enable,
            nms_iou=s.postprocess_nms_iou,
            runaway_nms_iou=s.postprocess_runaway_nms_iou,
            min_box_area_ratio=s.postprocess_min_box_area_ratio,
            max_box_area_ratio=s.postprocess_max_box_area_ratio,
            max_boxes=s.postprocess_max_boxes,
            preferred_label=preferred_label,
        )

        return {
            "task": task,
            "answer": answer,
            "image_width": width,
            "image_height": height,
            "boxes": refined["boxes"],
            "points": points,
            "masks": [],
            "postprocess": {
                "raw_box_count": refined["raw_box_count"],
                "kept_box_count": refined["kept_box_count"],
                "runaway_suspected": refined["runaway_suspected"],
                "postprocess_applied": refined["postprocess_applied"],
                "nms_iou_used": refined.get("nms_iou_used"),
                "truncated_by_max_boxes": refined["truncated_by_max_boxes"],
            },
            "timings": result.get("timings") or {},
        }

    def unload(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.model = None
            worker.processor = None
            worker.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("locate unload empty_cache: %s", exc)


def _parse_categories(categories: str) -> List[str]:
    raw = str(categories or "").replace("，", ",").replace("；", ",").replace(";", ",")
    parts: List[str] = []
    for chunk in raw.split(","):
        for line in chunk.splitlines():
            item = line.strip()
            if item:
                parts.append(item)
    return parts
