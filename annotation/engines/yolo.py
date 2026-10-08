"""从认证的 .niii-model 加载 YOLO 检测与分割模型。"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from annotation.engines.base import BaseEngine
from annotation.settings import ModelConfig

logger = logging.getLogger(__name__)


class YoloDetectEngine(BaseEngine):
    """task=detect。"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def load(self) -> None:
        path = Path(self.config.path)
        if not self.config.path or not path.is_file():
            raise FileNotFoundError(
                f"模型权重不存在: {self.config.path!r}。"
                "请在 config/annotation.yaml 的 models[].path 填写有效 .niii-model 路径。"
            )
        try:
            from toolkit.model_crypto import load_yolo_container
        except ImportError as exc:
            raise ImportError(
                "加密 YOLO 运行时依赖未安装，请检查 ultralytics/torch/cryptography"
            ) from exc

        t0 = time.perf_counter()
        device = self.config.device
        allow_cpu_for_tests = os.environ.get("NIII_ALLOW_CPU_MODEL_TESTS") == "1"
        self.model = load_yolo_container(
            path,
            device,
            self.config.task,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        logger.debug(
            "load  %s | task=%s | device=%s | %.2fs",
            path,
            self.config.task,
            device,
            time.perf_counter() - t0,
        )

    def classes(self) -> List[str]:
        if self.model is None:
            return []
        names = getattr(self.model, "names", None) or {}
        if isinstance(names, (list, tuple)):
            return [str(value) for value in names]
        if not isinstance(names, dict):
            return []
        values = []
        for key, value in names.items():
            try:
                order = (0, int(key))
            except (TypeError, ValueError):
                order = (1, str(key))
            values.append((order, str(value)))
        return [value for _, value in sorted(values, key=lambda item: item[0])]

    def _predict_raw(
        self,
        image: Image.Image,
        *,
        conf: Optional[float],
        iou: Optional[float],
        imgsz: Optional[int],
        max_det: Optional[int],
        **options: Any,
    ) -> tuple[int, int, list[Any], float]:
        if self.model is None:
            raise RuntimeError("模型未加载")
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        started = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=float(self.config.conf if conf is None else conf),
            iou=float(self.config.iou if iou is None else iou),
            imgsz=int(self.config.imgsz if imgsz is None else imgsz),
            max_det=int(self.config.max_det if max_det is None else max_det),
            device=self.config.device,
            verbose=False,
            **options,
        )
        return width, height, results, time.perf_counter() - started

    @staticmethod
    def _boxes_from_result(
        result: Any,
        *,
        width: int,
        height: int,
        coord_space: str,
    ) -> tuple[list[dict[str, Any]], list[Any], list[int], list[float], list[str]]:
        if result is None or result.boxes is None or not len(result.boxes):
            return [], [], [], [], []
        names = result.names or {}
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        class_ids = [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
        scores = [float(value) for value in result.boxes.conf.detach().cpu().tolist()]
        labels = [str(names.get(class_id, class_id)) for class_id in class_ids]
        boxes = []
        for coordinates, class_id, score, label in zip(
            xyxy,
            class_ids,
            scores,
            labels,
        ):
            x1, y1, x2, y2 = coordinates
            box = (
                {
                    "x1": x1 / width * 1000.0,
                    "y1": y1 / height * 1000.0,
                    "x2": x2 / width * 1000.0,
                    "y2": y2 / height * 1000.0,
                }
                if coord_space == "norm1000"
                else {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )
            boxes.append(
                {
                    "label": label,
                    "class_id": class_id,
                    "score": score,
                    **box,
                }
            )
        return boxes, xyxy, class_ids, scores, labels

    def predict(
        self,
        image: Image.Image,
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        coord_space: str = "pixel",
    ) -> Dict[str, Any]:
        w, h, results, infer_s = self._predict_raw(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
        )
        boxes, *_ = self._boxes_from_result(
            results[0] if results else None,
            width=w,
            height=h,
            coord_space=coord_space,
        )

        return {
            "task": "detect",
            "image_width": w,
            "image_height": h,
            "boxes": boxes,
            "masks": [],
            "timings": {
                "infer": infer_s,
                "total": infer_s,
            },
        }

    def unload(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("yolo unload empty_cache: %s", exc)


class YoloSegmentEngine(YoloDetectEngine):
    """YOLO 实例分割（Ultralytics segment）。"""

    def predict(
        self,
        image: Image.Image,
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        coord_space: str = "pixel",
        mask_format: str = "polygon_norm_pct",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from annotation.mask_format import SUPPORTED_MASK_FORMATS, segments_from_yolo_polys

        if mask_format not in SUPPORTED_MASK_FORMATS:
            raise ValueError(f"不支持的 mask_format: {mask_format!r}")

        w, h, results, infer_s = self._predict_raw(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            retina_masks=True,
        )

        boxes: List[Dict[str, Any]] = []
        segments: List[Dict[str, Any]] = []
        if results:
            r0 = results[0]
            boxes, xyxy, class_ids, score_list, labels = self._boxes_from_result(
                r0,
                width=w,
                height=h,
                coord_space=coord_space,
            )
            if boxes:
                polys: List[Any] = []
                if r0.masks is not None:
                    polys = list(r0.masks.xy) if hasattr(r0.masks, "xy") else []
                if polys:
                    segments = segments_from_yolo_polys(
                        polys,
                        image_width=w,
                        image_height=h,
                        mask_format=mask_format,
                        labels=labels,
                        class_ids=class_ids,
                        scores=score_list,
                        boxes_xyxy=xyxy,
                    )
                elif r0.masks is not None and hasattr(r0.masks, "data"):
                    from annotation.mask_format import segments_from_bool_masks

                    md = r0.masks.data.detach().cpu().numpy()
                    segments = segments_from_bool_masks(
                        [md[i] > 0.5 for i in range(len(md))],
                        image_width=w,
                        image_height=h,
                        mask_format=mask_format,
                        labels=labels,
                        scores=score_list,
                        boxes_xyxy=xyxy,
                        class_ids=class_ids,
                    )
                else:
                    segments = segments_from_yolo_polys(
                        [],
                        image_width=w,
                        image_height=h,
                        mask_format=mask_format,
                        labels=labels,
                        class_ids=class_ids,
                        scores=score_list,
                        boxes_xyxy=xyxy,
                    )

        return {
            "task": "segment",
            "annotation_type": "polygon",
            "segmentation_mode": "instance",
            "mask_format": mask_format,
            "image_width": w,
            "image_height": h,
            "boxes": boxes,
            "segments": segments,
            "masks": [],
            "timings": {
                "infer": infer_s,
                "total": infer_s,
            },
        }
