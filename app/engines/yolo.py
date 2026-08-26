"""YOLO 检测引擎（Ultralytics .pt；.onnx 同路径加载）。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from app.engines.base import BaseEngine
from app.settings import ModelConfig

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
                "请在 config.yaml 的 models[].path 填写有效 .pt/.onnx 路径。"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "未安装 ultralytics。请使用 YOLO 环境或: pip install ultralytics"
            ) from exc

        t0 = time.perf_counter()
        self.model = YOLO(str(path))
        # 触发一次设备绑定（不跑真实推理）
        device = self.config.device
        if hasattr(self.model, "to") and device:
            try:
                self.model.to(device)
            except Exception as exc:
                logger.warning("model.to(%s) failed for %s: %s", device, path, exc)
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
        if self.model is None:
            raise RuntimeError("模型未加载")
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        conf_v = float(self.config.conf if conf is None else conf)
        iou_v = float(self.config.iou if iou is None else iou)
        imgsz_v = int(self.config.imgsz if imgsz is None else imgsz)
        max_det_v = int(self.config.max_det if max_det is None else max_det)

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_v,
            iou=iou_v,
            imgsz=imgsz_v,
            max_det=max_det_v,
            device=self.config.device,
            verbose=False,
        )
        infer_s = time.perf_counter() - t0

        boxes: List[Dict[str, Any]] = []
        if results:
            r0 = results[0]
            names = r0.names or {}
            if r0.boxes is not None and len(r0.boxes):
                xyxy = r0.boxes.xyxy.detach().cpu().tolist()
                cls = r0.boxes.cls.detach().cpu().tolist()
                confs = r0.boxes.conf.detach().cpu().tolist()
                for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
                    cid = int(c)
                    label = str(names.get(cid, cid))
                    if coord_space == "norm1000":
                        box = {
                            "x1": x1 / w * 1000.0,
                            "y1": y1 / h * 1000.0,
                            "x2": x2 / w * 1000.0,
                            "y2": y2 / h * 1000.0,
                        }
                    else:
                        box = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                    boxes.append(
                        {
                            "label": label,
                            "class_id": cid,
                            "score": float(s),
                            **box,
                        }
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
        from app.mask_format import SUPPORTED_MASK_FORMATS, segments_from_yolo_polys

        if mask_format not in SUPPORTED_MASK_FORMATS:
            raise ValueError(f"不支持的 mask_format: {mask_format!r}")

        if self.model is None:
            raise RuntimeError("模型未加载")
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        conf_v = float(self.config.conf if conf is None else conf)
        iou_v = float(self.config.iou if iou is None else iou)
        imgsz_v = int(self.config.imgsz if imgsz is None else imgsz)
        max_det_v = int(self.config.max_det if max_det is None else max_det)

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_v,
            iou=iou_v,
            imgsz=imgsz_v,
            max_det=max_det_v,
            device=self.config.device,
            retina_masks=True,
            verbose=False,
        )
        infer_s = time.perf_counter() - t0

        boxes: List[Dict[str, Any]] = []
        segments: List[Dict[str, Any]] = []
        if results:
            r0 = results[0]
            names = r0.names or {}
            if r0.boxes is not None and len(r0.boxes):
                xyxy = r0.boxes.xyxy.detach().cpu().tolist()
                cls = r0.boxes.cls.detach().cpu().tolist()
                confs = r0.boxes.conf.detach().cpu().tolist()
                polys: List[Any] = []
                if r0.masks is not None:
                    polys = list(r0.masks.xy) if hasattr(r0.masks, "xy") else []

                for idx, ((x1, y1, x2, y2), c, s) in enumerate(zip(xyxy, cls, confs)):
                    cid = int(c)
                    label = str(names.get(cid, cid))
                    if coord_space == "norm1000":
                        box = {
                            "x1": x1 / w * 1000.0,
                            "y1": y1 / h * 1000.0,
                            "x2": x2 / w * 1000.0,
                            "y2": y2 / h * 1000.0,
                        }
                    else:
                        box = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                    boxes.append(
                        {
                            "label": label,
                            "class_id": cid,
                            "score": float(s),
                            **box,
                        }
                    )

                labels = [str(names.get(int(c), c)) for c in cls]
                class_ids = [int(c) for c in cls]
                score_list = [float(s) for s in confs]
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
                    from app.mask_format import segments_from_bool_masks

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
