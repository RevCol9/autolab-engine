"""LocateAnything 推理 Worker。

进程内常驻一份模型：启动时加载，请求间复用，避免反复占满显存。
坐标约定：模型输出为 0~1000 归一化坐标；本模块解析为原图像素坐标。
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)

# 模型文本输出中的框 / 点 / 指代标签
_BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_PATTERN = re.compile(r"<box><(\d+)><(\d+)></box>")
_REF_PATTERN = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)


class _DropNoisyTokenizerLogs(logging.Filter):
    """过滤 tokenizer 加载时刷屏的 AddedToken 明细。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "AddedToken" in msg:
            return False
        if "added_tokens_decoder" in msg:
            return False
        return True


@contextmanager
def _quiet_hf_loading_logs() -> Iterator[None]:
    """加载期降低 transformers 噪声，并过滤 AddedToken 大段输出。"""
    noisy_loggers = (
        "transformers",
        "transformers.tokenization_utils_base",
        "transformers.tokenization_utils",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "huggingface_hub",
    )
    prev_levels: Dict[str, int] = {}
    filters: List[tuple[logging.Logger, logging.Filter]] = []
    drop = _DropNoisyTokenizerLogs()

    root = logging.getLogger()
    root.addFilter(drop)
    filters.append((root, drop))

    for name in noisy_loggers:
        lg = logging.getLogger(name)
        prev_levels[name] = lg.level
        lg.setLevel(logging.ERROR)
        lg.addFilter(drop)
        filters.append((lg, drop))

    try:
        yield
    finally:
        for name, level in prev_levels.items():
            logging.getLogger(name).setLevel(level)
        for lg, flt in filters:
            lg.removeFilter(flt)


class LocateAnythingWorker:
    """有状态 Worker：加载一次 LocateAnything，供全部请求复用。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "float16",
        local_files_only: bool = True,
        device_map: Optional[str] = None,
        max_memory: Optional[Dict[Union[int, str], str]] = None,
        in_token_limit: int = 4096,
        max_image_side: int = 1280,
    ):
        self.model_path = model_path
        self.device = device
        self.local_files_only = local_files_only
        self.dtype = self._resolve_dtype(dtype)
        self.device_map = (device_map or "").strip() or None
        self.max_memory = max_memory
        self.in_token_limit = int(in_token_limit)
        # 0 表示不做「最长边」预缩放，仅依赖 image_processor.in_token_limit
        self.max_image_side = int(max_image_side) if max_image_side else 0
        self._uses_device_map = False

        logger.debug(
            "load  path=%s | device=%s | dtype=%s | token_limit=%s | side=%s",
            model_path,
            device,
            dtype,
            self.in_token_limit,
            self.max_image_side,
        )
        if device.startswith("cuda") or self.device_map:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"请求使用 CUDA，但 torch.cuda.is_available()=False。"
                    f" torch={torch.__version__}, compiled_cuda={torch.version.cuda}"
                )
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                free_b, total_b = torch.cuda.mem_get_info(idx)
                logger.debug(
                    "gpu   [%s] %s | free %.1f / %.1f GiB",
                    idx,
                    props.name,
                    free_b / (1024**3),
                    total_b / (1024**3),
                )

        t0 = time.perf_counter()
        load_steps = [
            ("tokenizer", self._load_tokenizer),
            ("processor", self._load_processor),
            ("model", self._load_model),
        ]

        with _quiet_hf_loading_logs():
            with tqdm(
                total=len(load_steps),
                desc="加载 LocateAnything",
                unit="step",
                leave=True,
                dynamic_ncols=True,
            ) as pbar:
                for step_name, loader in load_steps:
                    pbar.set_postfix_str(step_name)
                    t_step = time.perf_counter()
                    loader(model_path, local_files_only, device)
                    elapsed = time.perf_counter() - t_step
                    pbar.set_postfix_str(f"{step_name} {elapsed:.1f}s")
                    pbar.update(1)

        self._sync()
        self.load_seconds = time.perf_counter() - t0
        self._log_cuda_mem("load")
        logger.debug("load  done in %.1fs", self.load_seconds)

    def _load_tokenizer(self, model_path: str, local_files_only: bool, device: str) -> None:
        # tokenizer 的 __repr__/内部打印会刷出上千行 AddedToken，加载时吞掉 stdout/stderr 噪音
        import io
        from contextlib import redirect_stderr, redirect_stdout

        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=local_files_only
            )
        n_added = len(getattr(self.tokenizer, "added_tokens_decoder", {}) or {})
        logger.debug(
            "tokenizer vocab=%s added≈%s",
            getattr(self.tokenizer, "vocab_size", "?"),
            n_added,
        )

    def _load_processor(self, model_path: str, local_files_only: bool, device: str) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only
        )
        if hasattr(self.processor, "image_processor") and hasattr(
            self.processor.image_processor, "in_token_limit"
        ):
            old_limit = self.processor.image_processor.in_token_limit
            self.processor.image_processor.in_token_limit = self.in_token_limit
            logger.debug(
                "in_token_limit %s -> %s",
                old_limit,
                self.in_token_limit,
            )

    def _load_model(self, model_path: str, local_files_only: bool, device: str) -> None:
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        # device_map：多卡切分「权重」；不能降低 ViT 单卡激活峰值。
        # 高分辨率 OOM 请优先下调 in_token_limit / max_image_side。
        if self.device_map:
            if torch.cuda.is_available():
                # transformers 估算 max_memory 前需先完成 CUDA 初始化，否则会报：
                # RuntimeError: Invalid device argument : did you call init?
                torch.cuda.init()
                for idx in range(torch.cuda.device_count()):
                    with torch.cuda.device(idx):
                        _ = torch.empty(1, device=f"cuda:{idx}")
            load_kwargs["device_map"] = self.device_map
            if self.max_memory:
                load_kwargs["max_memory"] = self.max_memory
            self.model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
            self._uses_device_map = True
            logger.debug(
                "device_map=%s | hf_device_map=%s",
                self.device_map,
                getattr(self.model, "hf_device_map", None),
            )
        else:
            self.model = AutoModel.from_pretrained(model_path, **load_kwargs).to(device).eval()
            self._uses_device_map = False

    def _input_device(self) -> torch.device:
        """选择输入张量应放置的设备（device_map 时优先跟视觉塔）。"""
        if self._uses_device_map:
            try:
                return next(self.model.vision_model.parameters()).device
            except Exception:
                return next(self.model.parameters()).device
        return torch.device(self.device)

    def _log_cuda_mem(self, tag: str, *, force: bool = False) -> None:
        """打印各可见 GPU 显存；默认 DEBUG，OOM 等场景可 force 到 WARNING。"""
        if not torch.cuda.is_available():
            return
        parts = []
        for idx in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            parts.append(
                f"GPU{idx} alloc={allocated:.1f} res={reserved:.1f} "
                f"free={free_b / (1024**3):.1f}/{total_b / (1024**3):.1f}GiB"
            )
        msg = "vram  [%s] | %s" % (tag, " | ".join(parts))
        if force:
            logger.warning(msg)
        else:
            logger.debug(msg)

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        dtype = (dtype or "float16").lower()
        if dtype in ("fp16", "float16", "half"):
            return torch.float16
        if dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"不支持的 dtype: {dtype}")

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _maybe_downscale(self, image: Image.Image) -> Image.Image:
        """按最长边限制做预缩放，降低进入 ViT 前的分辨率。"""
        if not self.max_image_side or self.max_image_side <= 0:
            return image
        w, h = image.size
        side = max(w, h)
        if side <= self.max_image_side:
            return image
        scale = self.max_image_side / float(side)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        logger.debug(
            "resize %sx%s -> %sx%s",
            w,
            h,
            new_size[0],
            new_size[1],
        )
        return image.resize(new_size, Image.Resampling.BICUBIC)

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """通用生成入口：构造多模态消息 -> 预处理 -> generate -> 返回原文与耗时。"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self._maybe_downscale(image)

        timings: Dict[str, float] = {}
        t_total = time.perf_counter()
        logger.debug(
            "infer %sx%s | mode=%s tokens=%s | %s",
            image.width,
            image.height,
            generation_mode,
            max_new_tokens,
            question[:80].replace("\n", " "),
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        t0 = time.perf_counter()
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        input_device = self._input_device()
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        )
        inputs = {k: (v.to(input_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        self._sync()
        timings["preprocess"] = time.perf_counter() - t0

        gen_kwargs: Dict[str, Any] = {
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_hws": inputs.get("image_grid_hws", None),
            "tokenizer": self.tokenizer,
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
            "generation_mode": generation_mode,
            "temperature": float(temperature),
            "verbose": bool(verbose),
            "repetition_penalty": 1.1,
        }

        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        t0 = time.perf_counter()
        try:
            response = self.model.generate(**gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            self._log_cuda_mem("OOM", force=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        self._sync()
        timings["generate"] = time.perf_counter() - t0
        self._log_cuda_mem("infer")

        answer = response[0] if isinstance(response, tuple) else response
        if not isinstance(answer, str):
            answer = str(answer)

        result: Dict[str, Any] = {
            "answer": answer,
            "prompt": question,
            "timings": timings,
            "load_seconds": self.load_seconds,
            "in_token_limit": self.in_token_limit,
            "max_image_side": self.max_image_side,
        }
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]

        timings["total_request"] = time.perf_counter() - t_total
        logger.debug(
            "infer done | prep=%.2fs gen=%.2fs total=%.2fs | ans=%s",
            timings["preprocess"],
            timings["generate"],
            timings["total_request"],
            len(answer),
        )
        return result

    def detect(self, image: Image.Image, categories: List[str], **kwargs: Any) -> Dict[str, Any]:
        """开放词汇检测：类别列表，模型内部用 </c> 拼接。"""
        cats = "</c>".join([c.strip() for c in categories if c.strip()])
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """短语单实例定位。"""
        return self.predict(
            image,
            f"Locate a single instance that matches the following description: {phrase}.",
            **kwargs,
        )

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """短语多实例定位。"""
        return self.predict(
            image,
            f"Locate all the instances that match the following description: {phrase}.",
            **kwargs,
        )

    def ground_text(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """文本定位（按指代短语找文字区域）。"""
        return self.predict(image, f"Please locate the text referred as {phrase}.", **kwargs)

    def detect_text(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        """全图文本检测。"""
        return self.predict(image, "Detect all the text in box format.", **kwargs)

    def ground_gui(
        self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs: Any
    ) -> Dict[str, Any]:
        """GUI 元素定位；output_type=point 时改为指点。"""
        prompt = (
            f"Point to: {phrase}."
            if output_type == "point"
            else f"Locate the region that matches the following description: {phrase}."
        )
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """点定位。"""
        return self.predict(image, f"Point to: {phrase}.", **kwargs)

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """从模型原文解析 bbox，并映射到原图像素坐标。"""
        refs: List[Tuple[int, str]] = [
            (m.start(), m.group(1).strip()) for m in _REF_PATTERN.finditer(answer)
        ]
        boxes: List[Dict[str, Any]] = []
        for m in _BOX_PATTERN.finditer(answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            preceding_refs = [r for r in refs if r[0] < m.start()]
            label = preceding_refs[-1][1] if preceding_refs else ""
            boxes.append(
                {
                    "label": label,
                    "score": None,
                    "x1": x1 / 1000.0 * image_width,
                    "y1": y1 / 1000.0 * image_height,
                    "x2": x2 / 1000.0 * image_width,
                    "y2": y2 / 1000.0 * image_height,
                    "raw_1000": [x1, y1, x2, y2],
                }
            )
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """从模型原文解析点坐标，并映射到原图像素坐标。"""
        refs: List[Tuple[int, str]] = [
            (m.start(), m.group(1).strip()) for m in _REF_PATTERN.finditer(answer)
        ]
        points: List[Dict[str, Any]] = []
        for m in _POINT_PATTERN.finditer(answer):
            x, y = int(m.group(1)), int(m.group(2))
            preceding_refs = [r for r in refs if r[0] < m.start()]
            label = preceding_refs[-1][1] if preceding_refs else ""
            points.append(
                {
                    "label": label,
                    "x": x / 1000.0 * image_width,
                    "y": y / 1000.0 * image_height,
                    "raw_1000": [x, y],
                }
            )
        return points
