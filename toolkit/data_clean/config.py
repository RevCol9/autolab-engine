"""数据清洗可配置项。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
LABEL_EXTS = frozenset({".xml", ".txt", ".json", ".yaml", ".yml"})
LABEL_EXT_PRIORITY = (".xml", ".txt", ".json", ".yaml", ".yml")

IMAGE_DIR_CANDIDATES = ("images", "image", "imgs", "img")
LABEL_DIR_CANDIDATES = ("labels", "label", "annotations", "annotation", "anns", "ann")

DEFAULT_THRESHOLDS = {
    "dark_brightness_lt": 0.35,
    "odd_aspect_ratio_lt": 0.50,
    "low_information_entropy_lt": 9.0,
    "blurry_blurriness_lt": 44.0,
    "odd_size_lt": 224.0,
}

# 与 CleanVision / 标签校验对应的开关名（前端 / Java 传 enabledFilters）
FILTER_NAMES = (
    "dark",
    "odd_aspect_ratio",
    "low_information",
    "blurry",
    "odd_size",
    "near_duplicates",
    "exact_duplicates",
    "missing_label",
)

DEFAULT_ENABLED_FILTERS = frozenset(FILTER_NAMES)

_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RESERVED_OUTPUT_NAMES = frozenset(
    {
        *IMAGE_DIR_CANDIDATES,
        *LABEL_DIR_CANDIDATES,
        "weights",
        "baseline_eval",
        "trained_eval",
    }
)


def validate_output_name(value: object) -> str:
    """校验清洗输出目录名，避免覆盖数据集与训练产物目录。"""
    text = str(value or "").strip()
    if not text or not _SAFE_OUTPUT_NAME.fullmatch(text):
        raise ValueError(
            "outputName 须为安全的单层目录名（字母或数字开头，仅允许字母数字 . _ -）"
        )
    if text.lower() in _RESERVED_OUTPUT_NAMES:
        raise ValueError(f"outputName 不得使用保留目录名: {text}")
    return text


def validate_thresholds(values: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_THRESHOLDS)
    if values:
        unknown = set(values) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise ValueError(f"未知 thresholds: {sorted(unknown)}")
        merged.update(values)
    for name, raw in merged.items():
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"thresholds.{name} 须为数字") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"thresholds.{name} 须为非负有限数")
        if name in {"dark_brightness_lt", "odd_aspect_ratio_lt"} and value > 1:
            raise ValueError(f"thresholds.{name} 须位于 [0, 1]")
        if name == "odd_size_lt" and value <= 0:
            raise ValueError("thresholds.odd_size_lt 须大于 0")
        merged[name] = value
    return merged


@dataclass
class DataCleanConfig:
    """数据清洗流水线配置；阈值与目录名可按数据集覆盖。"""

    output_name: str = "clean_output"
    overwrite: bool = True
    skip_cleanvision: bool = False
    require_cleanvision: bool = True
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    enabled_filters: frozenset[str] | None = None
    image_dir_candidates: tuple[str, ...] = IMAGE_DIR_CANDIDATES
    label_dir_candidates: tuple[str, ...] = LABEL_DIR_CANDIDATES
    img_exts: frozenset[str] = IMG_EXTS
    label_exts: frozenset[str] = LABEL_EXTS
    label_ext_priority: tuple[str, ...] = LABEL_EXT_PRIORITY

    def __post_init__(self) -> None:
        self.output_name = validate_output_name(self.output_name)
        self.thresholds = validate_thresholds(self.thresholds)
        if self.enabled_filters is not None:
            unknown = set(self.enabled_filters) - set(FILTER_NAMES)
            if unknown:
                raise ValueError(
                    f"未知 enabledFilters: {sorted(unknown)}；可选: {list(FILTER_NAMES)}"
                )
            self.enabled_filters = frozenset(self.enabled_filters)

    def is_filter_enabled(self, name: str) -> bool:
        enabled = self.enabled_filters if self.enabled_filters is not None else DEFAULT_ENABLED_FILTERS
        return name in enabled

    @classmethod
    def from_api_params(
        cls,
        *,
        output_name: str | None = None,
        overwrite: bool | None = None,
        skip_cleanvision: bool | None = None,
        require_cleanvision: bool | None = None,
        thresholds: dict[str, float] | None = None,
        enabled_filters: list[str] | None = None,
    ) -> DataCleanConfig:
        """由 HTTP / Java 传入的可选字段构建配置；未传项保持默认。"""
        return cls(
            output_name=output_name if output_name is not None else "clean_output",
            overwrite=overwrite if overwrite is not None else True,
            skip_cleanvision=skip_cleanvision if skip_cleanvision is not None else False,
            require_cleanvision=require_cleanvision if require_cleanvision is not None else True,
            thresholds=thresholds or {},
            enabled_filters=frozenset(enabled_filters) if enabled_filters is not None else None,
        )
