"""数据清洗可配置项。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass
class DataCleanConfig:
    """数据清洗流水线配置；阈值与目录名可按数据集覆盖。"""

    output_name: str = "clean_output"
    overwrite: bool = True
    skip_cleanvision: bool = False
    require_cleanvision: bool = False
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    enabled_filters: frozenset[str] | None = None
    image_dir_candidates: tuple[str, ...] = IMAGE_DIR_CANDIDATES
    label_dir_candidates: tuple[str, ...] = LABEL_DIR_CANDIDATES
    img_exts: frozenset[str] = IMG_EXTS
    label_exts: frozenset[str] = LABEL_EXTS
    label_ext_priority: tuple[str, ...] = LABEL_EXT_PRIORITY

    @classmethod
    def autolab_storage(cls, output_name: str = "clean_output") -> DataCleanConfig:
        """autolab storage 常见布局：train{N}/images + train{N}/labels。"""
        return cls(output_name=output_name)

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
        cfg = cls()
        if output_name is not None:
            cfg.output_name = output_name
        if overwrite is not None:
            cfg.overwrite = overwrite
        if skip_cleanvision is not None:
            cfg.skip_cleanvision = skip_cleanvision
        if require_cleanvision is not None:
            cfg.require_cleanvision = require_cleanvision
        if thresholds:
            merged = dict(DEFAULT_THRESHOLDS)
            merged.update(thresholds)
            cfg.thresholds = merged
        if enabled_filters is not None:
            unknown = set(enabled_filters) - set(FILTER_NAMES)
            if unknown:
                raise ValueError(f"未知 enabledFilters: {sorted(unknown)}；可选: {list(FILTER_NAMES)}")
            cfg.enabled_filters = frozenset(enabled_filters)
        return cfg
