"""数据清洗流水线：可嵌入训练前处理或离线脚本。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from toolkit.data_clean.config import DataCleanConfig, DEFAULT_THRESHOLDS, FILTER_NAMES
from toolkit.data_clean.discovery import (
    collect_image_index,
    collect_label_candidates,
    discover_dirs,
)
from toolkit.data_clean.export import export_dataset
from toolkit.data_clean.filters import apply_cleanvision_filters, validate_labels


@dataclass
class DataCleanResult:
    data_root: Path
    image_dir: Path
    label_dir: Optional[Path]
    output_root: Path
    images_found: int
    labels_found: int
    exported: int
    removed: int
    collision_renamed: int
    manifest: Path
    removal_log: Path
    removal_breakdown: Dict[str, int]

    def to_api_dict(self) -> dict:
        """供 HTTP 返回给 Java / 前端的摘要（明细见 removal_log.csv）。"""
        return {
            "status": "success",
            "summary": {
                "imagesFound": self.images_found,
                "labelsFound": self.labels_found,
                "exported": self.exported,
                "removed": self.removed,
                "collisionRenamed": self.collision_renamed,
                "keepRate": round(self.exported / self.images_found, 4) if self.images_found else 0.0,
            },
            "paths": {
                "dataRoot": str(self.data_root),
                "outputRoot": str(self.output_root),
                "imagesDir": str(self.output_root / "images"),
                "labelsDir": str(self.output_root / "labels"),
                "manifest": str(self.manifest),
                "removalLog": str(self.removal_log),
            },
            "removalBreakdown": self.removal_breakdown,
        }


def _summarize_removals(removal_reasons: Dict[str, List[str]]) -> Dict[str, int]:
    breakdown: Dict[str, int] = defaultdict(int)
    for reasons in removal_reasons.values():
        for reason in reasons:
            key = reason.split(";")[0].strip()
            breakdown[key] += 1
    return dict(sorted(breakdown.items(), key=lambda item: (-item[1], item[0])))


class DataCleaner:
    """可插拔数据清洗器；通过 ``DataCleanConfig`` 注入阈值与目录规则。"""

    def __init__(self, config: DataCleanConfig | None = None) -> None:
        self.config = config or DataCleanConfig()

    def run(self, data_root: str | Path) -> DataCleanResult:
        data_root, image_dir, label_dir, output_root = discover_dirs(
            Path(data_root),
            self.config,
        )
        print(f"DATA_ROOT: {data_root}")
        print(f"image_dir: {image_dir}")
        print(f"label_dir: {label_dir if label_dir else 'not found; will check labels under image_dir'}")
        print(f"output_root: {output_root}")

        image_index = collect_image_index(image_dir, output_root, self.config)
        label_candidates = collect_label_candidates(image_dir, label_dir, output_root, self.config)
        active_keys = set(image_index)
        removal_reasons: Dict[str, List[str]] = defaultdict(list)

        print(f"Images found: {len(image_index)}")
        print(f"Label bases found: {len(label_candidates)}")

        if not self.config.skip_cleanvision:
            apply_cleanvision_filters(
                active_keys,
                removal_reasons,
                image_index,
                image_dir,
                self.config,
            )

        validate_labels(active_keys, removal_reasons, label_candidates, self.config)
        summary = export_dataset(
            active_keys,
            removal_reasons,
            image_index,
            label_candidates,
            data_root,
            output_root,
            self.config,
        )

        return DataCleanResult(
            data_root=data_root,
            image_dir=image_dir,
            label_dir=label_dir,
            output_root=output_root,
            images_found=len(image_index),
            labels_found=len(label_candidates),
            exported=summary["exported"],
            removed=summary["removed"],
            collision_renamed=summary["collision_renamed"],
            manifest=Path(summary["manifest"]),
            removal_log=Path(summary["removal_log"]),
            removal_breakdown=_summarize_removals(removal_reasons),
        )


def run_data_clean(
    data_root: str | Path,
    *,
    config: DataCleanConfig | None = None,
    output_name: str | None = None,
    skip_cleanvision: bool | None = None,
) -> DataCleanResult:
    """嵌入入口：清洗数据集并导出 ``images/`` + ``labels/``。"""
    cfg = config or DataCleanConfig()
    if output_name is not None:
        cfg.output_name = output_name
    if skip_cleanvision is not None:
        cfg.skip_cleanvision = skip_cleanvision
    return DataCleaner(cfg).run(data_root)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "FILTER_NAMES",
    "DataCleanConfig",
    "DataCleanResult",
    "DataCleaner",
    "run_data_clean",
]
