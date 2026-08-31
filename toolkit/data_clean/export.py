"""清洗结果导出为 images/ + labels/ 标准布局。"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence, Set, Tuple

from toolkit.data_clean.config import DataCleanConfig
from toolkit.data_clean.discovery import LabelCandidate
from toolkit.data_clean.paths import append_suffix, image_base_key, normalize_rel


def safe_reset_output_dir(output_root: Path, data_root: Path, overwrite: bool) -> Tuple[Path, Path]:
    output_root = output_root.resolve()
    data_root = data_root.resolve()
    if output_root == data_root or data_root not in output_root.parents:
        raise ValueError(f"Refuse to reset non-DATA_ROOT child path: {output_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    image_out = output_root / "images"
    label_out = output_root / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    return image_out, label_out


def unique_pair_rel(
    image_key: str,
    label_suffix: str,
    used_images: Set[str],
    used_labels: Set[str],
) -> Tuple[str, str, bool]:
    image_path = PurePosixPath(image_key)
    base_key = image_base_key(image_key)
    base = PurePosixPath(base_key)
    image_rel = normalize_rel(image_path)
    label_rel = append_suffix(base_key, label_suffix)

    if image_rel not in used_images and label_rel not in used_labels:
        return image_rel, label_rel, False

    digest = hashlib.sha1(image_key.encode("utf-8")).hexdigest()[:10]
    image_name = f"{image_path.stem}__{digest}{image_path.suffix.lower()}"
    label_name = f"{base.name}__{digest}{label_suffix.lower()}"
    image_rel = normalize_rel(str(image_path.with_name(image_name)))
    label_rel = normalize_rel(str(base.with_name(label_name)))

    counter = 1
    while image_rel in used_images or label_rel in used_labels:
        image_name = f"{image_path.stem}__{digest}_{counter}{image_path.suffix.lower()}"
        label_name = f"{base.name}__{digest}_{counter}{label_suffix.lower()}"
        image_rel = normalize_rel(str(image_path.with_name(image_name)))
        label_rel = normalize_rel(str(base.with_name(label_name)))
        counter += 1

    return image_rel, label_rel, True


def copy_to_relative(src: Path, dst_root: Path, rel_key: str) -> Path:
    dst = dst_root / Path(rel_key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copy2(src, dst)
    return dst


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_dataset(
    active_keys: Set[str],
    removal_reasons: Dict[str, List[str]],
    image_index: Dict[str, Path],
    label_candidates: Dict[str, List[LabelCandidate]],
    data_root: Path,
    output_root: Path,
    config: DataCleanConfig,
) -> dict:
    if not active_keys:
        raise ValueError("No images left to export. Check whether thresholds are too aggressive.")

    image_out, label_out = safe_reset_output_dir(output_root, data_root, config.overwrite)
    used_images: Set[str] = set()
    used_labels: Set[str] = set()
    manifest_rows = []
    collision_renamed = 0

    for image_key in sorted(active_keys):
        image_src = image_index[image_key]
        base = image_base_key(image_key)
        label_items = label_candidates.get(base)
        if not label_items:
            raise RuntimeError(f"Internal error: kept image has no label: {image_key}")

        label_src = label_items[0][3]
        image_rel, label_rel, renamed = unique_pair_rel(
            image_key,
            label_src.suffix,
            used_images,
            used_labels,
        )
        if renamed:
            collision_renamed += 1

        image_dst = copy_to_relative(image_src, image_out, image_rel)
        label_dst = copy_to_relative(label_src, label_out, label_rel)
        used_images.add(image_rel)
        used_labels.add(label_rel)
        manifest_rows.append(
            {
                "source_image_rel_path": image_key,
                "output_image_rel_path": image_rel,
                "output_label_rel_path": label_rel,
                "image_source": str(image_src),
                "label_source": str(label_src),
                "image_output": str(image_dst),
                "label_output": str(label_dst),
                "renamed_for_collision": str(renamed),
            }
        )

    removal_rows = [
        {"image_rel_path": key, "reasons": "; ".join(reasons)}
        for key, reasons in sorted(removal_reasons.items())
    ]
    write_csv(
        output_root / "kept_manifest.csv",
        [
            "source_image_rel_path",
            "output_image_rel_path",
            "output_label_rel_path",
            "image_source",
            "label_source",
            "image_output",
            "label_output",
            "renamed_for_collision",
        ],
        manifest_rows,
    )
    write_csv(output_root / "removal_log.csv", ["image_rel_path", "reasons"], removal_rows)

    exported_images = [
        p for p in image_out.rglob("*") if p.is_file() and p.suffix.lower() in config.img_exts
    ]
    exported_labels = [
        p for p in label_out.rglob("*") if p.is_file() and p.suffix.lower() in config.label_exts
    ]
    if len(exported_images) != len(manifest_rows):
        raise RuntimeError("Exported image count does not match manifest count")
    if len(exported_labels) != len(manifest_rows):
        raise RuntimeError("Exported label count does not match manifest count")

    summary = {
        "exported": len(manifest_rows),
        "removed": len(removal_rows),
        "collision_renamed": collision_renamed,
        "output_root": str(output_root),
        "manifest": str(output_root / "kept_manifest.csv"),
        "removal_log": str(output_root / "removal_log.csv"),
    }

    print(f"Exported images: {summary['exported']}")
    print(f"Removed images: {summary['removed']}")
    print(f"Renamed because of output collisions: {collision_renamed}")
    print(f"Output directory: {output_root}")
    print(f"Manifest: {output_root / 'kept_manifest.csv'}")
    print(f"Removal log: {output_root / 'removal_log.csv'}")
    return summary
