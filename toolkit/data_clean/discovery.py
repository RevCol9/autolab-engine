"""数据集目录发现与图像/标签索引。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from toolkit.data_clean.config import DataCleanConfig
from toolkit.data_clean.paths import image_base_key, is_under, normalize_rel

LabelCandidate = Tuple[int, int, str, Path]


def has_files_with_ext(
    root: Path,
    exts: set[str],
    output_root: Optional[Path] = None,
) -> bool:
    for path in root.rglob("*"):
        if output_root and is_under(path, output_root):
            continue
        if path.is_file() and path.suffix.lower() in exts:
            return True
    return False


def first_existing_dir(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate.resolve()
    return None


def discover_dirs(
    data_root: Path,
    config: DataCleanConfig,
) -> Tuple[Path, Path, Optional[Path], Path]:
    root = data_root.expanduser().resolve()
    output_root = (root / config.output_name).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT does not exist or is not a directory: {root}")

    image_dir = first_existing_dir(root, config.image_dir_candidates)
    if image_dir is None:
        if has_files_with_ext(root, set(config.img_exts), output_root):
            image_dir = root
        else:
            checked = ", ".join(str(root / name) for name in config.image_dir_candidates)
            raise FileNotFoundError(
                "No image directory found. Checked: "
                f"{checked}; DATA_ROOT itself also contains no supported images."
            )

    label_dir = first_existing_dir(root, config.label_dir_candidates)
    return root, image_dir, label_dir, output_root


def collect_image_index(
    image_dir: Path,
    output_root: Path,
    config: DataCleanConfig,
) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if is_under(path, output_root):
            continue
        if path.is_file() and path.suffix.lower() in config.img_exts:
            key = normalize_rel(path.relative_to(image_dir))
            if key in index:
                raise ValueError(f"Duplicate image relative path: {key}")
            index[key] = path
    if not index:
        raise ValueError(f"No supported images found under: {image_dir}")
    return index


def collect_label_candidates(
    image_dir: Path,
    label_dir: Optional[Path],
    output_root: Path,
    config: DataCleanConfig,
) -> Dict[str, List[LabelCandidate]]:
    candidates: Dict[str, List[LabelCandidate]] = defaultdict(list)
    roots: List[Tuple[int, Path, str]] = []
    if label_dir and label_dir.is_dir():
        roots.append((0, label_dir, "label_dir"))
    roots.append((1, image_dir, "image_dir"))

    for root_priority, root, origin in roots:
        for path in sorted(root.rglob("*")):
            if is_under(path, output_root):
                continue
            suffix = path.suffix.lower()
            if path.is_file() and suffix in config.label_exts:
                base_key = normalize_rel(path.relative_to(root).with_suffix(""))
                ext_priority = (
                    config.label_ext_priority.index(suffix)
                    if suffix in config.label_ext_priority
                    else len(config.label_ext_priority)
                )
                candidates[base_key].append((root_priority, ext_priority, origin, path))

    for key in candidates:
        candidates[key].sort(key=lambda item: (item[0], item[1], str(item[3]).lower()))
    return candidates


def resolve_image_key(
    name: object,
    image_index: Dict[str, Path],
    image_dir: Path,
    config: DataCleanConfig,
) -> Optional[str]:
    raw = str(name)
    candidates: List[str] = []

    try:
        path = Path(raw)
        if path.is_absolute():
            try:
                candidates.append(normalize_rel(path.resolve().relative_to(image_dir.resolve())))
            except ValueError:
                pass
    except Exception:
        pass

    text = raw.replace("\\", "/")
    for dirname in config.image_dir_candidates:
        marker = f"/{dirname}/"
        if marker in text:
            candidates.append(text.split(marker, 1)[1])
        prefix = f"{dirname}/"
        if text.startswith(prefix):
            candidates.append(text[len(prefix) :])
    candidates.append(text)

    for candidate in candidates:
        try:
            key = normalize_rel(candidate)
        except ValueError:
            continue
        if key in image_index:
            return key
    return None
