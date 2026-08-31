"""质量过滤：CleanVision + 标签校验。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from toolkit.data_clean.config import DataCleanConfig
from toolkit.data_clean.discovery import LabelCandidate, resolve_image_key


def add_reason(removal_reasons: Dict[str, List[str]], key: str, reason: str) -> None:
    if reason not in removal_reasons[key]:
        removal_reasons[key].append(reason)


def remove_keys(
    active_keys: Set[str],
    removal_reasons: Dict[str, List[str]],
    keys: Iterable[str],
    reason: str,
) -> int:
    removed = 0
    for key in sorted(set(keys)):
        if key in active_keys:
            active_keys.remove(key)
            add_reason(removal_reasons, key, reason)
            removed += 1
    print(f"{reason}: removed={removed}, saved_left={len(active_keys)}")
    return removed


def issue_index(imagelab, issue_name: str, column: str, threshold: float, op: str = "lt") -> List[object]:
    frame = imagelab.info.get(issue_name)
    if frame is None:
        print(f"Skip {issue_name}: CleanVision did not return this issue type")
        return []
    if column not in frame.columns:
        print(f"Skip {issue_name}: missing column {column!r}; columns={list(frame.columns)}")
        return []
    if op == "lt":
        return list(frame[frame[column] < threshold].index)
    if op == "gt":
        return list(frame[frame[column] > threshold].index)
    raise ValueError(f"Unsupported operator: {op}")


def remove_from_cleanvision_index(
    active_keys: Set[str],
    removal_reasons: Dict[str, List[str]],
    image_index: Dict[str, Path],
    image_dir: Path,
    index_obj: Iterable[object],
    reason: str,
    config: DataCleanConfig,
) -> int:
    keys = []
    unmatched = []
    for name in index_obj:
        key = resolve_image_key(name, image_index, image_dir, config)
        if key is None:
            unmatched.append(str(name))
        else:
            keys.append(key)
    removed = remove_keys(active_keys, removal_reasons, keys, reason)
    if unmatched:
        print(f"{reason}: unmatched={len(unmatched)}, examples={unmatched[:5]}")
    return removed


def get_duplicate_sets(imagelab, issue_name: str) -> Sequence[Sequence[object]]:
    issue = imagelab.info.get(issue_name)
    if not issue:
        return []
    if isinstance(issue, dict):
        return issue.get("sets", []) or []
    try:
        return issue["sets"] or []
    except Exception:
        return []


def remove_duplicate_sets(
    active_keys: Set[str],
    removal_reasons: Dict[str, List[str]],
    image_index: Dict[str, Path],
    image_dir: Path,
    sets_obj: Sequence[Sequence[object]],
    reason: str,
    config: DataCleanConfig,
) -> int:
    removed = 0
    unmatched = 0
    for group in sets_obj:
        keys = []
        for name in group:
            key = resolve_image_key(name, image_index, image_dir, config)
            if key is None:
                unmatched += 1
            elif key in active_keys:
                keys.append(key)
        keys = sorted(set(keys))
        if len(keys) <= 1:
            continue
        keep = keys[0]
        for key in keys[1:]:
            active_keys.remove(key)
            add_reason(removal_reasons, key, f"{reason}; kept={keep}")
            removed += 1
    print(f"{reason}: removed={removed}, unmatched={unmatched}, saved_left={len(active_keys)}")
    return removed


def apply_cleanvision_filters(
    active_keys: Set[str],
    removal_reasons: Dict[str, List[str]],
    image_index: Dict[str, Path],
    image_dir: Path,
    config: DataCleanConfig,
) -> None:
    try:
        from cleanvision import Imagelab
    except ImportError:
        message = "cleanvision is not installed; skip image-quality filters."
        if config.require_cleanvision:
            raise RuntimeError(message)
        print(message)
        return

    thresholds = config.thresholds
    imagelab = Imagelab(data_path=str(image_dir))
    imagelab.find_issues()
    print("CleanVision issue keys:", list(imagelab.info.keys()))
    try:
        print(imagelab.issue_summary)
    except Exception:
        pass

    checks = [
        ("dark", "dark", "brightness", thresholds["dark_brightness_lt"], "dark:brightness<"),
        (
            "odd_aspect_ratio",
            "odd_aspect_ratio",
            "aspect_ratio",
            thresholds["odd_aspect_ratio_lt"],
            "odd_aspect_ratio:aspect_ratio<",
        ),
        (
            "low_information",
            "low_information",
            "entropy",
            thresholds["low_information_entropy_lt"],
            "low_information:entropy<",
        ),
        ("blurry", "blurry", "blurriness", thresholds["blurry_blurriness_lt"], "blurry:blurriness<"),
        ("odd_size", "odd_size", "size", thresholds["odd_size_lt"], "odd_size:size<"),
    ]
    for filter_name, issue_name, column, threshold, reason_prefix in checks:
        if not config.is_filter_enabled(filter_name):
            continue
        idx = issue_index(imagelab, issue_name, column, threshold, "lt")
        remove_from_cleanvision_index(
            active_keys,
            removal_reasons,
            image_index,
            image_dir,
            idx,
            f"{reason_prefix}{threshold}",
            config,
        )

    if config.is_filter_enabled("near_duplicates"):
        remove_duplicate_sets(
            active_keys,
            removal_reasons,
            image_index,
            image_dir,
            get_duplicate_sets(imagelab, "near_duplicates"),
            "near_duplicates",
            config,
        )
    if config.is_filter_enabled("exact_duplicates"):
        remove_duplicate_sets(
            active_keys,
            removal_reasons,
            image_index,
            image_dir,
            get_duplicate_sets(imagelab, "exact_duplicates"),
            "exact_duplicates",
            config,
        )


def validate_labels(
    active_keys: Set[str],
    removal_reasons: Dict[str, List[str]],
    label_candidates: Dict[str, List[LabelCandidate]],
    config: DataCleanConfig,
) -> None:
    if not config.is_filter_enabled("missing_label"):
        return
    from toolkit.data_clean.paths import image_base_key

    missing = []
    ambiguous = []
    for key in sorted(active_keys):
        base = image_base_key(key)
        if base not in label_candidates:
            missing.append(key)
        elif len(label_candidates[base]) > 1:
            ambiguous.append(key)
    remove_keys(active_keys, removal_reasons, missing, "missing_label")
    if ambiguous:
        print(f"Multiple label candidates: {len(ambiguous)}, examples={ambiguous[:5]}")
