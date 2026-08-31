"""路径解析与相对路径工具。"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath


def parse_cli_path(value: str) -> Path:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]

    if len(text) >= 3 and text[0] in ("r", "R") and text[1] in ("'", '"') and text[-1] == text[1]:
        text = text[2:-1]
    elif re.match(r"^[rR][A-Za-z]:[\\/]", text):
        text = text[1:]

    return Path(text)


def normalize_rel(value: object) -> str:
    text = str(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Invalid relative path: {value!r}")
    return "/".join(parts)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def image_base_key(image_key: str) -> str:
    return normalize_rel(str(PurePosixPath(image_key).with_suffix("")))


def append_suffix(rel_stem: str, suffix: str) -> str:
    return normalize_rel(f"{rel_stem}{suffix.lower()}")
