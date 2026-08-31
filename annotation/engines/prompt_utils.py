"""Locate / SAM3 共用的类别与短语解析。"""

from __future__ import annotations

from typing import List


def parse_categories(categories: str) -> List[str]:
    raw = str(categories or "").replace("，", ",").replace("；", ",").replace(";", ",")
    parts: List[str] = []
    for chunk in raw.split(","):
        for line in chunk.splitlines():
            item = line.strip()
            if item:
                parts.append(item)
    return parts
