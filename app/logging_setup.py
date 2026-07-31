"""统一日志格式。"""

from __future__ import annotations

import logging


class _ShortNameFilter(logging.Filter):
    _MAP = {
        "autolab-engine": "app",
        "app.engines.yolo": "yolo",
        "uvicorn": "http",
        "uvicorn.error": "http",
        "uvicorn.access": "http",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        record.short_name = self._MAP.get(record.name, record.name.split(".")[-1][:12])
        return True


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setLevel(root.level)
    handler.addFilter(_ShortNameFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(short_name)s] %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
