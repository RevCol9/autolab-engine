"""推理 ↔ 训练 Swagger /docs 互链。

链接出现在各服务 OpenAPI 页顶 description 中。默认使用本机对外 IP；
仍可通过 ``AUTOLAB_DOCS_PUBLIC_HOST`` 或完整 ``AUTOLAB_*_DOCS_URL`` 覆盖。
"""

from __future__ import annotations

import os
from typing import Literal

# 后端从其它机器打开 /docs 时互链使用的主机名
DEFAULT_DOCS_PUBLIC_HOST = "10.65.48.104"


def _docs_public_host() -> str:
    for key in ("AUTOLAB_DOCS_PUBLIC_HOST", "AUTOLAB_PUBLIC_HOST"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return DEFAULT_DOCS_PUBLIC_HOST


def _normalize_docs_url(url: str) -> str:
    text = url.strip().rstrip("/")
    if text.endswith("/docs"):
        return text
    return f"{text}/docs"


def _infer_port() -> int:
    try:
        from annotation.bootstrap import SETTINGS

        return int(SETTINGS.port)
    except Exception:
        return 21010


def _train_port() -> int:
    try:
        from training.settings import training_server_port

        return int(training_server_port())
    except Exception:
        return 21011


def infer_docs_url() -> str:
    explicit = os.environ.get("AUTOLAB_INFER_DOCS_URL", "").strip()
    if explicit:
        return _normalize_docs_url(explicit)
    host = _docs_public_host()
    return f"http://{host}:{_infer_port()}/docs"


def train_docs_url() -> str:
    explicit = os.environ.get("AUTOLAB_TRAIN_DOCS_URL", "").strip()
    if explicit:
        return _normalize_docs_url(explicit)
    host = _docs_public_host()
    return f"http://{host}:{_train_port()}/docs"


def openapi_description(
    service: Literal["infer", "train"],
    *,
    summary: str,
) -> str:
    """生成 FastAPI ``description``，在页顶插入对端 /docs 链接。"""
    if service == "infer":
        peer_label = "训练服务（YOLO 闭环训练 start/stop）"
        peer_url = train_docs_url()
        self_port = _infer_port()
    else:
        peer_label = "推理 / 半自动标注服务（predict、模型 load/unload）"
        peer_url = infer_docs_url()
        self_port = _train_port()

    return (
        f"{summary}（本服务默认端口 **{self_port}**）。\n\n"
        f"**关联 OpenAPI**：[{peer_label}]({peer_url})"
    )
