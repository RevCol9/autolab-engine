#!/usr/bin/env python3
"""启动训练 HTTP 服务（默认 :21011，与推理 :21010 分离）。"""

from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("AUTOLAB_TRAIN_HOST", "0.0.0.0")
    port = int(os.environ.get("AUTOLAB_TRAIN_PORT", "21011"))
    uvicorn.run(
        "training.api:app",
        host=host,
        port=port,
        workers=1,
        log_level=os.environ.get("AUTOLAB_TRAIN_LOG_LEVEL", "info"),
    )
