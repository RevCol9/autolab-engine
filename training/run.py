#!/usr/bin/env python3
"""启动训练 HTTP 服务（默认 :21011，与推理 :21010 分离）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn

from training.settings import (
    apply_training_runtime_env,
    training_log_level,
    training_server_host,
    training_server_port,
)

apply_training_runtime_env()

if __name__ == "__main__":
    host = os.environ.get("AUTOLAB_TRAIN_HOST") or training_server_host()
    port = int(os.environ.get("AUTOLAB_TRAIN_PORT") or training_server_port())
    log_level = os.environ.get("AUTOLAB_TRAIN_LOG_LEVEL") or training_log_level()
    uvicorn.run(
        "api.train:app",
        host=host,
        port=port,
        workers=1,
        log_level=log_level,
    )
