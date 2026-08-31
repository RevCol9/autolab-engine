#!/usr/bin/env python3
"""启动 autolab-engine 推理服务。"""

from app.bootstrap import SETTINGS
from app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=SETTINGS.host,
        port=int(SETTINGS.port),
        workers=1,
        log_level=SETTINGS.log_level.lower(),
    )
