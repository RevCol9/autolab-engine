#!/usr/bin/env python3
"""启动 autolab-engine 推理服务。"""

from app.settings import apply_runtime_env, load_settings

_settings = load_settings()
apply_runtime_env(_settings)

from app.main import SETTINGS, app  # noqa: E402
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=SETTINGS.host,
        port=int(SETTINGS.port),
        workers=1,
        log_level=SETTINGS.log_level.lower(),
    )
