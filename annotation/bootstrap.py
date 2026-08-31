"""应用启动：加载 config.yaml 并应用运行时环境变量。"""

from __future__ import annotations

from annotation.settings import Settings, apply_runtime_env, load_settings

SETTINGS: Settings = load_settings()
apply_runtime_env(SETTINGS)
