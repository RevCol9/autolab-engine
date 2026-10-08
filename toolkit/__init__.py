"""
autolab-engine 可插拔工具集。

子模块：
- ``toolkit.model_crypto``：`.niii-model` 容器与受控 YOLO 加载
- ``toolkit.cython_build``：Cython 编译与 develop 部署目录同步

嵌入示例::

    from toolkit.model_crypto import load_yolo_container
    from toolkit.cython_build import CythonBuildConfig, CythonBuilder
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "CythonBuildConfig",
    "CythonBuilder",
    "ModelCryptoError",
    "create_yolo_architecture",
    "load_yolo_container",
]

_LAZY_EXPORTS = {
    "CythonBuildConfig": ("toolkit.cython_build", "CythonBuildConfig"),
    "CythonBuilder": ("toolkit.cython_build", "CythonBuilder"),
    "ModelCryptoError": ("toolkit.model_crypto", "ModelCryptoError"),
    "create_yolo_architecture": ("toolkit.model_crypto", "create_yolo_architecture"),
    "load_yolo_container": ("toolkit.model_crypto", "load_yolo_container"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_EXPORTS[name]
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from toolkit.cython_build import CythonBuildConfig, CythonBuilder
    from toolkit.model_crypto import (
        ModelCryptoError,
        create_yolo_architecture,
        load_yolo_container,
    )
