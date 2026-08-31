"""
autolab-engine 可插拔工具集。

子模块：
- ``toolkit.model_crypto``：YOLO 权重 AES 加密与内存加载
- ``toolkit.cython_build``：Cython 编译与 develop 部署目录同步

嵌入示例::

    from toolkit.model_crypto import load_yolo
    from toolkit.cython_build import CythonBuildConfig, CythonBuilder
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "CythonBuildConfig",
    "CythonBuilder",
    "ModelCryptoConfig",
    "ModelDecryptor",
    "encrypt_all_weights",
    "encrypt_weights",
    "is_encrypted_checkpoint",
    "load_yolo",
    "resolve_weight_path",
]

_LAZY_EXPORTS = {
    "CythonBuildConfig": ("toolkit.cython_build", "CythonBuildConfig"),
    "CythonBuilder": ("toolkit.cython_build", "CythonBuilder"),
    "ModelCryptoConfig": ("toolkit.model_crypto", "ModelCryptoConfig"),
    "ModelDecryptor": ("toolkit.model_crypto", "ModelDecryptor"),
    "encrypt_all_weights": ("toolkit.model_crypto", "encrypt_all_weights"),
    "encrypt_weights": ("toolkit.model_crypto", "encrypt_weights"),
    "is_encrypted_checkpoint": ("toolkit.model_crypto", "is_encrypted_checkpoint"),
    "load_yolo": ("toolkit.model_crypto", "load_yolo"),
    "resolve_weight_path": ("toolkit.model_crypto", "resolve_weight_path"),
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
        ModelCryptoConfig,
        ModelDecryptor,
        encrypt_all_weights,
        encrypt_weights,
        is_encrypted_checkpoint,
        load_yolo,
        resolve_weight_path,
    )
