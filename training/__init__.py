"""autolab 闭环检测训练模块。"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "MANAGER",
    "JobManager",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from training.service import MANAGER, JobManager

    value = {"MANAGER": MANAGER, "JobManager": JobManager}[name]
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from training.service import MANAGER, JobManager
