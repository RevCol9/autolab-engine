"""训练前数据清洗：解析 storage 路径并调用 toolkit。"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from toolkit.data_clean import DataCleanConfig, run_data_clean
from training.paths import train_save_dir


@dataclass
class _LockEntry:
    lock: threading.Lock
    users: int = 0


_LOCK_POOL_GUARD = threading.Lock()
_LOCK_POOL: dict[Path, _LockEntry] = {}


@contextmanager
def _clean_output_lock(path: Path) -> Iterator[None]:
    """同一输出目录串行，不阻塞其它数据集的清洗任务。"""
    key = path.resolve()
    with _LOCK_POOL_GUARD:
        entry = _LOCK_POOL.setdefault(key, _LockEntry(threading.Lock()))
        entry.users += 1
    entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _LOCK_POOL_GUARD:
            entry.users -= 1
            if entry.users == 0:
                _LOCK_POOL.pop(key, None)


def run_storage_data_clean(
    project_id: str,
    task_id: str,
    train_num: str,
    *,
    config: DataCleanConfig | None = None,
    **api_params: Any,
) -> dict:
    """对 ``{STORAGE}/{projectId}/{taskId}/{trainNum}/`` 执行清洗。

    ``api_params`` 支持：outputName, overwrite, skipCleanvision, requireCleanvision,
    thresholds, enabledFilters（与 HTTP body 字段一致）。
    """
    data_root = train_save_dir(project_id, task_id, train_num)
    if not data_root.is_dir():
        raise FileNotFoundError(f"训练数据目录不存在: {data_root}")

    cfg = config or DataCleanConfig.from_api_params(
        output_name=api_params.get("outputName"),
        overwrite=api_params.get("overwrite"),
        skip_cleanvision=api_params.get("skipCleanvision"),
        require_cleanvision=api_params.get("requireCleanvision"),
        thresholds=api_params.get("thresholds"),
        enabled_filters=api_params.get("enabledFilters"),
    )
    # 同步 FastAPI 路由在线程池中并发执行；同一输出目录必须串行提交。
    with _clean_output_lock(data_root / cfg.output_name):
        result = run_data_clean(data_root, config=cfg)
    payload = result.to_api_dict()
    payload["projectId"] = project_id
    payload["taskId"] = task_id
    payload["trainNum"] = train_num
    return payload
