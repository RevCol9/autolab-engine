"""训练前数据清洗：解析 storage 路径并调用 toolkit。"""

from __future__ import annotations

from typing import Any

from toolkit.data_clean import DataCleanConfig, run_data_clean
from training.paths import train_save_dir


def run_storage_data_clean(
    project_id: str,
    task_id: str,
    train_num: str,
    *,
    config: DataCleanConfig | None = None,
    **api_params: Any,
) -> dict:
    """
  对 ``{STORAGE}/{projectId}/{taskId}/{trainNum}/`` 执行清洗。

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
    result = run_data_clean(data_root, config=cfg)
    payload = result.to_api_dict()
    payload["projectId"] = project_id
    payload["taskId"] = task_id
    payload["trainNum"] = train_num
    return payload
