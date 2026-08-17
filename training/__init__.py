"""autolab 自有训练算法模块。

从 niii_training_algorithm 移植核心能力（闭环检测训练），去掉 Django / 内嵌 ultralytics 8.0 / SAM。
推理请使用仓库根目录 app/（autolab-engine :21010）。
"""

from training.service import MANAGER, JobManager
from training.trainer import build_closed_loop_cmd, run_detection_train

__all__ = [
    "MANAGER",
    "JobManager",
    "build_closed_loop_cmd",
    "run_detection_train",
]
