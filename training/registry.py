"""训练任务注册表（对外 re-export）。"""

from training.backends import get_backend, registered_tasks

__all__ = ["get_backend", "registered_tasks"]
