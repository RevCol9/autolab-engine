"""跨进程 GPU 互斥锁。

推理（:21010）与训练（:21011）为独立进程，线程锁无法串行化显存访问。
本模块用 ``fcntl.flock`` 在锁文件上施加排他锁，保证同一 GPU 索引同一时刻
仅有一个进程执行 load/predict 或长时间训练。

锁在进程崩溃时随文件描述符关闭自动释放，不会永久死锁。
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEVICE_INDEX_RE = re.compile(r"(?:cuda:)?(\d+)$", re.IGNORECASE)


class GpuBusyError(RuntimeError):
    """无法在超时内获取 GPU 排他锁。"""


def parse_device_index(device: str) -> str:
    """从 ``cuda:0`` / ``0`` 解析 GPU 索引字符串。"""
    text = (device or "0").strip()
    match = _DEVICE_INDEX_RE.search(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return "0"


def lock_path_for_device(device_index: str) -> Path:
    """锁文件路径，可通过 ``NIII_GPU_LOCK_DIR`` 覆盖目录。"""
    lock_dir = Path(os.environ.get("NIII_GPU_LOCK_DIR", "/var/run")).expanduser()
    return lock_dir / f"niii-gpu{device_index}.lock"


class GpuDeviceLock:
    """单 GPU 索引的排他文件锁；持有期间须保持实例与打开的文件描述符。"""

    def __init__(self, device: str = "0") -> None:
        self.device_index = parse_device_index(device)
        self.path = lock_path_for_device(self.device_index)
        self._fd: Optional[int] = None

    def acquire(self, *, blocking: bool = True, timeout: float = 120.0) -> bool:
        """获取排他锁。``blocking=False`` 时立即返回；带 timeout 时轮询。"""
        if self._fd is not None:
            return True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o666)
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB

        if blocking and timeout > 0:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fd = fd
                    logger.debug("gpu lock acquired device=%s path=%s", self.device_index, self.path)
                    return True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        return False
                    time.sleep(0.2)
        try:
            fcntl.flock(fd, flags)
            self._fd = fd
            logger.debug("gpu lock acquired device=%s path=%s", self.device_index, self.path)
            return True
        except BlockingIOError:
            os.close(fd)
            return False

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            logger.debug("gpu lock released device=%s", self.device_index)

    def is_held_by_other(self) -> bool:
        """探测锁是否被其它进程持有（本实例未持锁时可用）。"""
        if self._fd is not None:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            except BlockingIOError:
                return True
        finally:
            os.close(fd)

    def __enter__(self) -> GpuDeviceLock:
        if not self.acquire(blocking=True, timeout=120.0):
            raise GpuBusyError(f"GPU {self.device_index} 忙，超时未获取锁")
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
