"""跨进程 GPU 互斥锁。

推理（:21010）与训练（:21011）为独立进程，线程锁无法串行化显存访问。
本模块在锁文件上施加排他锁（Linux 使用 ``fcntl``，Windows 使用 ``msvcrt``），
保证同一物理 GPU 同一时刻仅有一个进程执行 load/predict 或长时间训练。

锁在进程崩溃时随文件描述符关闭自动释放，不会永久死锁。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from shared.device import device_lock_key, parse_device_index

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only
    fcntl = None
    import msvcrt
else:  # pragma: no cover - selected by platform
    msvcrt = None

logger = logging.getLogger(__name__)


def _try_lock(fd: int) -> bool:
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    assert msvcrt is not None
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False


def _unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    assert msvcrt is not None
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def lock_path_for_device(device: str) -> Path:
    """锁文件路径，可通过 ``NIII_GPU_LOCK_DIR`` 覆盖目录。"""
    lock_dir = Path(os.environ.get("NIII_GPU_LOCK_DIR", "/var/run")).expanduser()
    return lock_dir / f"niii-gpu{device_lock_key(device)}.lock"


class GpuDeviceLock:
    """单 GPU 索引的排他文件锁；持有期间须保持实例与打开的文件描述符。"""

    def __init__(self, device: str = "0") -> None:
        self.device_index = parse_device_index(device)
        self.device_key = device_lock_key(device)
        lock_dir = Path(os.environ.get("NIII_GPU_LOCK_DIR", "/var/run")).expanduser()
        self.path = lock_dir / f"niii-gpu{self.device_key}.lock"
        self._fd: Optional[int] = None

    def acquire(self, *, blocking: bool = True, timeout: float = 120.0) -> bool:
        """获取排他锁。``blocking=False`` 时立即返回；带 timeout 时轮询。"""
        if self._fd is not None:
            return True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o666)
        if fcntl is None:
            os.ftruncate(fd, 1)

        if blocking:
            deadline = time.monotonic() + timeout if timeout > 0 else None
            while True:
                if _try_lock(fd):
                    self._fd = fd
                    logger.debug("gpu lock acquired device=%s path=%s", self.device_index, self.path)
                    return True
                if deadline is not None and time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.2)
        if _try_lock(fd):
            self._fd = fd
            logger.debug("gpu lock acquired device=%s path=%s", self.device_index, self.path)
            return True
        os.close(fd)
        return False

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            _unlock(self._fd)
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
        if fcntl is None:
            os.ftruncate(fd, 1)
        try:
            if _try_lock(fd):
                _unlock(fd)
                return False
            return True
        finally:
            os.close(fd)
