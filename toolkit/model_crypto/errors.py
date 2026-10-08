"""Stable error contract for encrypted-model runtime callers."""

from __future__ import annotations


class ModelCryptoError(RuntimeError):
    """One runtime exception type with a stable machine-readable code."""

    KEY_NOT_FOUND = "MODEL_KEY_NOT_FOUND"
    KEY_INSECURE = "MODEL_KEY_INSECURE"
    KEY_INVALID = "MODEL_KEY_INVALID"
    KEY_ID_MISMATCH = "MODEL_KEY_ID_MISMATCH"
    AUTH_FAILED = "MODEL_AUTH_FAILED"
    FORMAT_INVALID = "MODEL_FORMAT_INVALID"
    TASK_MISMATCH = "MODEL_TASK_MISMATCH"
    ARCH_UNSUPPORTED = "MODEL_ARCH_UNSUPPORTED"
    DEVICE_UNAVAILABLE = "MODEL_DEVICE_UNAVAILABLE"
    WRITE_FAILED = "MODEL_WRITE_FAILED"

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


__all__ = ["ModelCryptoError"]
