"""`.niii-model` 密钥位置配置；密钥材料本身不得写入 YAML。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from shared.config_yaml import load_yaml_file

ENV_MODEL_KEK = "NIII_MODEL_KEK"
ENV_MODEL_KEK_ID = "NIII_MODEL_KEK_ID"
ENV_MODEL_KEYS_DIR = "NIII_MODEL_KEYS_DIR"
ENV_MODEL_CRYPTO_CONFIG = "NIII_MODEL_CRYPTO_CONFIG"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_CANDIDATES = (
    _REPO_ROOT / "config" / "model_crypto.yaml",
)
_CONFIG_KEYS = frozenset(
    {"keys_dir", "kek_id", "kek_file_name", "kek_id_file_name"}
)
_SAFE_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_key_id(value: Any) -> str:
    key_id = str(value or "").strip()
    if not key_id or len(key_id.encode("utf-8")) > 128:
        raise ValueError("kek_id 的 UTF-8 长度必须位于 1..128 字节")
    if any(ord(char) < 32 for char in key_id):
        raise ValueError("kek_id 不能包含控制字符")
    return key_id


def _validate_file_name(value: Any, *, field: str, default: str) -> str:
    name = default if value is None else str(value).strip()
    if not _SAFE_FILE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"{field} 必须是安全的单层文件名")
    return name


@dataclass(frozen=True)
class ModelCryptoConfig:
    """KEK 文件位置与标识；KEK 字节不进入本对象。"""

    keys_dir: str | None = None
    kek_id: str = "default"
    kek_file_name: str = "kek.key"
    kek_id_file_name: str = "kek_id.txt"

    def resolved_keys_dir(self) -> Path | None:
        env_dir = os.environ.get(ENV_MODEL_KEYS_DIR, "").strip()
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        if self.keys_dir:
            return Path(self.keys_dir).expanduser().resolve()
        return None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ModelCryptoConfig:
        raw = dict(data or {})
        unknown = sorted(set(raw) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"模型加密配置包含不支持的字段: {unknown}")
        return cls(
            keys_dir=_optional_str(raw.get("keys_dir")),
            kek_id=validate_key_id(raw.get("kek_id", "default")),
            kek_file_name=_validate_file_name(
                raw.get("kek_file_name"),
                field="kek_file_name",
                default="kek.key",
            ),
            kek_id_file_name=_validate_file_name(
                raw.get("kek_id_file_name"),
                field="kek_id_file_name",
                default="kek_id.txt",
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> ModelCryptoConfig:
        cfg_path = _resolve_config_path(path)
        if cfg_path is None:
            return cls()
        return cls.from_mapping(load_yaml_file(cfg_path))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"加密配置不存在: {candidate}")
        return candidate

    env_path = os.environ.get(ENV_MODEL_CRYPTO_CONFIG, "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"加密配置不存在: {candidate}")
        return candidate

    for candidate in _CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


_CACHED: ModelCryptoConfig | None = None


def default_model_crypto_config(*, reload: bool = False) -> ModelCryptoConfig:
    """进程内默认配置：YAML → 环境变量覆盖 keys_dir / kek_id。"""
    global _CACHED
    if _CACHED is not None and not reload:
        return _CACHED

    cfg = ModelCryptoConfig.from_yaml()
    env_keys = os.environ.get(ENV_MODEL_KEYS_DIR, "").strip()
    env_kid = os.environ.get(ENV_MODEL_KEK_ID, "").strip()
    updates: dict[str, Any] = {}
    if env_keys:
        updates["keys_dir"] = env_keys
    if env_kid:
        updates["kek_id"] = validate_key_id(env_kid)
    if updates:
        cfg = replace(cfg, **updates)
    _CACHED = cfg
    return cfg
