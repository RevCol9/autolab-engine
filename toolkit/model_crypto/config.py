"""信封加密策略配置；非密钥材料统一由 YAML 维护。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from shared.config_yaml import load_yaml_file

ENV_MODEL_KEK = "NIII_MODEL_KEK"
ENV_MODEL_KEK_ID = "NIII_MODEL_KEK_ID"
ENV_MODEL_KEYS_DIR = "NIII_MODEL_KEYS_DIR"
ENV_MODEL_CRYPTO_CONFIG = "NIII_MODEL_CRYPTO_CONFIG"

CRYPTO_VERSION = 2
KEK_WRAP_ALG = "AES-256-GCM"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_CANDIDATES = (
    _REPO_ROOT / "config" / "model_crypto.yaml",
    _REPO_ROOT / "config" / "model_crypto.example.yaml",
)

_DEFAULT_EXCLUDE = frozenset(
    {
        "demo",
        "test",
        "tests",
        ".git",
        "__pycache__",
        "keys",
        "develop",
        "release",
        "dist",
        "build",
        "toolkit",
        ".venv",
        "venv",
    }
)


@dataclass(frozen=True)
class ModelCryptoConfig:
    """加解密策略：路径约定、扫描规则、信封版本号。KEK 字节不进入本对象。"""

    keys_dir: str | None = None
    kek_id: str = "default"
    kek_file_name: str = "kek.key"
    kek_id_file_name: str = "kek_id.txt"
    encrypted_mark: str = "encrypted"
    enc_suffix: str = "_enc"
    crypto_version: int = CRYPTO_VERSION
    exclude_dirs: frozenset[str] = field(default_factory=lambda: _DEFAULT_EXCLUDE)
    weight_suffixes: frozenset[str] = field(
        default_factory=lambda: frozenset({".pt", ".pth"})
    )

    @property
    def default_kek_id(self) -> str:
        return self.kek_id

    def encrypted_path_for(self, src: Path) -> Path:
        if src.stem.endswith(self.enc_suffix):
            return src
        return src.with_name(f"{src.stem}{self.enc_suffix}{src.suffix}")

    def plain_path_for_enc(self, enc: Path) -> Path:
        if not enc.stem.endswith(self.enc_suffix):
            raise ValueError(f"不是加密后缀文件: {enc}")
        return enc.with_name(f"{enc.stem[: -len(self.enc_suffix)]}{enc.suffix}")

    def resolved_keys_dir(self) -> Path | None:
        env_dir = os.environ.get(ENV_MODEL_KEYS_DIR, "").strip()
        if env_dir:
            return Path(env_dir).expanduser()
        if self.keys_dir:
            return Path(self.keys_dir).expanduser()
        return None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ModelCryptoConfig:
        raw = dict(data or {})
        exclude = raw.get("exclude_dirs")
        suffixes = raw.get("weight_suffixes")
        return cls(
            keys_dir=_optional_str(raw.get("keys_dir")),
            kek_id=str(raw.get("kek_id") or "default"),
            kek_file_name=str(raw.get("kek_file_name") or "kek.key"),
            kek_id_file_name=str(raw.get("kek_id_file_name") or "kek_id.txt"),
            encrypted_mark=str(raw.get("encrypted_mark") or "encrypted"),
            enc_suffix=str(raw.get("enc_suffix") or "_enc"),
            crypto_version=int(raw.get("crypto_version") or CRYPTO_VERSION),
            exclude_dirs=frozenset(str(x) for x in exclude) if exclude else _DEFAULT_EXCLUDE,
            weight_suffixes=(
                frozenset(str(x) if str(x).startswith(".") else f".{x}" for x in suffixes)
                if suffixes
                else frozenset({".pt", ".pth"})
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
        updates["kek_id"] = env_kid
    if updates:
        cfg = replace(cfg, **updates)
    _CACHED = cfg
    return cfg
