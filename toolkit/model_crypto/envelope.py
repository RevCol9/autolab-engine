"""KEK 装载与 DEK 封装（AES-256-GCM）。"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
import stat
from pathlib import Path
from toolkit.model_crypto.config import (
    ENV_MODEL_KEK,
    ENV_MODEL_KEK_ID,
    ModelCryptoConfig,
    default_model_crypto_config,
    validate_key_id,
)


def _cfg(config: ModelCryptoConfig | None) -> ModelCryptoConfig:
    return config or default_model_crypto_config()


def parse_kek_material(raw: str | bytes) -> bytes:
    """将 hex / base64 / 原始 32 字节解析为 KEK。"""
    if isinstance(raw, bytes):
        if len(raw) == 32:
            return raw
        try:
            text = raw.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("KEK 必须是 32 字节（raw / hex / base64）") from exc
    else:
        text = str(raw).strip()

    if not text:
        raise ValueError("KEK 为空")

    try:
        decoded = binascii.unhexlify(text)
        if len(decoded) == 32:
            return decoded
    except (binascii.Error, ValueError):
        pass

    try:
        decoded = base64.b64decode(text, validate=True)
        if len(decoded) == 32:
            return decoded
    except (binascii.Error, ValueError):
        pass

    raw_bytes = text.encode("utf-8")
    if len(raw_bytes) == 32:
        return raw_bytes
    raise ValueError("KEK 必须是 32 字节（raw / hex / base64）")


def resolve_keys_dir(
    key_dir: str | Path | None = None,
    *,
    config: ModelCryptoConfig | None = None,
) -> Path | None:
    """Resolve an explicitly supplied or configured key directory."""
    if key_dir is not None:
        return Path(key_dir).expanduser().resolve()

    cfg = _cfg(config)
    configured = cfg.resolved_keys_dir()
    if configured is not None:
        return configured

    return None


def _read_key_file(path: Path) -> bytes:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"缺少 KEK 文件: {path}") from None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"KEK 路径必须是普通文件且不能是符号链接: {path}")
    if os.name != "nt" and stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise PermissionError(f"KEK 文件权限必须限制为 0600: {path}")
    if file_stat.st_size > 4096:
        raise ValueError(f"KEK 文件异常过大: {path}")
    return path.read_bytes()


def _read_key_id_file(path: Path) -> str | None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"kek_id 路径必须是普通文件且不能是符号链接: {path}")
    if file_stat.st_size > 512:
        raise ValueError(f"kek_id 文件异常过大: {path}")
    return validate_key_id(path.read_text(encoding="utf-8"))


def load_kek(
    *,
    kek: bytes | str | None = None,
    key_dir: str | Path | None = None,
    config: ModelCryptoConfig | None = None,
) -> tuple[bytes, str]:
    """
    装载 KEK 与 kek_id。

    材料优先级：显式 kek → NIII_MODEL_KEK → {keys_dir}/kek.key
    标识优先级：NIII_MODEL_KEK_ID → kek_id.txt → 配置 kek_id
    """
    cfg = _cfg(config)

    if kek is not None:
        return parse_kek_material(kek), (
            validate_key_id(os.environ.get(ENV_MODEL_KEK_ID, "").strip() or cfg.kek_id)
        )

    env_kek = os.environ.get(ENV_MODEL_KEK, "").strip()
    if env_kek:
        return parse_kek_material(env_kek), (
            validate_key_id(os.environ.get(ENV_MODEL_KEK_ID, "").strip() or cfg.kek_id)
        )

    keys = resolve_keys_dir(key_dir, config=cfg)
    if keys is None:
        raise FileNotFoundError(
            f"未找到 KEK：请设置 {ENV_MODEL_KEK}，或配置 keys_dir 并放置 {cfg.kek_file_name}"
        )

    kek_path = keys / cfg.kek_file_name
    material = parse_kek_material(_read_key_file(kek_path))
    kek_id = os.environ.get(ENV_MODEL_KEK_ID, "").strip()
    if not kek_id:
        kek_id = _read_key_id_file(keys / cfg.kek_id_file_name) or cfg.kek_id
    return material, validate_key_id(kek_id)


def init_kek(
    key_dir: str | Path,
    *,
    kek_id: str | None = None,
    config: ModelCryptoConfig | None = None,
) -> tuple[Path, str]:
    """在密钥目录写入随机 KEK（32 字节）及 kek_id。"""
    cfg = _cfg(config)
    key_dir = Path(key_dir).expanduser().resolve()
    key_dir.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(key_dir, 0o700)
    kek_path = key_dir / cfg.kek_file_name
    id_path = key_dir / cfg.kek_id_file_name
    kid = validate_key_id(kek_id or cfg.kek_id)

    if kek_path.exists() or id_path.exists():
        raise FileExistsError(
            f"KEK 或 kek_id 已存在，拒绝覆盖；轮换必须使用新的 key ID/目录: {key_dir}"
        )

    key_created = False
    id_created = False
    try:
        key_descriptor = os.open(
            kek_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        key_created = True
        with os.fdopen(key_descriptor, "wb") as stream:
            stream.write(secrets.token_bytes(32))
            stream.flush()
            os.fsync(stream.fileno())

        id_descriptor = os.open(
            id_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        id_created = True
        with os.fdopen(id_descriptor, "w", encoding="utf-8") as stream:
            stream.write(kid + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if key_created:
            kek_path.unlink(missing_ok=True)
        if id_created:
            id_path.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        directory_descriptor = os.open(key_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    return kek_path, kid
