"""受限文件目录中的本地 KEK keyring。"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
import stat
import tempfile
from pathlib import Path

from toolkit.model_crypto.config import (
    ENV_MODEL_KEK,
    ENV_MODEL_KEK_ID,
    ModelCryptoConfig,
    default_model_crypto_config,
    validate_key_id,
)

KEYRING_DIR_NAME = "keyring"


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


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _key_file_name(key_id: str) -> str:
    encoded = base64.urlsafe_b64encode(validate_key_id(key_id).encode("utf-8"))
    return f"kek-{encoded.decode('ascii').rstrip('=')}.key"


def _key_id_from_file_name(name: str) -> str:
    if not name.startswith("kek-") or not name.endswith(".key"):
        raise ValueError(f"密钥文件名非法: {name}")
    encoded = name[4:-4]
    try:
        raw = base64.b64decode(
            encoded + ("=" * (-len(encoded) % 4)),
            altchars=b"-_",
            validate=True,
        )
        key_id = validate_key_id(raw.decode("utf-8"))
    except (binascii.Error, UnicodeError, ValueError) as exc:
        raise ValueError(f"密钥文件名非法: {name}") from exc
    if _key_file_name(key_id) != name:
        raise ValueError(f"密钥文件名不是规范编码: {name}")
    return key_id


def _keyring_directory(keys: Path, *, create: bool) -> Path:
    path = keys / KEYRING_DIR_NAME
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        if not create:
            raise FileNotFoundError(f"密钥目录不存在: {path}") from None
        path.mkdir(mode=0o700)
        return path
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISDIR(file_stat.st_mode):
        raise ValueError(f"keyring 必须是普通目录且不能是符号链接: {path}")
    if os.name != "nt" and stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise PermissionError(f"keyring 目录权限必须限制为 0700: {path}")
    return path


def _keyring_key_path(keys: Path, key_id: str, *, create_dir: bool) -> Path:
    return _keyring_directory(keys, create=create_dir) / _key_file_name(key_id)


def _write_new_key(path: Path, material: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(material)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_active_key_id(path: Path, key_id: str) -> None:
    _read_key_id_file(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(validate_key_id(key_id) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _active_key_id(keys: Path, cfg: ModelCryptoConfig) -> str:
    env_id = os.environ.get(ENV_MODEL_KEK_ID, "").strip()
    if env_id:
        return validate_key_id(env_id)
    return _stored_active_key_id(keys, cfg)


def _stored_active_key_id(keys: Path, cfg: ModelCryptoConfig) -> str:
    return _read_key_id_file(keys / cfg.kek_id_file_name) or cfg.kek_id


def _register_legacy_key(
    keys: Path,
    cfg: ModelCryptoConfig,
    active_key_id: str,
) -> Path:
    keyring_path = _keyring_key_path(keys, active_key_id, create_dir=True)
    if keyring_path.exists():
        return keyring_path
    legacy_path = keys / cfg.kek_file_name
    material = parse_kek_material(_read_key_file(legacy_path))
    _write_new_key(keyring_path, material)
    _fsync_directory(keyring_path.parent)
    return keyring_path


def load_kek(
    *,
    kek: bytes | str | None = None,
    key_dir: str | Path | None = None,
    key_id: str | None = None,
    config: ModelCryptoConfig | None = None,
) -> tuple[bytes, str]:
    """
    装载 KEK 与 kek_id。

    材料优先级：显式 kek → NIII_MODEL_KEK → keyring → 旧版 kek.key。
    未指定 key_id 时读取 active key；指定后可读取历史 key。
    """
    cfg = _cfg(config)

    if kek is not None:
        configured_id = validate_key_id(
            os.environ.get(ENV_MODEL_KEK_ID, "").strip() or cfg.kek_id
        )
        selected_id = validate_key_id(key_id or configured_id)
        if selected_id != configured_id:
            raise FileNotFoundError(f"显式 KEK 不包含历史 key: {selected_id}")
        return parse_kek_material(kek), selected_id

    env_kek = os.environ.get(ENV_MODEL_KEK, "").strip()
    if env_kek:
        configured_id = validate_key_id(
            os.environ.get(ENV_MODEL_KEK_ID, "").strip() or cfg.kek_id
        )
        selected_id = validate_key_id(key_id or configured_id)
        if selected_id != configured_id:
            raise FileNotFoundError(f"环境 KEK 不包含历史 key: {selected_id}")
        return parse_kek_material(env_kek), selected_id

    keys = resolve_keys_dir(key_dir, config=cfg)
    if keys is None:
        raise FileNotFoundError(
            f"未找到 KEK：请设置 {ENV_MODEL_KEK}，或配置 keys_dir 并放置 {cfg.kek_file_name}"
        )

    active_key_id = _active_key_id(keys, cfg)
    selected_id = validate_key_id(key_id or active_key_id)
    try:
        key_path = _keyring_key_path(keys, selected_id, create_dir=False)
    except FileNotFoundError:
        key_path = None
    if key_path is not None and key_path.exists():
        return parse_kek_material(_read_key_file(key_path)), selected_id
    if selected_id == active_key_id:
        legacy_path = keys / cfg.kek_file_name
        if legacy_path.exists():
            return parse_kek_material(_read_key_file(legacy_path)), selected_id
    raise FileNotFoundError(f"keyring 中不存在 key_id 对应的 KEK: {selected_id}")


def init_kek(
    key_dir: str | Path,
    *,
    kek_id: str | None = None,
    config: ModelCryptoConfig | None = None,
) -> tuple[Path, str]:
    """创建本地 keyring，并将首个随机 KEK 设为 active。"""
    cfg = _cfg(config)
    key_dir = Path(key_dir).expanduser().resolve()
    key_dir.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(key_dir, 0o700)
    id_path = key_dir / cfg.kek_id_file_name
    kid = validate_key_id(kek_id or cfg.kek_id)

    legacy_path = key_dir / cfg.kek_file_name
    keyring_dir = key_dir / KEYRING_DIR_NAME
    if legacy_path.exists() or id_path.exists() or keyring_dir.exists():
        raise FileExistsError(
            f"keyring 或旧版 KEK 已存在，拒绝覆盖；请使用 rotate-kek: {key_dir}"
        )

    keyring_dir.mkdir(mode=0o700)
    kek_path = keyring_dir / _key_file_name(kid)
    try:
        _write_new_key(kek_path, secrets.token_bytes(32))
        _fsync_directory(keyring_dir)
        _write_active_key_id(id_path, kid)
    except Exception:
        kek_path.unlink(missing_ok=True)
        id_path.unlink(missing_ok=True)
        try:
            keyring_dir.rmdir()
        except OSError:
            pass
        raise
    _fsync_directory(key_dir)
    return kek_path, kid


def rotate_kek(
    key_dir: str | Path,
    *,
    kek_id: str,
    config: ModelCryptoConfig | None = None,
) -> tuple[Path, str]:
    """Create a new KEK and atomically make it active, retaining the old key."""
    cfg = _cfg(config)
    keys = Path(key_dir).expanduser().resolve()
    if not keys.is_dir():
        raise FileNotFoundError(f"密钥目录不存在: {keys}")
    active_id = _stored_active_key_id(keys, cfg)
    new_id = validate_key_id(kek_id)
    if new_id == active_id:
        raise ValueError(f"新 KEK ID 已经是 active: {new_id}")

    try:
        active_path = _keyring_key_path(keys, active_id, create_dir=False)
    except FileNotFoundError:
        active_path = _register_legacy_key(keys, cfg, active_id)
    if not active_path.exists():
        active_path = _register_legacy_key(keys, cfg, active_id)
    _read_key_file(active_path)

    new_path = _keyring_key_path(keys, new_id, create_dir=True)
    _write_new_key(new_path, secrets.token_bytes(32))
    _fsync_directory(new_path.parent)
    _write_active_key_id(keys / cfg.kek_id_file_name, new_id)
    return new_path, new_id


def set_active_kek(
    key_dir: str | Path,
    key_id: str,
    *,
    config: ModelCryptoConfig | None = None,
) -> str:
    """Atomically switch new writes to an existing keyring entry."""
    cfg = _cfg(config)
    keys = Path(key_dir).expanduser().resolve()
    selected_id = validate_key_id(key_id)
    try:
        key_path = _keyring_key_path(keys, selected_id, create_dir=False)
    except FileNotFoundError:
        active_id = _stored_active_key_id(keys, cfg)
        if selected_id != active_id:
            raise FileNotFoundError(
                f"keyring 中不存在 key_id 对应的 KEK: {selected_id}"
            ) from None
        key_path = _register_legacy_key(keys, cfg, active_id)
    _read_key_file(key_path)
    _write_active_key_id(keys / cfg.kek_id_file_name, selected_id)
    return selected_id


def keyring_status(
    key_dir: str | Path,
    *,
    config: ModelCryptoConfig | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Return the active key ID and all locally retained key IDs."""
    cfg = _cfg(config)
    keys = Path(key_dir).expanduser().resolve()
    active_id = _stored_active_key_id(keys, cfg)
    key_ids: list[str] = []
    try:
        keyring_dir = _keyring_directory(keys, create=False)
    except FileNotFoundError:
        keyring_dir = None
    if keyring_dir is not None:
        for path in keyring_dir.iterdir():
            file_stat = path.lstat()
            if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"keyring 只允许普通密钥文件: {path}")
            parse_kek_material(_read_key_file(path))
            key_ids.append(_key_id_from_file_name(path.name))
    if not key_ids and (keys / cfg.kek_file_name).exists():
        key_ids.append(active_id)
    if active_id not in key_ids:
        raise FileNotFoundError(f"active KEK 不存在: {active_id}")
    return active_id, tuple(sorted(key_ids))
