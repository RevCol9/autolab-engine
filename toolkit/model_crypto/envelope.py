"""KEK 装载与 DEK 封装（AES-256-GCM）。"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from toolkit.model_crypto.config import (
    ENV_MODEL_KEK,
    ENV_MODEL_KEK_ID,
    KEK_WRAP_ALG,
    ModelCryptoConfig,
    default_model_crypto_config,
)

_ENVELOPE_AAD = b"niii-model-crypto-envelope-v3"


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
    model_path: Path | None = None,
    config: ModelCryptoConfig | None = None,
) -> Path | None:
    """解析密钥目录：显式参数 > 配置/环境 > cwd/keys > 权重旁 keys/。"""
    if key_dir is not None:
        return Path(key_dir).expanduser()

    cfg = _cfg(config)
    configured = cfg.resolved_keys_dir()
    if configured is not None:
        return configured

    candidates = [Path.cwd() / "keys"]
    if model_path is not None:
        candidates.append(Path(model_path).parent / "keys")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def load_kek(
    *,
    kek: bytes | str | None = None,
    key_dir: str | Path | None = None,
    model_path: Path | None = None,
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
            os.environ.get(ENV_MODEL_KEK_ID, "").strip() or cfg.kek_id
        )

    env_kek = os.environ.get(ENV_MODEL_KEK, "").strip()
    if env_kek:
        return parse_kek_material(env_kek), (
            os.environ.get(ENV_MODEL_KEK_ID, "").strip() or cfg.kek_id
        )

    keys = resolve_keys_dir(key_dir, model_path=model_path, config=cfg)
    if keys is None:
        raise FileNotFoundError(
            f"未找到 KEK：请设置 {ENV_MODEL_KEK}，或配置 keys_dir 并放置 {cfg.kek_file_name}"
        )

    kek_path = keys / cfg.kek_file_name
    if not kek_path.is_file():
        raise FileNotFoundError(f"缺少 KEK 文件: {kek_path}")

    material = parse_kek_material(kek_path.read_bytes())
    kek_id = os.environ.get(ENV_MODEL_KEK_ID, "").strip()
    if not kek_id:
        id_path = keys / cfg.kek_id_file_name
        if id_path.is_file():
            kek_id = id_path.read_text(encoding="utf-8").strip()
    return material, kek_id or cfg.kek_id


def init_kek(
    key_dir: str | Path,
    *,
    kek_id: str | None = None,
    overwrite: bool = False,
    config: ModelCryptoConfig | None = None,
) -> tuple[Path, str]:
    """在密钥目录写入随机 KEK（32 字节）及 kek_id。"""
    cfg = _cfg(config)
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    kek_path = key_dir / cfg.kek_file_name
    id_path = key_dir / cfg.kek_id_file_name
    kid = (kek_id or cfg.kek_id).strip() or cfg.kek_id

    if kek_path.exists() and not overwrite:
        raise FileExistsError(f"KEK 已存在: {kek_path}")

    kek_path.write_bytes(secrets.token_bytes(32))
    id_path.write_text(kid + "\n", encoding="utf-8")
    try:
        os.chmod(kek_path, 0o600)
    except OSError:
        pass
    return kek_path, kid


def wrap_dek(kek: bytes, dek: bytes, *, kek_id: str) -> dict[str, Any]:
    """KEK 封装 DEK，结果写入 checkpoint.envelope。"""
    if len(kek) != 32 or len(dek) != 32:
        raise ValueError("KEK/DEK 均须为 32 字节")
    nonce = secrets.token_bytes(12)
    packed = AESGCM(kek).encrypt(nonce, dek, _ENVELOPE_AAD)
    return {
        "alg": KEK_WRAP_ALG,
        "kek_id": kek_id,
        "nonce": nonce,
        "aad": _ENVELOPE_AAD,
        "dek_wrapped": packed[:-16],
        "tag": packed[-16:],
    }


def unwrap_dek(kek: bytes, envelope: dict[str, Any]) -> bytes:
    """从 envelope 解出 DEK；认证失败视为密钥错误或密文损坏。"""
    if not envelope:
        raise ValueError("缺少 envelope")
    alg = envelope.get("alg") or KEK_WRAP_ALG
    if alg != KEK_WRAP_ALG:
        raise ValueError(f"不支持的 envelope 算法: {alg}")
    nonce = bytes(envelope["nonce"])
    wrapped = bytes(envelope["dek_wrapped"]) + bytes(envelope["tag"])
    aad = bytes(envelope["aad"]) if "aad" in envelope else None
    try:
        return AESGCM(kek).decrypt(nonce, wrapped, aad)
    except Exception as exc:
        raise ValueError("无法解开 DEK：KEK 不匹配或 envelope 已损坏") from exc
