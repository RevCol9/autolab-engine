"""YOLO 权重信封加解密：KEK 封装 DEK，DEK 加密 state_dict。"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from toolkit.model_crypto.config import (
    CRYPTO_VERSION,
    ModelCryptoConfig,
    default_model_crypto_config,
)
from toolkit.model_crypto.envelope import (
    init_kek,
    load_kek,
    unwrap_dek,
    wrap_dek,
)

logger = logging.getLogger(__name__)
_PAYLOAD_AUTH_ALG = "HMAC-SHA256"
_PAYLOAD_AUTH_CONTEXT = b"niii-model-crypto-payload-auth-v1"


def _cfg(config: ModelCryptoConfig | None) -> ModelCryptoConfig:
    return config or default_model_crypto_config()


def _peek_checkpoint(path: Path, config: ModelCryptoConfig) -> dict[str, Any] | None:
    if not path.exists() or path.suffix.lower() not in config.weight_suffixes:
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint if isinstance(checkpoint, dict) else None
    except Exception:
        return None


def is_encrypted_checkpoint(
    path: str | Path,
    *,
    config: ModelCryptoConfig | None = None,
) -> bool:
    cfg = _cfg(config)
    ckpt = _peek_checkpoint(Path(path), cfg)
    return bool(
        ckpt
        and ckpt.get(cfg.encrypted_mark)
        and "encrypted_state_dict" in ckpt
        and "envelope" in ckpt
    )


def _extract_yolo_model(ckpt: dict[str, Any]):
    model = ckpt.get("model")
    if model is None or not hasattr(model, "state_dict"):
        model = ckpt.get("ema")
    if model is not None and hasattr(model, "state_dict"):
        return model
    return None


def is_yolo_checkpoint(
    path: str | Path,
    *,
    config: ModelCryptoConfig | None = None,
) -> bool:
    cfg = _cfg(config)
    ckpt = _peek_checkpoint(Path(path), cfg)
    if not ckpt or not isinstance(ckpt, dict):
        return False
    if ckpt.get(cfg.encrypted_mark):
        return True
    return _extract_yolo_model(ckpt) is not None


def _ctr_crypt(key: bytes, nonce: bytes, payload: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError(f"AES key 须 32 字节，当前 {len(key)}")
    if len(nonce) != 16:
        raise ValueError(f"CTR nonce 须 16 字节，当前 {len(nonce)}")
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    cryptor = cipher.encryptor()
    return cryptor.update(payload) + cryptor.finalize()


def _canonical_json(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return str(item)

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _mac_chunk(mac: hmac.HMAC, payload: bytes) -> None:
    mac.update(len(payload).to_bytes(8, "big"))
    mac.update(payload)


def _payload_auth_tag(dek: bytes, checkpoint: dict[str, Any]) -> bytes:
    """认证全部密文张量及影响模型解释方式的元数据。"""
    mac_key = hmac.new(dek, _PAYLOAD_AUTH_CONTEXT, hashlib.sha256).digest()
    mac = hmac.new(mac_key, digestmod=hashlib.sha256)
    encrypted = checkpoint["encrypted_state_dict"]
    dtype_meta = checkpoint["dtype_metadata"]
    nonce_meta = checkpoint["nonce_metadata"]
    for name in sorted(encrypted):
        tensor = encrypted[name].detach().cpu().contiguous()
        _mac_chunk(mac, name.encode("utf-8"))
        _mac_chunk(mac, str(dtype_meta[name]).encode("ascii"))
        _mac_chunk(mac, _canonical_json(list(tensor.shape)))
        _mac_chunk(mac, bytes(nonce_meta[name]))
        _mac_chunk(mac, tensor.numpy().tobytes())
    _mac_chunk(mac, _canonical_json(checkpoint.get("model_yaml")))
    _mac_chunk(mac, _canonical_json(checkpoint.get("model_names")))
    return mac.digest()


def _verify_payload_auth(dek: bytes, checkpoint: dict[str, Any]) -> None:
    auth = checkpoint.get("payload_auth")
    if not isinstance(auth, dict):
        raise ValueError("加密权重缺少 payload_auth；请用 crypto_version=3 重新加密")
    if auth.get("alg") != _PAYLOAD_AUTH_ALG:
        raise ValueError(f"不支持的 payload_auth 算法: {auth.get('alg')!r}")
    expected = _payload_auth_tag(dek, checkpoint)
    try:
        actual = bytes(auth["tag"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("加密权重 payload_auth.tag 无效") from exc
    if not hmac.compare_digest(actual, expected):
        raise ValueError("加密权重完整性校验失败：密文或模型元数据已损坏")


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    """同目录落临时文件并 fsync，成功后原子替换目标。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class _TensorCipher:
    """DEK 下的按张量 AES-CTR；每个参数独立 nonce，避免密钥流复用。"""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("DEK 须 32 字节")
        self.key = key

    def encrypt_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bytes]:
        nonce = secrets.token_bytes(16)
        arr = tensor.detach().cpu().float().numpy()
        encrypted = _ctr_crypt(self.key, nonce, arr.tobytes())
        out = np.frombuffer(encrypted, dtype=np.float32)
        return torch.from_numpy(out.copy()).view(tensor.shape), nonce

    def decrypt_tensor(
        self,
        encrypted: torch.Tensor,
        dtype: torch.dtype,
        *,
        nonce: bytes,
    ) -> torch.Tensor:
        arr = encrypted.detach().cpu().float().numpy()
        decrypted = _ctr_crypt(self.key, nonce, arr.tobytes())
        out = np.frombuffer(decrypted, dtype=np.float32)
        return torch.from_numpy(out.copy()).view(encrypted.shape).to(dtype)


class ModelDecryptor:
    """内存解密：KEK → DEK → state_dict，不写明文权重。"""

    def __init__(
        self,
        key_dir: str | Path | None = None,
        *,
        kek: bytes | str | None = None,
        config: ModelCryptoConfig | None = None,
    ) -> None:
        self.config = _cfg(config)
        self.key_dir = Path(key_dir) if key_dir is not None else None
        self._kek_arg = kek
        self._kek: bytes | None = None
        self._kek_id: str | None = None

    def _ensure_kek(self, model_path: Path | None = None) -> tuple[bytes, str]:
        if self._kek is not None and self._kek_id is not None:
            return self._kek, self._kek_id
        self._kek, self._kek_id = load_kek(
            kek=self._kek_arg,
            key_dir=self.key_dir,
            model_path=model_path,
            config=self.config,
        )
        return self._kek, self._kek_id

    def read_encrypted_checkpoint(self, encrypted_path: str | Path) -> dict[str, Any]:
        encrypted_path = Path(encrypted_path)
        checkpoint = torch.load(encrypted_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{encrypted_path} 不是有效的检查点字典")
        if not checkpoint.get(self.config.encrypted_mark):
            raise ValueError(f"{encrypted_path} 不是加密权重")
        if "encrypted_state_dict" not in checkpoint or "envelope" not in checkpoint:
            raise ValueError(f"{encrypted_path} 缺少 encrypted_state_dict / envelope")
        if "nonce_metadata" not in checkpoint:
            raise ValueError(f"{encrypted_path} 缺少 nonce_metadata")
        if "payload_auth" not in checkpoint and checkpoint["envelope"].get("aad"):
            raise ValueError(
                f"{encrypted_path} 缺少 payload_auth，文件可能已损坏"
            )
        return checkpoint

    def get_model_metadata(
        self,
        checkpoint: dict[str, Any],
        meta_path: str | Path | None = None,
    ) -> tuple[dict, dict]:
        if "model_yaml" in checkpoint and "model_names" in checkpoint:
            return checkpoint["model_yaml"], checkpoint["model_names"]
        if not meta_path:
            raise ValueError("加密文件缺少 model_yaml/model_names")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        model = _extract_yolo_model(meta)
        if model is None:
            raise ValueError(f"{meta_path} 无法提取 model/ema")
        return model.yaml, dict(model.names)

    def decrypt_state_dict(
        self,
        encrypted_path: str | Path | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        path = Path(encrypted_path) if encrypted_path is not None else None
        checkpoint = checkpoint or self.read_encrypted_checkpoint(path)
        kek, _ = self._ensure_kek(path)
        dek = unwrap_dek(kek, checkpoint["envelope"])
        cipher = _TensorCipher(dek)

        dtype_meta = checkpoint["dtype_metadata"]
        nonce_meta = checkpoint["nonce_metadata"]
        state_dict: dict[str, torch.Tensor] = {}
        try:
            if "payload_auth" in checkpoint:
                _verify_payload_auth(dek, checkpoint)
            else:
                logger.warning(
                    "loading legacy unauthenticated encrypted weights: %s; "
                    "rotate KEK or re-encrypt to upgrade",
                    path,
                )
            for name, enc_tensor in checkpoint["encrypted_state_dict"].items():
                dtype = getattr(torch, dtype_meta[name].replace("torch.", ""))
                nonce = bytes(nonce_meta[name])
                state_dict[name] = cipher.decrypt_tensor(enc_tensor, dtype, nonce=nonce)
        finally:
            cipher.key = b"\x00" * 32
            dek = b"\x00" * 32
        return state_dict

    def build_yolo(
        self,
        encrypted_path: str | Path,
        device: str = "cpu",
        meta_path: str | Path | None = None,
    ):
        from ultralytics import YOLO as _UltralyticsYOLO

        encrypted_path = Path(encrypted_path)
        checkpoint = self.read_encrypted_checkpoint(encrypted_path)
        yaml_cfg, names = self.get_model_metadata(checkpoint, meta_path)
        state_dict = self.decrypt_state_dict(encrypted_path, checkpoint=checkpoint)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(yaml_cfg, f, allow_unicode=True, sort_keys=False)
            yaml_path = f.name
        try:
            model = _UltralyticsYOLO(yaml_path)
            model.model.load_state_dict(state_dict, strict=True)
            model.model.names = names
            model.model = model.model.to(device)
            return model
        finally:
            os.unlink(yaml_path)

    def verify(self, encrypted_path: str | Path, original_path: str | Path) -> float:
        """返回解密权重相对明文的最大绝对误差。"""
        decrypted = self.decrypt_state_dict(encrypted_path)
        original = torch.load(original_path, map_location="cpu", weights_only=False)
        original_model = _extract_yolo_model(original)
        if original_model is None:
            raise ValueError(f"{original_path} 无法提取 model/ema")
        original_sd = original_model.state_dict()
        if set(decrypted) != set(original_sd):
            raise ValueError("解密权重与原始 state_dict 的 key 不一致")
        max_diff = 0.0
        for name in original_sd:
            diff = (decrypted[name].float() - original_sd[name].float()).abs().max().item()
            max_diff = max(max_diff, diff)
        return max_diff


def encrypt_weights(
    src_pt: str | Path,
    dst_pt: str | Path | None = None,
    key_dir: str | Path | None = None,
    *,
    kek: bytes | str | None = None,
    kek_id: str | None = None,
    config: ModelCryptoConfig | None = None,
    skip_existing: bool = False,
) -> Path | None:
    """明文检查点 → 信封加密文件；每个权重文件使用独立 DEK。"""
    cfg = _cfg(config)
    src_pt = Path(src_pt)
    if is_encrypted_checkpoint(src_pt, config=cfg):
        logger.info("跳过已加密权重: %s", src_pt)
        return src_pt

    dst_pt = Path(dst_pt) if dst_pt else cfg.encrypted_path_for(src_pt)
    if skip_existing and dst_pt.exists():
        logger.info("跳过已存在的加密权重: %s", dst_pt)
        return dst_pt

    kek_bytes, resolved_kid = load_kek(
        kek=kek, key_dir=key_dir, model_path=src_pt, config=cfg
    )
    if kek_id:
        resolved_kid = kek_id

    checkpoint = torch.load(src_pt, map_location="cpu", weights_only=False)
    model = _extract_yolo_model(checkpoint)
    if model is None:
        raise ValueError(f"{src_pt} 不是含 model/ema 的 YOLO 检查点")

    dek = secrets.token_bytes(32)
    cipher = _TensorCipher(dek)
    encrypted_sd: dict[str, torch.Tensor] = {}
    dtype_meta: dict[str, str] = {}
    nonce_meta: dict[str, bytes] = {}
    state_dict = model.state_dict()
    total = len(state_dict)

    try:
        for i, (name, tensor) in enumerate(state_dict.items(), start=1):
            dtype_meta[name] = str(tensor.dtype)
            encrypted_sd[name], nonce_meta[name] = cipher.encrypt_tensor(tensor)
            if i % 100 == 0 or i == total:
                logger.info("权重加密进度: %d/%d", i, total)

        envelope = wrap_dek(kek_bytes, dek, kek_id=resolved_kid)
        out = copy.deepcopy(checkpoint)
        out[cfg.encrypted_mark] = True
        out["crypto_version"] = max(int(cfg.crypto_version), CRYPTO_VERSION)
        out["envelope"] = envelope
        out["encrypted_state_dict"] = encrypted_sd
        out["dtype_metadata"] = dtype_meta
        out["nonce_metadata"] = nonce_meta
        out["model_yaml"] = model.yaml
        out["model_names"] = dict(model.names)
        out["payload_auth"] = {
            "alg": _PAYLOAD_AUTH_ALG,
            "tag": _payload_auth_tag(dek, out),
        }
        out.pop("model", None)
        out.pop("ema", None)

        _atomic_torch_save(out, dst_pt)
    finally:
        cipher.key = b"\x00" * 32
        dek = b"\x00" * 32
    logger.info("权重加密完成: %s -> %s (kek_id=%s)", src_pt.name, dst_pt, resolved_kid)
    return dst_pt


def rotate_kek(
    enc_path: str | Path,
    *,
    old_kek: bytes | str | None = None,
    new_kek: bytes | str | None = None,
    old_key_dir: str | Path | None = None,
    new_key_dir: str | Path | None = None,
    new_kek_id: str | None = None,
    dst_pt: str | Path | None = None,
    config: ModelCryptoConfig | None = None,
) -> Path:
    """更换 KEK：仅重封装 DEK，不重加密权重张量。"""
    cfg = _cfg(config)
    enc_path = Path(enc_path)
    checkpoint = torch.load(enc_path, map_location="cpu", weights_only=False)
    if not checkpoint.get(cfg.encrypted_mark) or "envelope" not in checkpoint:
        raise ValueError(f"{enc_path} 不是信封加密权重")

    old_bytes, _ = load_kek(
        kek=old_kek, key_dir=old_key_dir, model_path=enc_path, config=cfg
    )
    new_bytes, new_kid = load_kek(
        kek=new_kek, key_dir=new_key_dir, model_path=enc_path, config=cfg
    )
    if new_kek_id:
        new_kid = new_kek_id

    dek = unwrap_dek(old_bytes, checkpoint["envelope"])
    if "payload_auth" in checkpoint:
        _verify_payload_auth(dek, checkpoint)
    else:
        checkpoint["crypto_version"] = CRYPTO_VERSION
        checkpoint["payload_auth"] = {
            "alg": _PAYLOAD_AUTH_ALG,
            "tag": _payload_auth_tag(dek, checkpoint),
        }
    checkpoint["envelope"] = wrap_dek(new_bytes, dek, kek_id=new_kid)
    dek = b"\x00" * 32

    out = Path(dst_pt) if dst_pt else enc_path
    _atomic_torch_save(checkpoint, out)
    logger.info("KEK 轮换完成: %s -> %s (kek_id=%s)", enc_path, out, new_kid)
    return out


def discover_yolo_weights(
    root: str | Path,
    exclude_dirs: Iterable[str] | None = None,
    *,
    config: ModelCryptoConfig | None = None,
) -> list[Path]:
    cfg = _cfg(config)
    root = Path(root).resolve()
    excludes = set(exclude_dirs or cfg.exclude_dirs)
    results: list[Path] = []
    seen: set[Path] = set()
    for suffix in sorted(cfg.weight_suffixes):
        for path in sorted(root.rglob(f"*{suffix}")):
            if path in seen or path.suffix.lower() != suffix.lower():
                continue
            if any(part in excludes for part in path.parts):
                continue
            if path.stem.endswith(cfg.enc_suffix):
                continue
            if is_yolo_checkpoint(path, config=cfg):
                seen.add(path)
                results.append(path)
    return results


def encrypt_all_weights(
    root: str | Path = ".",
    key_dir: str | Path | None = None,
    exclude_dirs: Iterable[str] | None = None,
    *,
    kek: bytes | str | None = None,
    config: ModelCryptoConfig | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    cfg = _cfg(config)
    keys = key_dir or cfg.resolved_keys_dir() or (Path(root) / "keys")
    load_kek(kek=kek, key_dir=keys, config=cfg)

    files = discover_yolo_weights(root, exclude_dirs=exclude_dirs, config=cfg)
    logger.info("发现 %d 个待加密 YOLO 权重", len(files))
    encrypted: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for src in files:
        try:
            dst = encrypt_weights(
                src,
                key_dir=keys,
                kek=kek,
                config=cfg,
                skip_existing=skip_existing,
            )
            if dst:
                encrypted.append(dst)
        except Exception as exc:
            failed.append((src, str(exc)))
            logger.exception("权重加密失败: %s", src)

    logger.info("批量加密完成: 成功 %d，失败 %d", len(encrypted), len(failed))
    for src, err in failed:
        logger.error("权重加密失败汇总: %s: %s", src, err)
    return encrypted


def decrypt_to_pt(
    enc_path: str | Path,
    dst_pt: str | Path | None = None,
    key_dir: str | Path | None = None,
    *,
    kek: bytes | str | None = None,
    config: ModelCryptoConfig | None = None,
) -> Path:
    """离线导出明文检查点；在线推理应使用 load_yolo。"""
    cfg = _cfg(config)
    enc_path = Path(enc_path)
    if not is_encrypted_checkpoint(enc_path, config=cfg):
        raise ValueError(f"{enc_path} 不是信封加密权重")

    dst_pt = Path(dst_pt) if dst_pt else cfg.plain_path_for_enc(enc_path)
    dec = ModelDecryptor(key_dir, kek=kek, config=cfg)
    checkpoint = dec.read_encrypted_checkpoint(enc_path)
    state_dict = dec.decrypt_state_dict(enc_path, checkpoint=checkpoint)
    yaml_cfg, names = dec.get_model_metadata(checkpoint)

    from ultralytics import YOLO as _UltralyticsYOLO

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(yaml_cfg, f, allow_unicode=True, sort_keys=False)
        yaml_path = f.name
    try:
        yolo = _UltralyticsYOLO(yaml_path)
        yolo.model.load_state_dict(state_dict, strict=True)
        yolo.model.names = names
        restored = yolo.model
    finally:
        os.unlink(yaml_path)

    out = copy.deepcopy(checkpoint)
    for key in (
        cfg.encrypted_mark,
        "crypto_version",
        "envelope",
        "encrypted_state_dict",
        "dtype_metadata",
        "nonce_metadata",
        "model_yaml",
        "model_names",
        "payload_auth",
    ):
        out.pop(key, None)
    out["model"] = restored
    out.pop("ema", None)

    _atomic_torch_save(out, dst_pt)
    logger.info("已导出明文权重: %s -> %s", enc_path, dst_pt)
    return dst_pt


def restore_plain_weights_from_enc(
    enc_root: str | Path,
    dst_root: str | Path,
    key_dir: str | Path | None = None,
    *,
    kek: bytes | str | None = None,
    config: ModelCryptoConfig | None = None,
) -> list[Path]:
    cfg = _cfg(config)
    enc_root = Path(enc_root).resolve()
    dst_root = Path(dst_root).resolve()
    restored: list[Path] = []
    for enc_path in sorted(enc_root.rglob(f"*{cfg.enc_suffix}.pt")):
        rel = enc_path.relative_to(enc_root)
        dst_pt = dst_root / cfg.plain_path_for_enc(rel)
        restored.append(decrypt_to_pt(enc_path, dst_pt, key_dir, kek=kek, config=cfg))
    return restored


__all__ = [
    "ModelDecryptor",
    "decrypt_to_pt",
    "discover_yolo_weights",
    "encrypt_all_weights",
    "encrypt_weights",
    "init_kek",
    "is_encrypted_checkpoint",
    "is_yolo_checkpoint",
    "restore_plain_weights_from_enc",
    "rotate_kek",
]
