"""YOLO 权重加解密核心实现。"""

from __future__ import annotations

import copy
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

from toolkit.model_crypto.config import ModelCryptoConfig, default_model_crypto_config


def _cfg(config: ModelCryptoConfig | None) -> ModelCryptoConfig:
    return config or default_model_crypto_config()


def _peek_checkpoint(path: Path, config: ModelCryptoConfig) -> dict[str, Any] | None:
    if not path.exists() or path.suffix not in config.weight_suffixes:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def is_encrypted_checkpoint(
    path: str | Path,
    *,
    config: ModelCryptoConfig | None = None,
) -> bool:
    cfg = _cfg(config)
    ckpt = _peek_checkpoint(Path(path), cfg)
    return bool(ckpt and ckpt.get(cfg.encrypted_mark) and "encrypted_state_dict" in ckpt)


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


def _resolve_key_dir(
    model_path: Path,
    key_dir: str | Path | None,
    config: ModelCryptoConfig,
) -> Path:
    if key_dir is not None:
        return Path(key_dir)
    for candidate in (Path.cwd() / "keys", model_path.parent / "keys"):
        key_file = candidate / config.key_file_name
        nonce_file = candidate / config.nonce_file_name
        if key_file.is_file() and nonce_file.is_file():
            return candidate
    raise FileNotFoundError(
        "找不到密钥目录。请将密钥放在项目根 keys/，"
        "或调用 load_yolo(..., key_dir='your/keys') 显式指定。"
    )


class _AESEncryptor:
    def __init__(
        self,
        key_dir: str | Path,
        config: ModelCryptoConfig,
        *,
        reuse_existing: bool = True,
    ) -> None:
        self.key_dir = Path(key_dir)
        self.config = config
        self.key_dir.mkdir(parents=True, exist_ok=True)
        key_file = self.key_dir / config.key_file_name
        nonce_file = self.key_dir / config.nonce_file_name
        if reuse_existing and key_file.is_file() and nonce_file.is_file():
            self.key = key_file.read_bytes()
            self.nonce = nonce_file.read_bytes()
            return
        self.key = secrets.token_bytes(32)
        self.nonce = secrets.token_bytes(16)
        key_file.write_bytes(self.key)
        nonce_file.write_bytes(self.nonce)

    def encrypt_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        cipher = Cipher(algorithms.AES(self.key), modes.CTR(self.nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        arr = tensor.detach().cpu().float().numpy()
        encrypted = encryptor.update(arr.tobytes()) + encryptor.finalize()
        out = np.frombuffer(encrypted, dtype=np.float32)
        return torch.from_numpy(out.copy()).view(tensor.shape)


class _AESDecryptor:
    def __init__(self, key_dir: str | Path, config: ModelCryptoConfig) -> None:
        key_dir = Path(key_dir)
        self.key = (key_dir / config.key_file_name).read_bytes()
        self.nonce = (key_dir / config.nonce_file_name).read_bytes()

    def decrypt_tensor(self, encrypted: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        cipher = Cipher(algorithms.AES(self.key), modes.CTR(self.nonce), backend=default_backend())
        decryptor = cipher.decryptor()
        arr = encrypted.detach().cpu().float().numpy()
        decrypted = decryptor.update(arr.tobytes()) + decryptor.finalize()
        out = np.frombuffer(decrypted, dtype=np.float32)
        return torch.from_numpy(out.copy()).view(encrypted.shape).to(dtype)


class ModelDecryptor:
    """解密器：可单独注入训练/推理服务。"""

    def __init__(
        self,
        key_dir: str | Path,
        *,
        config: ModelCryptoConfig | None = None,
    ) -> None:
        self.config = _cfg(config)
        self.key_dir = Path(key_dir)
        self._decryptor = _AESDecryptor(self.key_dir, self.config)

    def read_encrypted_checkpoint(self, encrypted_path: str | Path) -> dict[str, Any]:
        encrypted_path = Path(encrypted_path)
        checkpoint = torch.load(encrypted_path, map_location="cpu", weights_only=False)
        if not checkpoint.get(self.config.encrypted_mark):
            raise ValueError(f"{encrypted_path} 不是加密权重文件。")
        if "encrypted_state_dict" not in checkpoint:
            raise ValueError(f"{encrypted_path} 缺少 encrypted_state_dict。")
        return checkpoint

    def get_model_metadata(
        self,
        checkpoint: dict[str, Any],
        meta_path: str | Path | None = None,
    ) -> tuple[dict, dict]:
        if "model_yaml" in checkpoint and "model_names" in checkpoint:
            return checkpoint["model_yaml"], checkpoint["model_names"]
        if not meta_path:
            raise ValueError("加密文件缺少 model_yaml/model_names，请提供 meta_path。")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        model = _extract_yolo_model(meta)
        if model is None:
            raise ValueError(f"{meta_path} 无法提取 model/ema 元数据。")
        return model.yaml, dict(model.names)

    def decrypt_state_dict(
        self,
        encrypted_path: str | Path | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        checkpoint = checkpoint or self.read_encrypted_checkpoint(encrypted_path)
        state_dict: dict[str, torch.Tensor] = {}
        for name, enc_tensor in checkpoint["encrypted_state_dict"].items():
            dtype = getattr(torch, checkpoint["dtype_metadata"][name].replace("torch.", ""))
            state_dict[name] = self._decryptor.decrypt_tensor(enc_tensor, dtype)
        return state_dict

    def build_yolo(
        self,
        encrypted_path: str | Path,
        device: str = "cpu",
        meta_path: str | Path | None = None,
    ):
        from ultralytics import YOLO as _UltralyticsYOLO

        checkpoint = self.read_encrypted_checkpoint(encrypted_path)
        yaml_cfg, names = self.get_model_metadata(checkpoint, meta_path)
        state_dict = self.decrypt_state_dict(checkpoint=checkpoint)

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
        decrypted = self.decrypt_state_dict(encrypted_path)
        original = torch.load(original_path, map_location="cpu", weights_only=False)
        original_model = _extract_yolo_model(original)
        if original_model is None:
            raise ValueError(f"{original_path} 无法提取原始 model/ema。")
        original_sd = original_model.state_dict()
        if set(decrypted) != set(original_sd):
            raise ValueError("解密后权重 key 与原始模型不一致。")
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
    config: ModelCryptoConfig | None = None,
    reuse_keys: bool = True,
    skip_existing: bool = False,
) -> Path | None:
    cfg = _cfg(config)
    src_pt = Path(src_pt)
    if is_encrypted_checkpoint(src_pt, config=cfg):
        print(f"跳过（已是加密文件）: {src_pt}")
        return src_pt

    dst_pt = Path(dst_pt) if dst_pt else cfg.encrypted_path_for(src_pt)
    if key_dir is None:
        key_dir = Path.cwd() / "keys"
    if skip_existing and dst_pt.exists():
        print(f"跳过（已存在）: {dst_pt}")
        return dst_pt

    checkpoint = torch.load(src_pt, map_location="cpu", weights_only=False)
    model = _extract_yolo_model(checkpoint)
    if model is None:
        raise ValueError(f"{src_pt} 不是包含 model/ema state_dict 的 YOLO 检查点。")
    encryptor = _AESEncryptor(key_dir, cfg, reuse_existing=reuse_keys)

    encrypted_sd: dict[str, torch.Tensor] = {}
    dtype_meta: dict[str, str] = {}
    state_dict = model.state_dict()
    total = len(state_dict)

    for i, (name, tensor) in enumerate(state_dict.items(), start=1):
        dtype_meta[name] = str(tensor.dtype)
        encrypted_sd[name] = encryptor.encrypt_tensor(tensor)
        if i % 100 == 0 or i == total:
            print(f"  加密进度: {i}/{total}")

    out = copy.deepcopy(checkpoint)
    out[cfg.encrypted_mark] = True
    out["encrypted_state_dict"] = encrypted_sd
    out["dtype_metadata"] = dtype_meta
    out["model_yaml"] = model.yaml
    out["model_names"] = dict(model.names)
    out.pop("model", None)

    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_pt)
    print(f"加密完成: {src_pt.name} -> {dst_pt}")
    return dst_pt


def discover_yolo_weights(
    root: str | Path,
    exclude_dirs: Iterable[str] | None = None,
    *,
    config: ModelCryptoConfig | None = None,
) -> list[Path]:
    cfg = _cfg(config)
    root = Path(root).resolve()
    excludes = set(exclude_dirs or cfg.default_exclude_dirs)
    results: list[Path] = []
    for path in sorted(root.rglob("*.pt")):
        if any(part in excludes for part in path.parts):
            continue
        if path.stem.endswith(cfg.enc_suffix):
            continue
        if is_yolo_checkpoint(path, config=cfg):
            results.append(path)
    return results


def encrypt_all_weights(
    root: str | Path = ".",
    key_dir: str | Path | None = None,
    exclude_dirs: Iterable[str] | None = None,
    *,
    config: ModelCryptoConfig | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    cfg = _cfg(config)
    key_dir = Path(key_dir or Path(root) / "keys")
    key_dir.mkdir(parents=True, exist_ok=True)
    key_file = key_dir / cfg.key_file_name
    _AESEncryptor(key_dir, cfg, reuse_existing=key_file.exists())

    files = discover_yolo_weights(root, exclude_dirs=exclude_dirs, config=cfg)
    print(f"发现 {len(files)} 个 YOLO 权重待加密，密钥目录: {key_dir.resolve()}")
    encrypted: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for src in files:
        try:
            dst = encrypt_weights(
                src,
                key_dir=key_dir,
                config=cfg,
                reuse_keys=True,
                skip_existing=skip_existing,
            )
            if dst:
                encrypted.append(dst)
        except Exception as exc:
            failed.append((src, str(exc)))
            print(f"失败: {src} -> {exc}")

    print(f"\n完成: 成功 {len(encrypted)}，失败 {len(failed)}")
    if failed:
        for src, err in failed:
            print(f"  - {src}: {err}")
    return encrypted


def decrypt_to_pt(
    enc_path: str | Path,
    dst_pt: str | Path | None = None,
    key_dir: str | Path | None = None,
    *,
    config: ModelCryptoConfig | None = None,
) -> Path:
    cfg = _cfg(config)
    enc_path = Path(enc_path)
    if not is_encrypted_checkpoint(enc_path, config=cfg):
        raise ValueError(f"{enc_path} 不是加密权重文件。")

    if dst_pt is None:
        dst_pt = cfg.plain_path_for_enc(enc_path)
    else:
        dst_pt = Path(dst_pt)

    resolved_keys = _resolve_key_dir(enc_path, key_dir, cfg)
    dec = ModelDecryptor(resolved_keys, config=cfg)
    checkpoint = dec.read_encrypted_checkpoint(enc_path)
    state_dict = dec.decrypt_state_dict(checkpoint=checkpoint)
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
    for key in (cfg.encrypted_mark, "encrypted_state_dict", "dtype_metadata", "model_yaml", "model_names"):
        out.pop(key, None)

    if "model" not in out and "ema" in out:
        out["ema"] = restored
    else:
        out["model"] = restored

    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_pt)
    print(f"解密恢复: {enc_path} -> {dst_pt}")
    return dst_pt


def restore_plain_weights_from_enc(
    enc_root: str | Path,
    dst_root: str | Path,
    key_dir: str | Path | None = None,
    *,
    config: ModelCryptoConfig | None = None,
) -> list[Path]:
    cfg = _cfg(config)
    enc_root = Path(enc_root).resolve()
    dst_root = Path(dst_root).resolve()
    restored: list[Path] = []
    for enc_path in sorted(enc_root.rglob(f"*{cfg.enc_suffix}.pt")):
        rel = enc_path.relative_to(enc_root)
        dst_pt = dst_root / cfg.plain_path_for_enc(rel)
        restored.append(decrypt_to_pt(enc_path, dst_pt, key_dir, config=cfg))
    return restored
