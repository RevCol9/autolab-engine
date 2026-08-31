"""权重加载入口：明文与加密路径统一解析，供推理/训练嵌入。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolkit.model_crypto.config import ModelCryptoConfig, default_model_crypto_config
from toolkit.model_crypto.core import (
    ModelDecryptor,
    _resolve_key_dir,
    is_encrypted_checkpoint,
)


def resolve_weight_path(
    path: str | Path,
    *,
    config: ModelCryptoConfig | None = None,
) -> Path:
    """解析权重路径：开发用明文 .pt，部署仅有 *_enc.pt 时自动选用。"""
    cfg = config or default_model_crypto_config()
    p = Path(path)
    if p.is_file():
        return p.resolve()

    if p.stem.endswith(cfg.enc_suffix):
        plain = cfg.plain_path_for_enc(p)
        if plain.is_file():
            return plain.resolve()
        return p.resolve()

    enc = cfg.encrypted_path_for(p)
    if enc.is_file():
        return enc.resolve()
    return p.resolve()


def load_yolo(
    model: str | Path,
    key_dir: str | Path | None = None,
    device: str = "cpu",
    meta_path: str | Path | None = None,
    *,
    config: ModelCryptoConfig | None = None,
    **kwargs: Any,
):
    """
    统一加载 YOLO：自动识别加密权重并在内存中解密，不落盘明文。

    嵌入推理引擎示例::

        from toolkit.model_crypto import load_yolo
        yolo = load_yolo(model_path, key_dir="/etc/niii/keys", device="cuda:0")
    """
    cfg = config or default_model_crypto_config()
    model_path = resolve_weight_path(model, config=cfg)
    if is_encrypted_checkpoint(model_path, config=cfg):
        resolved_keys = _resolve_key_dir(model_path, key_dir, cfg)
        return ModelDecryptor(resolved_keys, config=cfg).build_yolo(
            model_path,
            device=device,
            meta_path=meta_path,
        )

    from ultralytics import YOLO

    return YOLO(str(model_path), **kwargs)
