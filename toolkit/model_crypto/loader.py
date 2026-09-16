"""权重路径解析与 YOLO 加载入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolkit.model_crypto.config import ModelCryptoConfig, default_model_crypto_config
from toolkit.model_crypto.core import ModelDecryptor, is_encrypted_checkpoint


def resolve_weight_path(
    path: str | Path,
    *,
    config: ModelCryptoConfig | None = None,
) -> Path:
    """明文路径不存在时回退到同名 *_enc.pt。"""
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
    kek: bytes | str | None = None,
    config: ModelCryptoConfig | None = None,
    **kwargs: Any,
):
    """加载 YOLO：加密权重在内存解密后构造模型，明文路径走 Ultralytics 原逻辑。"""
    cfg = config or default_model_crypto_config()
    model_path = resolve_weight_path(model, config=cfg)
    if is_encrypted_checkpoint(model_path, config=cfg):
        return ModelDecryptor(key_dir, kek=kek, config=cfg).build_yolo(
            model_path,
            device=device,
            meta_path=meta_path,
        )

    from ultralytics import YOLO

    return YOLO(str(model_path), **kwargs)
