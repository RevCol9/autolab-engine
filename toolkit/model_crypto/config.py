"""权重加密可配置项；可按项目覆盖默认排除目录与命名约定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelCryptoConfig:
    """加密标记、文件后缀与扫描排除规则。"""

    encrypted_mark: str = "encrypted"
    enc_suffix: str = "_enc"
    key_file_name: str = "aes_key.key"
    nonce_file_name: str = "aes_nonce.nonce"
    default_exclude_dirs: frozenset[str] = field(
        default_factory=lambda: frozenset(
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
    )
    weight_suffixes: frozenset[str] = field(
        default_factory=lambda: frozenset({".pt", ".crypt", ".pth"})
    )

    def encrypted_path_for(self, src: Path) -> Path:
        if src.stem.endswith(self.enc_suffix):
            return src
        return src.with_name(f"{src.stem}{self.enc_suffix}{src.suffix}")

    def plain_path_for_enc(self, enc: Path) -> Path:
        if not enc.stem.endswith(self.enc_suffix):
            raise ValueError(f"不是加密后缀文件: {enc}")
        return enc.with_name(f"{enc.stem[: -len(self.enc_suffix)]}{enc.suffix}")


def default_model_crypto_config() -> ModelCryptoConfig:
    return ModelCryptoConfig()
