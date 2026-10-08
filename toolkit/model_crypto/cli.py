"""`.niii-model` key initialization and trusted offline import CLI."""

from __future__ import annotations

import argparse
import logging

from toolkit.model_crypto.config import default_model_crypto_config
from toolkit.model_crypto.envelope import init_kek
from toolkit.model_crypto.offline_convert import convert_trusted_pt


def _keys_default() -> str:
    cfg = default_model_crypto_config()
    if cfg.keys_dir:
        return cfg.keys_dir
    return "keys"


def _cmd_init_kek(args: argparse.Namespace) -> None:
    path, kid = init_kek(args.keys, kek_id=args.kek_id, overwrite=args.overwrite)
    print(f"KEK: {path} (kek_id={kid})")


def _cmd_pack_v1(args: argparse.Namespace) -> None:
    if not args.trust_source_pt:
        raise SystemExit("pack-v1 只能在隔离构建机处理可信 .pt；请显式指定 --trust-source-pt")
    result = convert_trusted_pt(
        args.src, args.dst, task=args.task, key_dir=args.keys,
    )
    print(f"已生成并验证加密模型: {result}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    keys_default = _keys_default()
    parser = argparse.ArgumentParser(description="`.niii-model` 离线导入工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-kek", help="生成 KEK")
    p_init.add_argument("--keys", default=keys_default)
    p_init.add_argument("--kek-id", default=None)
    p_init.add_argument("--overwrite", action="store_true")
    p_init.set_defaults(func=_cmd_init_kek)

    p_pack = sub.add_parser("pack-v1", help="隔离构建机：可信 .pt 转换为 .niii-model")
    p_pack.add_argument("--src", required=True)
    p_pack.add_argument("--dst", required=True)
    p_pack.add_argument("--task", choices=("detect", "segment"), required=True)
    p_pack.add_argument("--keys", default=keys_default)
    p_pack.add_argument("--trust-source-pt", action="store_true")
    p_pack.set_defaults(func=_cmd_pack_v1)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
