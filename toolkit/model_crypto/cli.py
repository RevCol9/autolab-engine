"""Runtime key initialization CLI for `.niii-model`."""

from __future__ import annotations

import argparse
import logging

from toolkit.model_crypto.config import default_model_crypto_config
from toolkit.model_crypto.envelope import init_kek


def _keys_default() -> str:
    cfg = default_model_crypto_config()
    if cfg.keys_dir:
        return cfg.keys_dir
    return "keys"


def _cmd_init_kek(args: argparse.Namespace) -> None:
    path, kid = init_kek(args.keys, kek_id=args.kek_id)
    print(f"KEK: {path} (kek_id={kid})")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    keys_default = _keys_default()
    parser = argparse.ArgumentParser(description="`.niii-model` 运行时密钥工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-kek", help="生成 KEK")
    p_init.add_argument("--keys", default=keys_default)
    p_init.add_argument("--kek-id", default=None)
    p_init.set_defaults(func=_cmd_init_kek)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
