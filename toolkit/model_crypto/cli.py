"""Runtime key initialization CLI for `.niii-model`."""

from __future__ import annotations

import argparse
import logging

from toolkit.model_crypto.config import default_model_crypto_config
from toolkit.model_crypto.envelope import (
    init_kek,
    keyring_status,
    rotate_kek,
    set_active_kek,
)


def _keys_default() -> str:
    cfg = default_model_crypto_config()
    if cfg.keys_dir:
        return cfg.keys_dir
    return "keys"


def _cmd_init_kek(args: argparse.Namespace) -> None:
    path, kid = init_kek(args.keys, kek_id=args.kek_id)
    print(f"KEK: {path} (kek_id={kid})")


def _cmd_rotate_kek(args: argparse.Namespace) -> None:
    path, kid = rotate_kek(args.keys, kek_id=args.kek_id)
    print(f"active KEK: {path} (kek_id={kid})")


def _cmd_set_active_kek(args: argparse.Namespace) -> None:
    kid = set_active_kek(args.keys, args.kek_id)
    print(f"active kek_id={kid}")


def _cmd_keyring_status(args: argparse.Namespace) -> None:
    active_id, key_ids = keyring_status(args.keys)
    for key_id in key_ids:
        marker = "*" if key_id == active_id else " "
        print(f"{marker} {key_id}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    keys_default = _keys_default()
    parser = argparse.ArgumentParser(description="`.niii-model` 运行时密钥工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-kek", help="生成 KEK")
    p_init.add_argument("--keys", default=keys_default)
    p_init.add_argument("--kek-id", default=None)
    p_init.set_defaults(func=_cmd_init_kek)

    p_rotate = sub.add_parser("rotate-kek", help="生成新 KEK 并原子切换 active key")
    p_rotate.add_argument("--keys", default=keys_default)
    p_rotate.add_argument("--kek-id", required=True)
    p_rotate.set_defaults(func=_cmd_rotate_kek)

    p_active = sub.add_parser("set-active-kek", help="回滚到已有 KEK")
    p_active.add_argument("--keys", default=keys_default)
    p_active.add_argument("--kek-id", required=True)
    p_active.set_defaults(func=_cmd_set_active_kek)

    p_status = sub.add_parser("keyring-status", help="列出 active 和历史 KEK ID")
    p_status.add_argument("--keys", default=keys_default)
    p_status.set_defaults(func=_cmd_keyring_status)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
