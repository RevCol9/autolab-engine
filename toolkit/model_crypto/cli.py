"""权重信封加解密 CLI。"""

from __future__ import annotations

import argparse
import logging

from toolkit.model_crypto.config import default_model_crypto_config
from toolkit.model_crypto.core import (
    ModelDecryptor,
    decrypt_to_pt,
    encrypt_all_weights,
    encrypt_weights,
    init_kek,
    restore_plain_weights_from_enc,
    rotate_kek,
)
from toolkit.model_crypto.loader import load_yolo


def _keys_default() -> str:
    cfg = default_model_crypto_config()
    if cfg.keys_dir:
        return cfg.keys_dir
    return "keys"


def _cmd_init_kek(args: argparse.Namespace) -> None:
    path, kid = init_kek(args.keys, kek_id=args.kek_id, overwrite=args.overwrite)
    print(f"KEK: {path} (kek_id={kid})")


def _cmd_encrypt(args: argparse.Namespace) -> None:
    encrypt_weights(args.src, args.dst, args.keys, kek_id=args.kek_id)


def _cmd_encrypt_all(args: argparse.Namespace) -> None:
    encrypt_all_weights(
        root=args.root,
        key_dir=args.keys,
        exclude_dirs=args.exclude or None,
        skip_existing=not args.force,
    )


def _cmd_predict(args: argparse.Namespace) -> None:
    model = load_yolo(args.src, key_dir=args.keys, meta_path=args.meta, device=args.device)
    print(f"类别: {model.model.names}")
    results = model.predict(
        source=args.image,
        conf=args.conf,
        save=args.save,
        project=args.project,
        name=args.name,
        device=args.device,
    )
    for i, result in enumerate(results):
        n = 0 if result.boxes is None else len(result.boxes)
        print(f"[{i}] 检测数: {n}")


def _cmd_verify(args: argparse.Namespace) -> None:
    diff = ModelDecryptor(args.keys).verify(args.src, args.original)
    print(f"max_abs_err={diff:.8f}")


def _cmd_decrypt(args: argparse.Namespace) -> None:
    decrypt_to_pt(args.src, args.dst, args.keys)


def _cmd_restore_all(args: argparse.Namespace) -> None:
    files = restore_plain_weights_from_enc(args.enc_root, args.dst_root, args.keys)
    print(f"恢复 {len(files)} 个明文权重")


def _cmd_rotate_kek(args: argparse.Namespace) -> None:
    rotate_kek(
        args.src,
        old_key_dir=args.old_keys,
        new_key_dir=args.new_keys,
        new_kek_id=args.kek_id,
        dst_pt=args.dst,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    keys_default = _keys_default()
    parser = argparse.ArgumentParser(description="YOLO 权重信封加解密")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-kek", help="生成 KEK")
    p_init.add_argument("--keys", default=keys_default)
    p_init.add_argument("--kek-id", default=None)
    p_init.add_argument("--overwrite", action="store_true")
    p_init.set_defaults(func=_cmd_init_kek)

    p_enc = sub.add_parser("encrypt", help="加密单个权重")
    p_enc.add_argument("--src", required=True)
    p_enc.add_argument("--dst", default=None)
    p_enc.add_argument("--keys", default=keys_default)
    p_enc.add_argument("--kek-id", default=None)
    p_enc.set_defaults(func=_cmd_encrypt)

    p_all = sub.add_parser("encrypt-all", help="批量加密")
    p_all.add_argument("--root", default=".")
    p_all.add_argument("--keys", default=keys_default)
    p_all.add_argument("--exclude", nargs="*", default=None)
    p_all.add_argument("--force", action="store_true")
    p_all.set_defaults(func=_cmd_encrypt_all)

    p_pred = sub.add_parser("predict", help="内存解密推理")
    p_pred.add_argument("--src", required=True)
    p_pred.add_argument("--keys", default=keys_default)
    p_pred.add_argument("--image", required=True)
    p_pred.add_argument("--meta", default=None)
    p_pred.add_argument("--device", default="cpu")
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--save", action="store_true")
    p_pred.add_argument("--project", default="runs/predict")
    p_pred.add_argument("--name", default="exp")
    p_pred.set_defaults(func=_cmd_predict)

    p_ver = sub.add_parser("verify", help="校验加解密误差")
    p_ver.add_argument("--src", required=True)
    p_ver.add_argument("--keys", default=keys_default)
    p_ver.add_argument("--original", required=True)
    p_ver.set_defaults(func=_cmd_verify)

    p_dec = sub.add_parser("decrypt", help="导出明文权重（离线）")
    p_dec.add_argument("--src", required=True)
    p_dec.add_argument("--dst", default=None)
    p_dec.add_argument("--keys", default=keys_default)
    p_dec.set_defaults(func=_cmd_decrypt)

    p_restore = sub.add_parser("restore-all", help="批量导出明文（离线）")
    p_restore.add_argument("--enc-root", required=True)
    p_restore.add_argument("--dst-root", default=".")
    p_restore.add_argument("--keys", default=keys_default)
    p_restore.set_defaults(func=_cmd_restore_all)

    p_rot = sub.add_parser("rotate-kek", help="更换 KEK（重封装 DEK）")
    p_rot.add_argument("--src", required=True)
    p_rot.add_argument("--old-keys", required=True)
    p_rot.add_argument("--new-keys", required=True)
    p_rot.add_argument("--dst", default=None)
    p_rot.add_argument("--kek-id", default=None)
    p_rot.set_defaults(func=_cmd_rotate_kek)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
