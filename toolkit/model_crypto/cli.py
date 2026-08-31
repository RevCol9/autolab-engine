"""权重加解密命令行入口。"""

from __future__ import annotations

import argparse

from toolkit.model_crypto.core import (
    ModelDecryptor,
    decrypt_to_pt,
    encrypt_all_weights,
    encrypt_weights,
    restore_plain_weights_from_enc,
)
from toolkit.model_crypto.loader import load_yolo


def _cmd_encrypt(args: argparse.Namespace) -> None:
    encrypt_weights(args.src, args.dst, args.keys, reuse_keys=not args.new_keys)


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
    print(f"校验通过，最大误差: {diff:.8f}")


def _cmd_decrypt(args: argparse.Namespace) -> None:
    decrypt_to_pt(args.src, args.dst, args.keys)


def _cmd_restore_all(args: argparse.Namespace) -> None:
    files = restore_plain_weights_from_enc(args.enc_root, args.dst_root, args.keys)
    print(f"\n完成: 恢复 {len(files)} 个明文 .pt")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YOLO 权重加解密工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enc = sub.add_parser("encrypt", help="加密单个权重")
    p_enc.add_argument("--src", required=True)
    p_enc.add_argument("--dst", default=None)
    p_enc.add_argument("--keys", default="keys")
    p_enc.add_argument("--new-keys", action="store_true", help="强制重新生成密钥")
    p_enc.set_defaults(func=_cmd_encrypt)

    p_all = sub.add_parser("encrypt-all", help="批量加密项目内所有 YOLO .pt")
    p_all.add_argument("--root", default=".", help="扫描根目录")
    p_all.add_argument("--keys", default="keys", help="全局密钥目录")
    p_all.add_argument("--exclude", nargs="*", default=None, help="排除目录名")
    p_all.add_argument("--force", action="store_true", help="覆盖已有 *_enc.pt")
    p_all.set_defaults(func=_cmd_encrypt_all)

    p_pred = sub.add_parser("predict", help="解密并推理")
    p_pred.add_argument("--src", required=True)
    p_pred.add_argument("--keys", default="keys")
    p_pred.add_argument("--image", required=True)
    p_pred.add_argument("--meta", default=None)
    p_pred.add_argument("--device", default="cpu")
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--save", action="store_true")
    p_pred.add_argument("--project", default="runs/predict")
    p_pred.add_argument("--name", default="exp")
    p_pred.set_defaults(func=_cmd_predict)

    p_ver = sub.add_parser("verify", help="校验解密")
    p_ver.add_argument("--src", required=True)
    p_ver.add_argument("--keys", default="keys")
    p_ver.add_argument("--original", required=True)
    p_ver.set_defaults(func=_cmd_verify)

    p_dec = sub.add_parser("decrypt", help="解密单个 *_enc.pt 为 .pt")
    p_dec.add_argument("--src", required=True)
    p_dec.add_argument("--dst", default=None)
    p_dec.add_argument("--keys", default="keys")
    p_dec.set_defaults(func=_cmd_decrypt)

    p_restore = sub.add_parser("restore-all", help="批量从加密目录恢复明文 .pt")
    p_restore.add_argument("--enc-root", required=True)
    p_restore.add_argument("--dst-root", default=".")
    p_restore.add_argument("--keys", default="keys")
    p_restore.set_defaults(func=_cmd_restore_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
