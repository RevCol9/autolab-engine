"""Cython 编译命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from toolkit.cython_build.builder import CythonBuilder
from toolkit.cython_build.config import CythonBuildConfig, PLATFORM_CHOICES
from toolkit.cython_build.sync import sync_develop, sync_develop_both


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cython 编译到 develop/ 并同步部署资源（可插拔配置）"
    )
    parser.add_argument("targets", nargs="*", help="要编译的 .py 或目录")
    parser.add_argument("--all", action="store_true", help="编译 production_targets")
    parser.add_argument(
        "--platform",
        choices=PLATFORM_CHOICES,
        default=None,
        help="部署平台目录（默认按当前系统自动选择）",
    )
    parser.add_argument("--sync", action="store_true", help="仅同步资源到 develop/<platform>/")
    parser.add_argument(
        "--sync-both",
        action="store_true",
        help="将资源同步到 develop/windows/ 和 develop/linux/（不编译）",
    )
    parser.add_argument("--no-sync", action="store_true", help="编译后不同步资源")
    parser.add_argument("--verify", action="store_true", help="编译后验证 import")
    parser.add_argument("--dry-run", action="store_true", help="预览将要编译的文件")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="项目根目录（默认 autolab-engine 根）",
    )
    args = parser.parse_args(argv)

    config = CythonBuildConfig.autolab_engine(args.project_root)
    builder = CythonBuilder(config)

    if args.sync_both:
        sync_develop_both(config)
        print("\n完成。develop/windows/ 与 develop/linux/ 资源已对齐。")
        print("注意: .pyd/.so 仍需在各自系统上分别编译。")
        return

    platform = builder.resolve_platform(args.platform)

    if args.sync:
        sync_develop(config, platform)
        entry = config.entry_scripts[0] if config.entry_scripts else "run.py"
        print(f"\n完成。复制 develop/{platform}/ 到目标机器后运行: python {entry}")
        return

    if args.all:
        raw_targets = list(config.resolve_targets(use_all=True))
    elif args.targets:
        raw_targets = args.targets
    else:
        raw_targets = list(config.resolve_targets())

    if not raw_targets:
        print("请指定编译目标，例如:")
        print("  python -m toolkit.cython_build annotation/")
        print("  python -m toolkit.cython_build --all --platform linux")
        sys.exit(1)

    py_files = builder.build(
        raw_targets,
        platform=platform,
        verify=args.verify,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        sys.exit(0)

    if not args.no_sync:
        sync_develop(config, platform)

    entry = config.entry_scripts[0] if config.entry_scripts else "run.py"
    print(f"\n完成。复制 develop/{platform}/ 到目标机器部署。")
    print(f"启动: python {entry}")
    if platform == "windows":
        print("Linux 包需在 Linux 上再执行: python -m toolkit.cython_build --all --platform linux")

    return py_files


if __name__ == "__main__":
    main()
