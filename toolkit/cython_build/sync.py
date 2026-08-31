"""develop 部署目录资源同步。"""

from __future__ import annotations

import shutil
from pathlib import Path

from toolkit.cython_build.builder import CythonBuilder
from toolkit.cython_build.config import CythonBuildConfig, PLATFORM_CHOICES


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def sync_develop(config: CythonBuildConfig, platform: str) -> None:
    """同步部署资源到 develop/<platform>/（明文入口脚本 + 配置/权重）。"""
    root = config.project_root.resolve()
    out_root = config.develop_root(platform)
    out_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []

    for name in config.entry_scripts:
        src = root / name
        if src.is_file():
            _copy_file(src, out_root / name)
            copied.append(name)

    for rel in config.config_files:
        src = root / rel
        if src.is_file():
            _copy_file(src, out_root / rel)
            copied.append(rel)

    for dirname in config.asset_dirs:
        src = root / dirname
        if src.is_dir():
            _copy_tree(src, out_root / dirname)
            copied.append(f"{dirname}/")

    for rel in config.deploy_weight_paths:
        src = root / rel
        if src.is_file():
            _copy_file(src, out_root / rel)
            copied.append(rel)
        else:
            missing.append(rel)

    if config.include_init_on_sync:
        builder = CythonBuilder(config)
        for init_py in root.rglob("__init__.py"):
            if builder._should_skip_path(init_py):
                continue
            rel = init_py.relative_to(root)
            _copy_file(init_py, out_root / rel)
            copied.append(str(rel))

    print(f"\n已同步 develop/{platform}/ ({out_root}):\n")
    for item in copied:
        print(f"  - {item}")
    if missing:
        print(f"\n  缺少权重/资源（按需准备后再 --sync）: {len(missing)} 个")
        for item in missing:
            print(f"      {item}")


def sync_develop_both(config: CythonBuildConfig) -> None:
    for platform in PLATFORM_CHOICES:
        sync_develop(config, platform)
