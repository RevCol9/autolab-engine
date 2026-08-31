"""Cython 编译配置；支持项目级覆盖与 autolab-engine 预设。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "test",
        "tests",
        "develop",
        "release",
        "dist",
        "build",
        "toolkit",
    }
)

PLATFORM_CHOICES = ("windows", "linux")


@dataclass
class CythonBuildConfig:
    """Cython 编译与 develop 同步的可插拔配置。"""

    project_root: Path
    develop_dir: str = "develop"
    production_targets: tuple[str, ...] = ()
    compile_targets: tuple[str, ...] = ()
    entry_scripts: tuple[str, ...] = ()
    never_compile_extra: frozenset[str] = field(default_factory=frozenset)
    config_files: tuple[str, ...] = ()
    asset_dirs: tuple[str, ...] = ()
    deploy_weight_paths: tuple[str, ...] = ()
    skip_dirs: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_DIRS)
    verify_skip_suffixes: tuple[str, ...] = ("__init__",)
    include_init_on_sync: bool = True

    @property
    def develop_base(self) -> Path:
        return self.project_root / self.develop_dir

    def develop_root(self, platform: str) -> Path:
        return self.develop_base / platform

    @property
    def never_compile(self) -> frozenset[str]:
        return frozenset(self.entry_scripts) | self.never_compile_extra

    @classmethod
    def autolab_engine(cls, project_root: Path | None = None) -> CythonBuildConfig:
        """autolab-engine 默认编译目标（可按需改 production_targets）。"""
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        return cls(
            project_root=root,
            production_targets=(
                "annotation/",
                "training/",
                "shared/",
            ),
            entry_scripts=(
                "run.py",
                "training/run.py",
            ),
            never_compile_extra=frozenset(),
            config_files=(
                "config/annotation.example.yaml",
                "config/training/base.example.yaml",
                "config/training/detection.example.yaml",
                "config/training/segmentation.example.yaml",
            ),
            asset_dirs=(),
            deploy_weight_paths=(),
        )

    def resolve_targets(
        self,
        *,
        use_all: bool = False,
        cli_targets: list[str] | None = None,
    ) -> tuple[str, ...]:
        if use_all:
            return self.production_targets
        if cli_targets:
            return tuple(cli_targets)
        if self.compile_targets:
            return self.compile_targets
        return ()


def default_cython_build_config(project_root: Path | None = None) -> CythonBuildConfig:
    return CythonBuildConfig.autolab_engine(project_root)
