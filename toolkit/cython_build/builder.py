"""Cython 编译核心：收集目标、编译、验证。"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

from toolkit.cython_build.config import CythonBuildConfig, PLATFORM_CHOICES


class CythonBuilder:
    """可嵌入的 Cython 构建器；通过 ``CythonBuildConfig`` 注入项目规则。"""

    def __init__(self, config: CythonBuildConfig) -> None:
        self.config = config
        self.project_root = config.project_root.resolve()

    def detect_platform(self) -> str:
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform.startswith("linux"):
            return "linux"
        raise SystemExit(
            f"当前系统 {sys.platform} 不在支持列表内，请使用 --platform windows 或 --platform linux，"
            "且需在对应操作系统上执行编译。"
        )

    def resolve_platform(self, name: str | None) -> str:
        platform = name or self.detect_platform()
        if platform not in PLATFORM_CHOICES:
            raise SystemExit(f"--platform 只能是: {', '.join(PLATFORM_CHOICES)}")
        current = self.detect_platform()
        if platform != current:
            print(
                f"警告: 请求平台 {platform}，但当前运行在 {current}。"
                f" Cython 无法交叉编译，实际只能在 {current} 上生成 {current} 的二进制。"
            )
        return platform

    def check_dependencies(self) -> None:
        try:
            import Cython  # noqa: F401
        except ImportError:
            print("未安装 Cython，请先执行: pip install cython setuptools")
            sys.exit(1)

    def _is_py_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix == ".py"

    def _should_skip_path(self, path: Path) -> bool:
        return bool(set(path.parts) & self.config.skip_dirs)

    def _collect_py_files(
        self,
        target: Path,
        *,
        include_init: bool,
        recursive: bool,
    ) -> list[Path]:
        if not target.exists():
            raise FileNotFoundError(f"路径不存在: {target}")

        if self._is_py_file(target):
            if target.name in self.config.never_compile:
                return []
            if target.name == "__init__.py" and not include_init:
                return []
            return [target.relative_to(self.project_root)]

        if not target.is_dir():
            raise ValueError(f"不是 .py 文件或目录: {target}")

        result: list[Path] = []
        walker = target.rglob("*.py") if recursive else target.glob("*.py")
        for py_path in sorted(walker):
            if self._should_skip_path(py_path):
                continue
            if py_path.name in self.config.never_compile:
                continue
            if py_path.name == "__init__.py" and not include_init:
                continue
            result.append(py_path.relative_to(self.project_root))
        return result

    def collect_targets(
        self,
        raw_paths: list[str],
        *,
        include_init: bool = False,
        recursive: bool = True,
    ) -> list[Path]:
        seen: set[Path] = set()
        ordered: list[Path] = []

        for raw in raw_paths:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            target = (self.project_root / raw).resolve()
            if self.project_root not in target.parents and target != self.project_root:
                raise ValueError(f"路径必须在项目根目录内: {raw}")

            for rel in self._collect_py_files(
                target,
                include_init=include_init,
                recursive=recursive,
            ):
                if rel not in seen:
                    seen.add(rel)
                    ordered.append(rel)

        return ordered

    @staticmethod
    def py_to_module_name(rel_py: Path) -> str:
        return ".".join(rel_py.with_suffix("").parts)

    def cleanup_build_artifacts(self, py_files: list[Path]) -> None:
        removed: list[str] = []
        for rel_py in py_files:
            c_file = self.project_root / rel_py.with_suffix(".c")
            if c_file.is_file():
                c_file.unlink()
                removed.append(str(c_file.relative_to(self.project_root)))
        build_dir = self.project_root / "build"
        if build_dir.is_dir():
            shutil.rmtree(build_dir)
            removed.append("build/")
        if removed:
            print("已清理临时文件:\n")
            for item in removed:
                print(f"  - {item}")
            print()

    def run_build(self, py_files: list[Path], out_root: Path) -> None:
        from setuptools import Extension, setup
        from Cython.Build import cythonize

        out_root.mkdir(parents=True, exist_ok=True)
        os.chdir(self.project_root)

        extensions = [
            Extension(
                self.py_to_module_name(rel_py),
                [str(rel_py).replace("\\", "/")],
            )
            for rel_py in py_files
        ]

        print(f"\n开始编译 {len(extensions)} 个文件 -> {out_root}\n")
        for rel_py in py_files:
            print(f"  - {rel_py}")

        setup(
            script_args=["build_ext", f"--build-lib={out_root}"],
            ext_modules=cythonize(
                extensions,
                compiler_directives={
                    "language_level": "3",
                    "annotation_typing": False,
                },
                quiet=False,
            ),
        )

    @staticmethod
    def find_extension_files(rel_py: Path, root: Path) -> list[Path]:
        stem = rel_py.stem
        parent = root / rel_py.parent
        if not parent.is_dir():
            return []
        found: list[Path] = []
        for p in parent.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if name.startswith(stem + ".") and (name.endswith(".pyd") or ".so" in name):
                found.append(p.relative_to(root))
        return sorted(found)

    def verify_imports(self, py_files: list[Path], out_root: Path) -> bool:
        os.chdir(out_root)
        if str(out_root) not in sys.path:
            sys.path.insert(0, str(out_root))

        ok = True
        print(f"\n验证 develop import ({out_root.name}):\n")
        for rel_py in py_files:
            if rel_py.stem in self.config.verify_skip_suffixes:
                continue
            mod = self.py_to_module_name(rel_py)
            try:
                if mod in sys.modules:
                    del sys.modules[mod]
                m = importlib.import_module(mod)
                print(f"  OK  {mod}")
                print(f"      -> {getattr(m, '__file__', '?')}")
            except Exception as e:
                ok = False
                print(f"  FAIL {mod}: {e}")
        return ok

    def print_summary(self, py_files: list[Path], out_root: Path, platform: str) -> None:
        ok_count = 0
        ext_name = ".pyd" if platform == "windows" else ".so"
        print(f"\n编译结果（develop/{platform}/，共 {len(py_files)} 个）:\n")
        for rel_py in py_files:
            exts = self.find_extension_files(rel_py, out_root)
            if exts:
                for ext in exts:
                    print(f"  OK   {rel_py}  ->  develop/{platform}/{ext}")
                ok_count += 1
            else:
                print(f"  MISS {rel_py}  ->  (未找到 {ext_name})")
        print(f"\n成功: {ok_count}/{len(py_files)}")
        if ok_count < len(py_files):
            print("若有 MISS，请检查 C/C++ 编译器或重新运行 cython_build")

    def build(
        self,
        raw_targets: list[str],
        *,
        platform: str | None = None,
        verify: bool = False,
        dry_run: bool = False,
    ) -> list[Path]:
        platform = self.resolve_platform(platform)
        out_root = self.config.develop_root(platform)

        try:
            py_files = self.collect_targets(raw_targets)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: {e}")
            sys.exit(1)

        if not py_files:
            print("没有找到可编译的 .py 文件。")
            sys.exit(1)

        print(f"项目根目录:   {self.project_root}")
        print(f"部署目录:     {out_root}")
        print(f"目标平台:     {platform}")
        print(f"待编译文件:   {len(py_files)} 个")
        if self.config.entry_scripts:
            print(f"不编译脚本:   {', '.join(self.config.entry_scripts)}")

        if dry_run:
            print("\n[dry-run] 将编译:\n")
            for p in py_files:
                print(f"  {p}")
            print(f"\n输出目录: develop/{platform}/")
            return py_files

        self.check_dependencies()

        try:
            self.cleanup_build_artifacts(py_files)
            self.run_build(py_files, out_root)
            self.cleanup_build_artifacts(py_files)
        except Exception as e:
            err = str(e)
            print(f"\n编译失败: {e}")
            if "Cannot assign type" in err or "Cythonizing" in err or ".py:" in err:
                print("\n请根据上方行号修改对应 .py。")
            elif "error: Microsoft Visual C++" in err or "cl.exe" in err.lower():
                print("\nWindows 需安装 Visual Studio「使用 C++ 的桌面开发」。")
            elif "python3-dev" in err or "Python.h" in err:
                print("\nLinux 需: sudo apt install python3-dev build-essential")
            sys.exit(1)

        self.print_summary(py_files, out_root, platform)
        if verify and not self.verify_imports(py_files, out_root):
            sys.exit(1)
        return py_files

    def build_all(
        self,
        *,
        platform: str | None = None,
        verify: bool = False,
        dry_run: bool = False,
    ) -> list[Path]:
        targets = self.config.resolve_targets(use_all=True)
        if not targets:
            print("production_targets 为空，请在 CythonBuildConfig 中配置编译目标。")
            sys.exit(1)
        return self.build(list(targets), platform=platform, verify=verify, dry_run=dry_run)
