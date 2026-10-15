"""从已提交版本生成并检查生产运行时发布包。"""

from __future__ import annotations

import argparse
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_PATHS = (
    "README.md",
    "ARCHITECTURE.md",
    "run.py",
    "requirements.txt",
    "requirements-vlm.txt",
    "annotation",
    "api",
    "config/annotation.example.yaml",
    "config/model_crypto.example.yaml",
    "config/training",
    "deploy/niii-autolab.service.example",
    "deploy/niii-autolab-train.service.example",
    "shared",
    "toolkit/__init__.py",
    "toolkit/data_clean",
    "toolkit/model_crypto",
    "training",
)
FORBIDDEN_MODEL = re.compile(r"\.(?:pt|pth|ckpt|onnx|engine|safetensors)(?:\.|$)", re.I)
FORBIDDEN_NAMES = {".git", ".env", "keys", "keyring", "kek.key", "kek_id.txt"}


def check_archive(path: Path) -> int:
    """拒绝明文模型、密钥、Git 元数据和非运行时代码。"""
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            parts = name.parts
            if (
                name.is_absolute()
                or ".." in parts
                or (parts and parts[0] in {"scripts", "tests"})
                or any(
                    part.lower() in FORBIDDEN_NAMES or part.lower().startswith(".env.")
                    for part in parts
                )
                or any(FORBIDDEN_MODEL.search(part) for part in parts)
                or name.suffix.lower() == ".niii-model"
                or name.suffix.lower() in {".key", ".pem", ".p12", ".pfx"}
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"发布包包含禁止的条目: {member.name}")
            if member.isfile():
                count += 1
    if count == 0:
        raise ValueError("发布包为空")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="输出 .tar.gz 路径；默认写入 dist/")
    args = parser.parse_args()

    commit = subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output = args.output or ROOT / "dist" / f"autolab-engine-{commit}.tar.gz"
    output = output.resolve()
    if output.exists():
        parser.error(f"发布包已存在: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output.parent) as temp_dir:
        archive_path = Path(temp_dir) / "release.tar.gz"
        subprocess.run(
            [
                "git", "archive", "--format=tar.gz", f"--output={archive_path}",
                "HEAD", *RUNTIME_PATHS,
            ],
            cwd=ROOT,
            check=True,
        )
        count = check_archive(archive_path)
        archive_path.rename(output)
    print(f"发布包检查通过：{output}（{count} 个文件，commit {commit}）")


if __name__ == "__main__":
    main()
