"""发布包扫描门禁。"""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.build_release import check_archive


class ReleaseArchiveTests(unittest.TestCase):
    def _archive(self, root: Path, names: list[str]) -> Path:
        path = root / "release.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name in names:
                data = b"test"
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        return path

    def test_accepts_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = self._archive(Path(temp_dir), ["run.py", "toolkit/model_crypto/runtime.py"])
            self.assertEqual(check_archive(archive), 2)

    def test_rejects_historical_model_and_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for name in ("weights/best.pt", "weights/best.pt.tmp", ".git/objects/abc"):
                with self.subTest(name=name):
                    archive = self._archive(Path(temp_dir), ["run.py", name])
                    with self.assertRaisesRegex(ValueError, "禁止的条目"):
                        check_archive(archive)

    def test_rejects_key_material_and_encrypted_model_in_code_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for name in (
                "keys/kek.key",
                "config/.env.prod",
                "models/best.niii-model",
                "scripts/convert_trusted_model.py",
            ):
                with self.subTest(name=name):
                    archive = self._archive(Path(temp_dir), [name])
                    with self.assertRaisesRegex(ValueError, "禁止的条目"):
                        check_archive(archive)


if __name__ == "__main__":
    unittest.main()
