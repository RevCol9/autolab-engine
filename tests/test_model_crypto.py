from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import toolkit.model_crypto as model_crypto
from toolkit.model_crypto.config import ModelCryptoConfig
from toolkit.model_crypto.envelope import (
    init_kek,
    load_kek,
    parse_kek_material,
    resolve_keys_dir,
)


class ModelCryptoKeyTest(unittest.TestCase):
    def test_raw_32_byte_kek_is_not_stripped(self):
        raw = b" " + (b"x" * 30) + b"\n"
        self.assertEqual(raw, parse_kek_material(raw))

    def test_initialized_key_can_be_loaded_with_its_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_dir = Path(tmp) / "keys"
            key_path, key_id = init_kek(key_dir, kek_id="test-key")
            material, loaded_id = load_kek(key_dir=key_dir)

            self.assertEqual(key_path.read_bytes(), material)
            self.assertEqual("test-key", key_id)
            self.assertEqual(key_id, loaded_id)

            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                init_kek(key_dir, kek_id="replacement")
            self.assertEqual(material, key_path.read_bytes())

            if os.name != "nt":
                self.assertEqual(0o600, key_path.stat().st_mode & 0o777)

    def test_key_lookup_never_falls_back_to_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp)
            (working_dir / "keys").mkdir()
            with patch("pathlib.Path.cwd", return_value=working_dir):
                self.assertIsNone(resolve_keys_dir(config=ModelCryptoConfig()))

    @unittest.skipIf(os.name == "nt", "POSIX permissions are not available")
    def test_permissive_key_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_dir = Path(tmp) / "keys"
            key_path, _ = init_kek(key_dir)
            key_path.chmod(0o644)
            with self.assertRaisesRegex(PermissionError, "0600"):
                load_kek(key_dir=key_dir)

    def test_key_cli_import_does_not_load_ml_runtime(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import toolkit.model_crypto.cli; "
                "assert 'torch' not in sys.modules; "
                "assert 'ultralytics' not in sys.modules",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_runtime_cli_does_not_expose_trusted_pt_import(self):
        completed = subprocess.run(
            [sys.executable, "-m", "toolkit.model_crypto", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("init-kek", completed.stdout)
        self.assertIn("rotate-kek", completed.stdout)
        self.assertIn("set-active-kek", completed.stdout)
        self.assertIn("keyring-status", completed.stdout)
        self.assertNotIn("pack-v1", completed.stdout)

    def test_keyring_cli_rotates_lists_and_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / "keys"
            base = [sys.executable, "-m", "toolkit.model_crypto"]
            repository = Path(__file__).resolve().parents[1]
            commands = (
                [*base, "init-kek", "--keys", str(keys), "--kek-id", "old"],
                [*base, "rotate-kek", "--keys", str(keys), "--kek-id", "new"],
                [*base, "set-active-kek", "--keys", str(keys), "--kek-id", "old"],
            )
            for command in commands:
                completed = subprocess.run(
                    command,
                    cwd=repository,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, completed.returncode, completed.stderr)

            status = subprocess.run(
                [*base, "keyring-status", "--keys", str(keys)],
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, status.returncode, status.stderr)
            self.assertEqual(["  new", "* old"], sorted(status.stdout.splitlines()))

    def test_key_config_rejects_path_like_file_names(self):
        with self.assertRaisesRegex(ValueError, "单层文件名"):
            ModelCryptoConfig.from_mapping({"kek_file_name": "../kek.key"})
        with self.assertRaisesRegex(ValueError, "1..128"):
            ModelCryptoConfig.from_mapping({"kek_id": ""})

    def test_public_surface_is_small_and_runtime_only(self):
        self.assertEqual(
            {
                "ModelCryptoError",
                "create_yolo_architecture",
                "load_yolo_container",
            },
            set(model_crypto.__all__),
        )

    def test_trusted_pt_importer_is_outside_runtime_package(self):
        repository = Path(__file__).resolve().parents[1]
        runtime_dir = repository / "toolkit" / "model_crypto"
        self.assertFalse((runtime_dir / "offline_convert.py").exists())
        for source in runtime_dir.glob("*.py"):
            self.assertNotIn("torch.load(", source.read_text(encoding="utf-8"), source)
        build_script = repository / "scripts" / "convert_trusted_model.py"
        self.assertIn("torch.load(", build_script.read_text(encoding="utf-8"))

    def test_removed_configuration_fields_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "不支持的字段"):
            ModelCryptoConfig.from_mapping({"unsupported_option": True})


if __name__ == "__main__":
    unittest.main()
