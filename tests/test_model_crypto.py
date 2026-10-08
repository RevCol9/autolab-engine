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

    def test_key_config_rejects_path_like_file_names(self):
        with self.assertRaisesRegex(ValueError, "单层文件名"):
            ModelCryptoConfig.from_mapping({"kek_file_name": "../kek.key"})
        with self.assertRaisesRegex(ValueError, "1..128"):
            ModelCryptoConfig.from_mapping({"kek_id": ""})

    def test_public_surface_contains_only_container_runtime_operations(self):
        self.assertEqual(
            {
                "ENV_MODEL_CRYPTO_CONFIG",
                "ENV_MODEL_KEK",
                "ENV_MODEL_KEK_ID",
                "ENV_MODEL_KEYS_DIR",
                "FORMAT_VERSION",
                "ModelContainer",
                "ModelCryptoConfig",
                "create_yolo_architecture",
                "default_model_crypto_config",
                "init_kek",
                "load_kek",
                "load_yolo_container",
                "parse_kek_material",
                "read_model_container",
                "write_model_container",
            },
            set(model_crypto.__all__),
        )

    def test_removed_configuration_fields_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "不支持的字段"):
            ModelCryptoConfig.from_mapping({"unsupported_option": True})


if __name__ == "__main__":
    unittest.main()
