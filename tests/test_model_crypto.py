from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import toolkit.model_crypto as model_crypto
from toolkit.model_crypto.config import ModelCryptoConfig
from toolkit.model_crypto.envelope import init_kek, load_kek, parse_kek_material


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
