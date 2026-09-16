from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("cryptography"),
    "requires torch and cryptography",
)
class ModelContainerV1Test(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch
        self.kek = b"k" * 32
        self.tensors = {
            "precise": torch.tensor([1.2345678901234567], dtype=torch.float64),
            "large_int": torch.tensor([2**60 + 1], dtype=torch.int64),
            "half": torch.tensor([1.5], dtype=torch.bfloat16),
            "boolean": torch.tensor([True, False], dtype=torch.bool),
            "empty": torch.empty((0, 2), dtype=torch.float32),
        }

    def _write(self, path: Path) -> None:
        from toolkit.model_crypto.container_v1 import write_model_container

        write_model_container(
            path,
            state_dict=self.tensors,
            model_yaml={"nc": 1, "backbone": [[-1, 1, "Conv", [8, 3, 1]]]},
            model_names={0: "helmet"},
            task="detect",
            kek=self.kek,
            key_id="test-key",
        )

    def test_all_supported_dtypes_keep_exact_bytes(self):
        from toolkit.model_crypto.container_v1 import read_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            artifact = read_model_container(path, kek=self.kek)

        self.assertEqual("detect", artifact.task)
        self.assertEqual({0: "helmet"}, artifact.model_names)
        for name, original in self.tensors.items():
            restored = artifact.state_dict[name]
            self.assertEqual(original.dtype, restored.dtype)
            self.assertEqual(original.shape, restored.shape)
            self.assertTrue(self.torch.equal(original, restored), name)

    def test_tampering_is_rejected_before_json_deserialization(self):
        from toolkit.model_crypto.container_v1 import read_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            damaged = bytearray(path.read_bytes())
            damaged[-20] ^= 1
            path.write_bytes(damaged)
            with (
                patch("toolkit.model_crypto.container_v1.json.loads") as loads,
                self.assertRaisesRegex(ValueError, "认证失败"),
            ):
                read_model_container(path, kek=self.kek)
            loads.assert_not_called()

    def test_wrong_key_is_rejected_before_json_deserialization(self):
        from toolkit.model_crypto.container_v1 import read_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            with (
                patch("toolkit.model_crypto.container_v1.json.loads") as loads,
                self.assertRaisesRegex(ValueError, "认证失败"),
            ):
                read_model_container(path, kek=b"x" * 32)
            loads.assert_not_called()

    def test_plaintext_metadata_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            data = path.read_bytes()
        self.assertNotIn(b"helmet", data)
        self.assertNotIn(b"backbone", data)


if __name__ == "__main__":
    unittest.main()
