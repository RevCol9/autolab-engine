from __future__ import annotations

import importlib.util
import tempfile
import tracemalloc
import unittest
from pathlib import Path


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
            "unsigned_byte": torch.tensor([0, 255], dtype=torch.uint8),
            "signed_byte": torch.tensor([-128, 127], dtype=torch.int8),
            "short": torch.tensor([-32768, 32767], dtype=torch.int16),
            "integer": torch.tensor([-2**31, 2**31 - 1], dtype=torch.int32),
            "float16": torch.tensor([-0.0, 1.5], dtype=torch.float16),
            "float32": torch.tensor([-0.0, 1.234567], dtype=torch.float32),
            "nan_payload": torch.tensor([0x7FC01234], dtype=torch.int32).view(torch.float32),
            "complex64": torch.tensor([1.25 - 2.5j], dtype=torch.complex64),
            "complex128": torch.tensor([1.25 - 2.5j], dtype=torch.complex128),
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
            self.assertTrue(
                self.torch.equal(
                    original.contiguous().reshape(-1).view(self.torch.uint8),
                    restored.contiguous().reshape(-1).view(self.torch.uint8),
                ),
                name,
            )

    def test_authenticated_header_key_id_and_ciphertext_tampering_is_rejected(self):
        from toolkit.model_crypto.container_v1 import read_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            original = path.read_bytes()
            # Offsets are from the public v1 header layout, not codec internals.
            for field, position in (
                ("wrap_nonce", 28),
                ("payload_nonce", 40),
                ("wrapped_dek", 52),
                ("key_id", 100),
                ("ciphertext", 108),
                ("tag", len(original) - 1),
            ):
                with self.subTest(field=field):
                    damaged = bytearray(original)
                    damaged[position] ^= 1
                    path.write_bytes(damaged)
                    with self.assertRaisesRegex(ValueError, "认证失败"):
                        read_model_container(path, kek=self.kek)

    def test_wrong_key_is_rejected_before_payload_parsing(self):
        from toolkit.model_crypto.container_v1 import read_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            with self.assertRaisesRegex(ValueError, "认证失败"):
                read_model_container(path, kek=b"x" * 32)

    def test_truncated_or_extended_file_is_rejected(self):
        from toolkit.model_crypto.container_v1 import read_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            original = path.read_bytes()
            for name, damaged in (
                ("truncated", original[:-1]),
                ("extended", original + b"X"),
            ):
                with self.subTest(name=name):
                    path.write_bytes(damaged)
                    with self.assertRaisesRegex(ValueError, "长度"):
                        read_model_container(path, kek=self.kek)

    def test_small_model_read_does_not_reserve_the_one_gib_limit(self):
        from toolkit.model_crypto.container_v1 import read_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            tracemalloc.start()
            try:
                artifact = read_model_container(path, kek=self.kek)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertEqual(set(self.tensors), set(artifact.state_dict))
        self.assertLess(peak, 64 * 1024**2)

    def test_plaintext_metadata_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            self._write(path)
            data = path.read_bytes()
        self.assertNotIn(b"helmet", data)
        self.assertNotIn(b"backbone", data)

    def test_writer_rejects_tensor_rank_that_reader_cannot_load(self):
        from toolkit.model_crypto.container_v1 import write_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            with self.assertRaisesRegex(ValueError, "shape"):
                write_model_container(
                    path,
                    state_dict={"too_many_dimensions": self.torch.zeros((1,) * 17)},
                    model_yaml={"nc": 1},
                    model_names={0: "helmet"},
                    task="detect",
                    kek=self.kek,
                    key_id="test-key",
                )
            self.assertFalse(path.exists())

    def test_writer_rejects_metadata_keys_that_would_change_after_json(self):
        from toolkit.model_crypto.container_v1 import write_model_container

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.niii-model"
            with self.assertRaisesRegex(TypeError, "元数据.*key"):
                write_model_container(
                    path,
                    state_dict=self.tensors,
                    model_yaml={1: "numeric", "1": "text"},
                    model_names={0: "helmet"},
                    task="detect",
                    kek=self.kek,
                    key_id="test-key",
                )
            self.assertFalse(path.exists())

    @unittest.skipUnless(importlib.util.find_spec("ultralytics"), "requires ultralytics")
    def test_trusted_yolo_detection_and_segmentation_conversion_retains_tensors(self):
        from ultralytics import YOLO

        from toolkit.model_crypto.container_v1 import read_model_container
        from toolkit.model_crypto.offline_convert import convert_trusted_pt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "keys"
            keys.mkdir()
            (keys / "kek.key").write_bytes(self.kek)
            for task, architecture in (
                ("detect", "yolo11n.yaml"),
                ("segment", "yolo11n-seg.yaml"),
            ):
                with self.subTest(task=task):
                    original = YOLO(architecture).model
                    source = root / f"{task}.pt"
                    encrypted = root / f"{task}.niii-model"
                    self.torch.save({"model": original}, source)
                    convert_trusted_pt(source, encrypted, task=task, key_dir=keys)
                    restored = read_model_container(encrypted, kek=self.kek)

                    self.assertEqual(task, restored.task)
                    self.assertEqual(original.yaml, restored.model_yaml)
                    self.assertEqual(original.names, restored.model_names)
                    self.assertEqual(set(original.state_dict()), set(restored.state_dict))
                    for name, tensor in original.state_dict().items():
                        got = restored.state_dict[name]
                        self.assertEqual(tensor.dtype, got.dtype, name)
                        self.assertEqual(tensor.shape, got.shape, name)
                        self.assertTrue(
                            self.torch.equal(
                                tensor.contiguous().reshape(-1).view(self.torch.uint8),
                                got.contiguous().reshape(-1).view(self.torch.uint8),
                            ),
                            name,
                        )


if __name__ == "__main__":
    unittest.main()
