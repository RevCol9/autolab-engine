from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.validate_model_crypto_cuda import (
    _normalize_cuda_device,
    _prepare_run_dir,
    _require_cuda,
    _scan_plaintext_artifacts,
)


class CudaValidationScriptTest(unittest.TestCase):
    def test_normalizes_only_single_cuda_devices(self):
        self.assertEqual(torch.device("cuda:0"), _normalize_cuda_device("0"))
        self.assertEqual(torch.device("cuda:1"), _normalize_cuda_device("cuda:1"))
        with self.assertRaisesRegex(ValueError, "只接受"):
            _normalize_cuda_device("cpu")

    def test_missing_cuda_fails_instead_of_skipping(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "不允许跳过"):
                _require_cuda(torch.device("cuda:0"))

    def test_run_directory_must_be_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(root.resolve(), _prepare_run_dir(root))
            (root / "existing.txt").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "必须不存在或为空"):
                _prepare_run_dir(root)

    def test_plaintext_model_artifact_scan_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "best.niii-model").write_bytes(b"ciphertext")
            self.assertEqual([], _scan_plaintext_artifacts(root))
            plaintext = root / "weights" / "best.pt"
            plaintext.parent.mkdir()
            plaintext.write_bytes(b"plaintext")
            self.assertEqual([str(plaintext.resolve())], _scan_plaintext_artifacts(root))


if __name__ == "__main__":
    unittest.main()
