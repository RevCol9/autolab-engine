from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.validate_model_crypto_failures import CASE_NAMES, run_failure_matrix
from toolkit.model_crypto.container_v1 import write_model_container
from toolkit.model_crypto.envelope import init_kek, rotate_kek


class ModelCryptoFailureMatrixTest(unittest.TestCase):
    def test_matrix_preserves_source_and_reports_every_fault_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "keys"
            old_path, old_id = init_kek(keys, kek_id="old")
            source = root / "source.niii-model"
            write_model_container(
                source,
                state_dict={"weight": torch.arange(8, dtype=torch.float32)},
                model_yaml={"nc": 1},
                model_names={0: "item"},
                task="detect",
                kek=old_path.read_bytes(),
                key_id=old_id,
            )
            source_bytes = source.read_bytes()
            rotate_kek(keys, kek_id="new")

            report = run_failure_matrix(
                source,
                task="detect",
                key_dir=keys,
                work_dir=root / "faults",
                include_cuda=False,
            )

            self.assertEqual("passed", report["status"])
            self.assertEqual(set(CASE_NAMES), set(report["cases"]))
            self.assertEqual("skipped", report["cases"]["cuda_failure"]["status"])
            self.assertTrue(
                all(
                    result["status"] in {"passed", "skipped"}
                    for result in report["cases"].values()
                )
            )
            self.assertEqual([], report["artifactScan"]["matches"])
            self.assertEqual(source_bytes, source.read_bytes())
            self.assertTrue((root / "faults" / "failure-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
