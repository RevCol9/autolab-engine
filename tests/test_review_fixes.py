from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.device import parse_device_index
from toolkit.cython_build.config import CythonBuildConfig
from toolkit.data_clean.config import DataCleanConfig
from toolkit.data_clean.export import export_dataset
from toolkit.data_clean.filters import apply_cleanvision_filters
from training.label_validation import validate_image_dir, validate_yolo_labels
from training.paths import safe_id
from training.run_artifacts import reset_run_artifacts


class SafePathTest(unittest.TestCase):
    def test_safe_id_rejects_dot_segments(self):
        for value in (".", ".."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_id("taskId", value)

    def test_clean_output_rejects_reserved_or_nested_names(self):
        for value in ("images", "labels", "weights", "../clean", "nested/clean"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DataCleanConfig(output_name=value)


class AtomicCleanExportTest(unittest.TestCase):
    def test_successful_export_replaces_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "clean_output"
            output.mkdir()
            (output / "previous.txt").write_text("old", encoding="utf-8")
            image = root / "one.jpg"
            image.write_bytes(b"image")
            label = root / "one.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            summary = export_dataset(
                {"one.jpg"},
                {},
                {"one.jpg": image},
                {"one": [(0, 0, "label_dir", label)]},
                root,
                output,
                DataCleanConfig(),
            )

            self.assertEqual(1, summary["exported"])
            self.assertFalse((output / "previous.txt").exists())
            self.assertTrue((output / "images" / "one.jpg").is_file())
            self.assertTrue((output / "labels" / "one.txt").is_file())
            self.assertTrue((output / "kept_manifest.csv").is_file())

    def test_failed_export_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "clean_output"
            output.mkdir()
            marker = output / "previous.txt"
            marker.write_text("keep", encoding="utf-8")
            label = root / "one.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                export_dataset(
                    {"one.jpg"},
                    {},
                    {"one.jpg": root / "missing.jpg"},
                    {"one": [(0, 0, "label_dir", label)]},
                    root,
                    output,
                    DataCleanConfig(),
                )

            self.assertEqual("keep", marker.read_text(encoding="utf-8"))
            self.assertEqual([], list(root.glob(".clean_output.tmp-*")))


class CleanVisionContractTest(unittest.TestCase):
    def test_missing_cleanvision_fails_by_default(self):
        config = DataCleanConfig()
        original_import = __import__

        def import_without_cleanvision(name, *args, **kwargs):
            if name == "cleanvision":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_cleanvision):
            with self.assertRaisesRegex(ValueError, "cleanvision is not installed"):
                apply_cleanvision_filters(set(), {}, {}, Path("."), config)


class LabelValidationTest(unittest.TestCase):
    def test_detection_validates_nested_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "labels" / "nested"
            labels.mkdir(parents=True)
            (labels / "one.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
            self.assertEqual(1, validate_yolo_labels(labels.parent, task="detection"))

    def test_detection_rejects_wrong_column_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "bad.txt").write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "5 列"):
                validate_yolo_labels(labels, task="detection")

    def test_segmentation_rejects_non_numeric_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "bad.txt").write_text(
                "0 0.1 0.1 0.2 nope 0.3 0.3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "须为数字"):
                validate_yolo_labels(labels, task="segmentation")

    def test_image_validation_accepts_nested_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "nested"
            images.mkdir()
            (images / "one.jpg").write_bytes(b"not-decoded-during-preflight")
            self.assertEqual(1, validate_image_dir(Path(tmp)))


class DeviceParsingTest(unittest.TestCase):
    def test_supported_devices(self):
        self.assertEqual("0", parse_device_index("cuda:0"))
        self.assertEqual("1", parse_device_index("1"))
        self.assertEqual("cpu", parse_device_index("cpu"))

    def test_multi_gpu_is_rejected_until_scheduler_supports_it(self):
        with self.assertRaisesRegex(ValueError, "单设备"):
            parse_device_index("0,1")


class RunArtifactResetTest(unittest.TestCase):
    def test_reset_removes_outputs_but_preserves_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            (save_dir / "images").mkdir()
            (save_dir / "labels").mkdir()
            (save_dir / "weights").mkdir()
            (save_dir / "weights" / "best.pt").write_bytes(b"old")
            (save_dir / "trainning_data.csv").write_text("old", encoding="utf-8")
            (save_dir / "train.log").write_text("old", encoding="utf-8")
            (save_dir / "images" / "source.jpg").write_bytes(b"source")

            reset_run_artifacts(save_dir)

            self.assertFalse((save_dir / "weights").exists())
            self.assertFalse((save_dir / "trainning_data.csv").exists())
            self.assertFalse((save_dir / "train.log").exists())
            self.assertTrue((save_dir / "images" / "source.jpg").is_file())


class CythonPackageConfigTest(unittest.TestCase):
    def test_runtime_imports_are_in_production_targets(self):
        config = CythonBuildConfig.autolab_engine(Path.cwd())
        self.assertIn("api/", config.production_targets)
        self.assertIn("toolkit/data_clean/", config.production_targets)
        self.assertNotIn("toolkit", config.skip_dirs)


if __name__ == "__main__":
    unittest.main()
