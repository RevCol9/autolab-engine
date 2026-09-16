from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from annotation.settings import parse_models
from shared.device import device_lock_key, parse_device_index, physical_cuda_device
from shared.gpu_lock import GpuDeviceLock
from toolkit.cython_build.config import CythonBuildConfig
from toolkit.data_clean.config import DataCleanConfig
from toolkit.data_clean.export import export_dataset
from toolkit.data_clean.filters import apply_cleanvision_filters
from training.dataset import read_class_names, validate_image_dir, validate_yolo_labels
from training.hparams import resolve_model_path
from training.paths import resolve_pretrained_model_path, safe_id
from training.run_artifacts import ensure_input_survives_reset, reset_run_artifacts
from training.resource_sampler import capture_gpu_snapshot
from training.service import JobManager, TrainJob


class SafePathTest(unittest.TestCase):
    def test_safe_id_rejects_dot_segments(self):
        for value in (".", ".."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_id("taskId", value)

    def test_clean_output_rejects_reserved_or_nested_names(self):
        for value in ("images", "labels", "weights", "../clean", "nested/clean"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DataCleanConfig(output_name=value)

    def test_clean_thresholds_reject_invalid_values(self):
        for thresholds in (
            {"dark_brightness_lt": 1.1},
            {"odd_size_lt": 0},
            {"blurry_blurriness_lt": float("nan")},
            {"unknown": 1.0},
        ):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                DataCleanConfig(thresholds=thresholds)


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
            self.assertEqual(
                1,
                validate_yolo_labels(labels.parent, task="detection", class_count=1),
            )

    def test_detection_rejects_wrong_column_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "bad.txt").write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "5 列"):
                validate_yolo_labels(labels, task="detection", class_count=1)

    def test_segmentation_rejects_non_numeric_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "bad.txt").write_text(
                "0 0.1 0.1 0.2 nope 0.3 0.3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "须为数字"):
                validate_yolo_labels(labels, task="segmentation", class_count=1)

    def test_image_validation_accepts_nested_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "nested"
            images.mkdir()
            (images / "one.jpg").write_bytes(b"not-decoded-during-preflight")
            self.assertEqual(1, validate_image_dir(Path(tmp)))

    def test_classes_must_be_non_empty_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            classes = Path(tmp) / "classes.txt"
            classes.write_text("helmet\nhelmet\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复类别"):
                read_class_names(classes)
            classes.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不能为空"):
                read_class_names(classes)

    def test_label_class_id_must_exist_in_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "bad.txt").write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "超出 classes.txt 范围"):
                validate_yolo_labels(labels, task="detection", class_count=1)


class DeviceParsingTest(unittest.TestCase):
    def test_supported_devices(self):
        self.assertEqual("0", parse_device_index("cuda:0"))
        self.assertEqual("1", parse_device_index("1"))
        self.assertEqual("cpu", parse_device_index("cpu"))

    def test_multi_gpu_is_rejected_until_scheduler_supports_it(self):
        with self.assertRaisesRegex(ValueError, "单设备"):
            parse_device_index("0,1")

    def test_lock_key_uses_physical_visible_device(self):
        self.assertEqual("3", physical_cuda_device("cuda:1", visible_devices="2,3"))
        self.assertEqual("3", device_lock_key("1", visible_devices="2,3"))

    def test_resource_sampler_queries_physical_visible_device(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="42, 1024, 8192, GPU Three\n",
            stderr="",
        )
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}),
            patch("training.resource_sampler.subprocess.run", return_value=completed) as run,
        ):
            snapshot = capture_gpu_snapshot("cuda:1")

        self.assertIn("--id=3", run.call_args.args[0])
        self.assertEqual("3", snapshot["physical_device"])
        self.assertEqual(42, snapshot["gpu"])

    def test_device_outside_visible_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "超出 CUDA_VISIBLE_DEVICES"):
            physical_cuda_device("1", visible_devices="4")

    def test_device_lock_is_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"NIII_GPU_LOCK_DIR": tmp, "CUDA_VISIBLE_DEVICES": "0"},
        ):
            first = GpuDeviceLock("0")
            second = GpuDeviceLock("cuda:0")
            self.assertTrue(first.acquire(blocking=False))
            self.assertFalse(second.acquire(blocking=False))
            first.release()
            self.assertTrue(second.acquire(blocking=False))
            second.release()


class AnnotationConfigValidationTest(unittest.TestCase):
    def test_unknown_engine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "engine=.*不受支持"):
            parse_models([{"key": "bad", "engine": "sam4", "task": "detect"}])

    def test_duplicate_model_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "key 重复"):
            parse_models([{"key": "same"}, {"key": "same"}])

    def test_invalid_yolo_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "YOLO 模型.*task"):
            parse_models([{"key": "bad", "engine": "yolo", "task": "vlm"}])

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conf 须位于"):
            parse_models([{"key": "bad", "conf": 1.5}])


class TrainStopStateTest(unittest.TestCase):
    def test_gpu_lock_is_kept_when_process_does_not_terminate(self):
        class FakeProcess:
            pid = 123

            @staticmethod
            def poll():
                return None

        class FakeLock:
            released = False

            def release(self):
                self.released = True

        manager = JobManager()
        lock = FakeLock()
        manager._job = TrainJob(
            job_id="train-1",
            param={},
            status="running",
            pid=123,
            _proc=FakeProcess(),
            _gpu_lock=lock,
        )
        with patch("training.service.kill_process_group", return_value=False):
            result = manager.stop()

        self.assertFalse(result["stopped"])
        self.assertEqual("stopping", result["job"]["status"])
        self.assertFalse(lock.released)


class PretrainedModelPathTest(unittest.TestCase):
    def test_resolves_conventional_best_weight_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = root / "published" / "weights" / "best.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"weights")
            with patch("training.paths.TRAINING_MODEL_ROOTS", (root,)):
                resolved = resolve_pretrained_model_path("published")
            self.assertEqual(str(model), resolved)

    def test_resolves_unique_pt_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = root / "published" / "custom.pt"
            model.parent.mkdir()
            model.write_bytes(b"weights")
            with patch("training.paths.TRAINING_MODEL_ROOTS", (root,)):
                resolved = resolve_pretrained_model_path(model.parent)
            self.assertEqual(str(model), resolved)

    def test_rejects_path_outside_configured_roots(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            allowed = Path(tmp).resolve()
            model = Path(other).resolve() / "model.pt"
            model.write_bytes(b"weights")
            with (
                patch("training.paths.TRAINING_MODEL_ROOTS", (allowed,)),
                self.assertRaisesRegex(ValueError, "路径越界"),
            ):
                resolve_pretrained_model_path(model)

    def test_continue_and_pretrained_path_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "不能同时使用"):
            resolve_model_path(
                {
                    "is_continue": True,
                    "last_train": "storage/algorithms/demo/models/baseline",
                    "pretrained_model_path": "/models/best.pt",
                },
                task="detection",
            )

    def test_model_and_pretrained_path_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "不能同时使用"):
            resolve_model_path(
                {
                    "model": "yolo11n.pt",
                    "pretrained_model_path": "/models/best.pt",
                },
                task="detection",
            )

    def test_empty_pretrained_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能为空"):
            resolve_model_path(
                {"pretrained_model_path": "   "},
                task="detection",
            )

    def test_rejects_model_that_would_be_deleted_on_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp).resolve()
            model = save_dir / "weights" / "best.pt"
            with self.assertRaisesRegex(ValueError, "会清理"):
                ensure_input_survives_reset(model, save_dir)


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
