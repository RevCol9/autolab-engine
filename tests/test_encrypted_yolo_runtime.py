"""Public contract tests for loading authenticated YOLO containers."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "ultralytics", "cryptography")),
    "requires torch, ultralytics and cryptography",
)
class EncryptedYoloRuntimeTest(unittest.TestCase):
    def setUp(self):
        from ultralytics import YOLO

        self.models = {
            "detect": YOLO("yolo11n.yaml", task="detect"),
            "segment": YOLO("yolo11n-seg.yaml", task="segment"),
        }
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.keys = self.root / "keys"
        self.keys.mkdir()
        (self.keys / "kek.key").write_bytes(b"k" * 32)
        (self.keys / "kek_id.txt").write_text("runtime-test\n", encoding="utf-8")

    def _write(
        self, task="detect", *, model=None, model_yaml=None, model_names=None, state_dict=None
    ):
        from toolkit.model_crypto import write_model_container

        source = model if model is not None else self.models[task]
        path = self.root / f"{task}.niii-model"
        write_model_container(
            path,
            state_dict=state_dict if state_dict is not None else source.model.state_dict(),
            model_yaml=model_yaml if model_yaml is not None else source.model.yaml,
            model_names=model_names if model_names is not None else source.model.names,
            task=task,
            kek=b"k" * 32,
            key_id="runtime-test",
        )
        return path

    def test_detection_container_restores_model_without_plaintext_checkpoint(self):
        import torch
        from ultralytics import YOLO

        from toolkit.model_crypto import load_yolo_container

        original = self.models["detect"]
        path = self._write()
        restored = load_yolo_container(
            path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
        )

        self.assertIsInstance(restored, YOLO)
        self.assertEqual(restored.task, "detect")
        self.assertEqual(restored.names, original.names)
        for name, tensor in original.model.state_dict().items():
            self.assertTrue(torch.equal(restored.model.state_dict()[name], tensor), name)
        self.assertEqual(list(self.root.rglob("*.pt")), [])

    def test_detection_and_segmentation_predictions_match_source(self):
        import numpy as np
        import torch

        from toolkit.model_crypto import load_yolo_container

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        for task in ("detect", "segment"):
            with self.subTest(task=task):
                path = self._write(task)
                restored = load_yolo_container(
                    path, "cpu", task, key_dir=self.keys, allow_cpu_for_tests=True,
                )
                options = {
                    "imgsz": 64, "conf": 0.001, "verbose": False,
                    "save": False, "device": "cpu",
                }
                original_result = self.models[task].predict(image, **options)[0]
                restored_result = restored.predict(image, **options)[0]
                torch.testing.assert_close(original_result.boxes.data, restored_result.boxes.data)
                if task == "segment":
                    self.assertEqual(original_result.masks is None, restored_result.masks is None)
                    if original_result.masks is not None:
                        torch.testing.assert_close(
                            original_result.masks.data, restored_result.masks.data,
                        )
        self.assertEqual(list(self.root.rglob("*.pt")), [])

    def test_v8_and_v26_detection_and_segmentation_load_from_encrypted_models(self):
        import numpy as np
        import torch
        from ultralytics import YOLO

        from toolkit.model_crypto import load_yolo_container, write_model_container

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        for family in ("yolov8", "yolo26"):
            for task in ("detect", "segment"):
                with self.subTest(family=family, task=task):
                    model_name = f"{family}n{'-seg' if task == 'segment' else ''}.yaml"
                    original = YOLO(model_name, task=task)
                    path = self.root / f"{family}-{task}.niii-model"
                    write_model_container(
                        path,
                        state_dict=original.model.state_dict(),
                        model_yaml=original.model.yaml,
                        model_names=original.model.names,
                        task=task,
                        kek=b"k" * 32,
                        key_id="runtime-test",
                    )

                    restored = load_yolo_container(
                        path, "cpu", task, key_dir=self.keys, allow_cpu_for_tests=True,
                    )
                    self.assertEqual(restored.model.yaml["yaml_file"], model_name)
                    options = {"imgsz": 64, "verbose": False, "save": False, "device": "cpu"}
                    actual = restored.predict(image, **options)[0]
                    expected = original.predict(image, **options)[0]
                    torch.testing.assert_close(actual.boxes.data, expected.boxes.data)
                    self.assertEqual(actual.masks is None, expected.masks is None)
                    if actual.masks is not None:
                        torch.testing.assert_close(actual.masks.data, expected.masks.data)
        self.assertEqual(list(self.root.rglob("*.pt")), [])

    def test_original_half_dtype_is_not_silently_cast_to_float32(self):
        import torch
        from ultralytics import YOLO

        from toolkit.model_crypto import load_yolo_container

        source = YOLO("yolo11n.yaml", task="detect")
        source.model.half()
        path = self._write(model=source)
        restored = load_yolo_container(
            path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
        )
        name = next(iter(source.model.state_dict()))
        self.assertEqual(restored.model.state_dict()[name].dtype, torch.float16)
        self.assertTrue(
            torch.equal(restored.model.state_dict()[name], source.model.state_dict()[name])
        )

    def test_custom_class_count_uses_the_trusted_architecture(self):
        import torch
        from ultralytics.nn.tasks import DetectionModel

        from toolkit.model_crypto import load_yolo_container

        source = DetectionModel(cfg=deepcopy(self.models["detect"].model.yaml), nc=2, verbose=False)
        path = self._write(
            model_yaml=source.yaml,
            model_names={0: "helmet", 1: "person"},
            state_dict=source.state_dict(),
        )
        restored = load_yolo_container(
            path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
        )
        self.assertEqual(restored.model.yaml["nc"], 2)
        self.assertEqual(restored.names, {0: "helmet", 1: "person"})
        for name, tensor in source.state_dict().items():
            self.assertTrue(torch.equal(restored.model.state_dict()[name], tensor), name)

    def test_non_default_yolo11_scale_is_restored(self):
        import torch
        from ultralytics import YOLO

        from toolkit.model_crypto import load_yolo_container

        source = YOLO("yolo11s.yaml", task="detect")
        path = self._write(model=source)
        restored = load_yolo_container(
            path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
        )
        self.assertEqual(restored.model.yaml["scale"], "s")
        for name, tensor in source.model.state_dict().items():
            self.assertTrue(torch.equal(restored.model.state_dict()[name], tensor), name)

    def test_source_filename_cannot_select_a_different_yolo_family(self):
        from ultralytics import YOLO

        from toolkit.model_crypto import load_yolo_container

        source = YOLO("yolo26n.yaml", task="detect")
        source_yaml = deepcopy(source.model.yaml)
        source_yaml["yaml_file"] = "yolov8n.yaml"  # Adversarial metadata, not a conversion step.
        path = self._write(model=source, model_yaml=source_yaml)
        restored = load_yolo_container(
            path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
        )
        self.assertEqual(restored.model.yaml["yaml_file"], "yolo26n.yaml")

    def test_gpu_load_is_available_when_cuda_exists(self):
        import torch

        from toolkit.model_crypto import load_yolo_container

        if not torch.cuda.is_available():
            self.skipTest("niiicv has no CUDA device")
        restored = load_yolo_container(self._write(), "cuda:0", "detect", key_dir=self.keys)
        self.assertEqual(next(restored.model.parameters()).device.type, "cuda")

    def test_unavailable_cuda_does_not_fall_back_to_cpu(self):
        import torch

        from toolkit.model_crypto import load_yolo_container

        if torch.cuda.is_available():
            self.skipTest("a CUDA device is present")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            load_yolo_container(self._write(), "cuda:0", "detect", key_dir=self.keys)

    def test_cpu_loading_requires_explicit_test_only_opt_in(self):
        from toolkit.model_crypto import load_yolo_container

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            load_yolo_container(self._write(), "cpu", "detect", key_dir=self.keys)

    def test_wrong_key_and_key_id_fail_closed(self):
        from toolkit.model_crypto import load_yolo_container

        path = self._write()
        (self.keys / "kek.key").write_bytes(b"x" * 32)
        with self.assertRaises(ValueError):
            load_yolo_container(
                path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
            )
        (self.keys / "kek.key").write_bytes(b"k" * 32)
        (self.keys / "kek_id.txt").write_text("another-key\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "key_id"):
            load_yolo_container(
                path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
            )

    def test_wrong_task_and_unsupported_yaml_are_rejected(self):
        from toolkit.model_crypto import load_yolo_container

        path = self._write()
        with self.assertRaisesRegex(ValueError, "任务不匹配"):
            load_yolo_container(
                path, "cpu", "segment", key_dir=self.keys, allow_cpu_for_tests=True,
            )

        unsafe_yaml = deepcopy(self.models["detect"].model.yaml)
        unsafe_yaml["activation"] = "__import__('os').system('echo unsafe')"
        unsafe_path = self.root / "unsafe.niii-model"
        from toolkit.model_crypto import write_model_container

        write_model_container(
            unsafe_path,
            state_dict=self.models["detect"].model.state_dict(),
            model_yaml=unsafe_yaml,
            model_names=self.models["detect"].model.names,
            task="detect",
            kek=b"k" * 32,
            key_id="runtime-test",
        )
        with self.assertRaisesRegex(ValueError, "架构"):
            load_yolo_container(
                unsafe_path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
            )

    def test_missing_state_key_and_invalid_class_ids_are_rejected(self):
        from toolkit.model_crypto import load_yolo_container

        state = dict(self.models["detect"].model.state_dict())
        state.pop(next(iter(state)))
        path = self._write(state_dict=state)
        with self.assertRaisesRegex(ValueError, "state_dict"):
            load_yolo_container(
                path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
            )

        names = dict(self.models["detect"].model.names)
        names.pop(0)
        bad_names_path = self.root / "bad-names.niii-model"
        from toolkit.model_crypto import write_model_container

        write_model_container(
            bad_names_path,
            state_dict=self.models["detect"].model.state_dict(),
            model_yaml=self.models["detect"].model.yaml,
            model_names=names,
            task="detect",
            kek=b"k" * 32,
            key_id="runtime-test",
        )
        with self.assertRaisesRegex(ValueError, "类别 ID"):
            load_yolo_container(
                bad_names_path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
            )

    def test_class_names_cannot_be_used_as_output_paths(self):
        from toolkit.model_crypto import load_yolo_container

        names = dict(self.models["detect"].model.names)
        names[0] = "../../checkpoint.pt"
        path = self._write(model_names=names)
        with self.assertRaisesRegex(ValueError, "类别名称"):
            load_yolo_container(
                path, "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
            )

    def test_plaintext_writers_are_disabled(self):
        from toolkit.model_crypto import load_yolo_container

        restored = load_yolo_container(
            self._write(), "cpu", "detect", key_dir=self.keys, allow_cpu_for_tests=True,
        )
        for method in ("train", "save", "export", "tune", "benchmark", "load"):
            with self.subTest(method=method), self.assertRaises(RuntimeError):
                getattr(restored, method)()
        self.assertEqual(list(self.root.rglob("*.pt")), [])


if __name__ == "__main__":
    unittest.main()
