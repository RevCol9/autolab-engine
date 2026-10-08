from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation.engines.yolo import YoloDetectEngine
from annotation.settings import ModelConfig, parse_models
from training.hparams import load_job_config, resolve_model_path
from training.paths import baseline_model_from_last_train, resolve_pretrained_model_path
from training.reporting import collect_artifacts
from training.trainer import collect_train_result


class EncryptedInferenceMainChainTest(unittest.TestCase):
    def test_yolo_engine_loads_authenticated_container_without_plaintext_model(self):
        import torch
        from ultralytics import YOLO

        from toolkit.model_crypto import write_model_container

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "keys"
            keys.mkdir()
            (keys / "kek.key").write_bytes(b"k" * 32)
            (keys / "kek_id.txt").write_text("e3-test\n", encoding="utf-8")
            source = YOLO("yolo11n.yaml", task="detect")
            encrypted = root / "detect.niii-model"
            write_model_container(
                encrypted,
                state_dict=source.model.state_dict(),
                model_yaml=source.model.yaml,
                model_names=source.model.names,
                task="detect",
                kek=b"k" * 32,
                key_id="e3-test",
            )
            config = ModelConfig(
                key="detect", name="detect", task="detect",
                path=str(encrypted), device="cpu",
            )
            with patch.dict(
                os.environ,
                {
                    "NIII_MODEL_KEYS_DIR": str(keys),
                    "NIII_MODEL_KEK_ID": "e3-test",
                    "NIII_ALLOW_CPU_MODEL_TESTS": "1",
                },
            ):
                engine = YoloDetectEngine(config)
                engine.load()

            self.assertEqual(source.names, engine.model.names)
            for name, tensor in source.model.state_dict().items():
                self.assertTrue(torch.equal(engine.model.model.state_dict()[name], tensor), name)
            self.assertEqual([], list(root.rglob("*.pt")))

    def test_yolo_configuration_rejects_plaintext_model_formats(self):
        for path in ("model.pt", "model.onnx", "model.pth"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "niii-model"):
                parse_models([{"key": "unsafe", "engine": "yolo", "path": path}])


class EncryptedTrainingArtifactContractTest(unittest.TestCase):
    def test_encrypted_trainer_rejects_fused_pretrained_model(self):
        from ultralytics import YOLO

        from training.encrypted_trainer import create_encrypted_trainer

        model = YOLO("yolo11n.yaml", task="detect").model
        model.fuse(verbose=False)

        with self.assertRaisesRegex(ValueError, "fused"):
            create_encrypted_trainer(
                model,
                task="detection",
                overrides={},
                allow_cpu_for_tests=True,
            )

    def _run_one_epoch_training(self, task: str) -> None:
        import yaml
        from PIL import Image
        from ultralytics import YOLO

        from toolkit.model_crypto import load_yolo_container
        from training.encrypted_checkpoints import save_training_model
        from training.encrypted_trainer import create_encrypted_trainer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "keys"
            keys.mkdir()
            (keys / "kek.key").write_bytes(b"k" * 32)
            (keys / "kek_id.txt").write_text("training-test\n", encoding="utf-8")
            images = root / "dataset" / "images"
            labels = root / "dataset" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            label = (
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
                if task == "segmentation"
                else "0 0.5 0.5 0.25 0.25\n"
            )
            for index in range(2):
                Image.new("RGB", (64, 64), color=(index * 40, 0, 0)).save(
                    images / f"{index}.jpg"
                )
                (labels / f"{index}.txt").write_text(label, encoding="utf-8")
            data_yaml = root / "dataset.yaml"
            data_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(root / "dataset"),
                        "train": "images",
                        "val": "images",
                        "names": {0: "object"},
                    }
                ),
                encoding="utf-8",
            )
            model_name = "yolo11n-seg.yaml" if task == "segmentation" else "yolo11n.yaml"
            container_task = "segment" if task == "segmentation" else "detect"
            source = YOLO(model_name, task=container_task)
            pretrained = root / "pretrained.niii-model"
            save_training_model(
                source.model,
                pretrained,
                task=task,
                key_dir=keys,
            )
            loaded = load_yolo_container(
                pretrained,
                "cpu",
                container_task,
                key_dir=keys,
                allow_cpu_for_tests=True,
            )
            trainer = create_encrypted_trainer(
                loaded.model,
                task=task,
                key_dir=keys,
                allow_cpu_for_tests=True,
                overrides={
                    "model": str(pretrained),
                    "data": str(data_yaml),
                    "project": str(root / "runs"),
                    "name": "train",
                    "exist_ok": True,
                    "epochs": 1,
                    "batch": 2,
                    "imgsz": 64,
                    "device": "cpu",
                    "workers": 0,
                    "amp": False,
                    "plots": False,
                    "val": False,
                    "save": True,
                    "save_period": -1,
                    "pretrained": False,
                    "close_mosaic": 0,
                    "verbose": False,
                },
            )
            trainer.train()

            self.assertTrue((trainer.wdir / "best.niii-model").is_file())
            self.assertTrue((trainer.wdir / "last.niii-model").is_file())
            self.assertEqual([], list(root.rglob("*.pt")))

    def test_one_epoch_detection_training_writes_no_plaintext_checkpoint(self):
        self._run_one_epoch_training("detection")

    def test_one_epoch_segmentation_training_writes_no_plaintext_checkpoint(self):
        self._run_one_epoch_training("segmentation")

    def test_training_snapshot_writes_and_replaces_only_encrypted_weights(self):
        import torch
        from ultralytics import YOLO

        from toolkit.model_crypto import read_model_container
        from training.encrypted_checkpoints import save_training_model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "keys"
            keys.mkdir()
            (keys / "kek.key").write_bytes(b"k" * 32)
            (keys / "kek_id.txt").write_text("training-test\n", encoding="utf-8")
            model = YOLO("yolo11n.yaml", task="detect").model
            destination = root / "weights" / "last.niii-model"

            save_training_model(model, destination, task="detection", key_dir=keys)
            first_name = next(iter(model.state_dict()))
            with torch.no_grad():
                model.state_dict()[first_name].fill_(0.25)
            save_training_model(model, destination, task="detection", key_dir=keys)

            restored = read_model_container(destination, kek=b"k" * 32)
            self.assertTrue(
                torch.equal(restored.state_dict[first_name], model.state_dict()[first_name].cpu())
            )
            self.assertEqual([], list(root.rglob("*.pt")))

    def test_pretrained_directory_resolves_encrypted_best_and_rejects_plaintext(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            published = root / "published" / "weights"
            published.mkdir(parents=True)
            encrypted = published / "best.niii-model"
            encrypted.write_bytes(b"encrypted-container-placeholder")
            with patch("training.paths.TRAINING_MODEL_ROOTS", (root,)):
                self.assertEqual(
                    str(encrypted), resolve_pretrained_model_path(root / "published")
                )
                plaintext = root / "plaintext.pt"
                plaintext.write_bytes(b"plaintext")
                with self.assertRaisesRegex(ValueError, "niii-model"):
                    resolve_pretrained_model_path(plaintext)

    def test_continue_requires_an_existing_encrypted_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            best = root / "storage" / "algorithms" / "demo" / "weights" / "best.niii-model"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"encrypted-container-placeholder")
            with (
                patch("training.paths.STORAGE_ROOT", root / "storage"),
                patch("training.paths.TRAINING_MODEL_ROOTS", (root,)),
            ):
                self.assertEqual(
                    str(best),
                    baseline_model_from_last_train("storage/algorithms/demo"),
                )
                best.unlink()
                with self.assertRaises(FileNotFoundError):
                    baseline_model_from_last_train("storage/algorithms/demo")

    def test_default_architecture_is_yaml_and_plaintext_model_name_is_rejected(self):
        self.assertEqual(
            "yolo11n.yaml",
            resolve_model_path({}, task="detection", defaults={"model": "yolo11n.yaml"}),
        )
        with self.assertRaisesRegex(ValueError, "受控 YOLO"):
            resolve_model_path({}, task="detection", defaults={"model": "yolo11n.pt"})
        with self.assertRaisesRegex(ValueError, "分割"):
            resolve_model_path({}, task="segmentation", defaults={"model": "yolo11n.yaml"})

    def test_from_scratch_architecture_ignores_working_directory_shadow(self):
        from toolkit.model_crypto import create_yolo_architecture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "yolo11n.yaml").write_text("not: a trusted model\n", encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                model = create_yolo_architecture(
                    "yolo11n.yaml",
                    "cpu",
                    "detect",
                    allow_cpu_for_tests=True,
                )
            finally:
                os.chdir(previous)
            self.assertEqual("yolo11n.yaml", model.model.yaml["yaml_file"])
            self.assertEqual([], list(root.rglob("*.pt")))

    def test_resume_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "resume"):
            resolve_model_path(
                {"resume": True},
                task="detection",
                defaults={"model": "yolo11n.yaml"},
            )

    def test_persisted_job_config_cannot_bypass_model_contract(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "data": "dataset.yaml",
                "save_dir": str(root / "run"),
                "train_task": "detection",
            }
            for model, extra in (
                ("yolo11n.pt", {}),
                ("arbitrary.yaml", {}),
                ("yolo11n.yaml", {"resume": True}),
            ):
                with self.subTest(model=model, extra=extra):
                    config_path = root / f"job-{len(list(root.glob('job-*')))}.yaml"
                    config_path.write_text(
                        yaml.safe_dump({**base, "model": model, **extra}),
                        encoding="utf-8",
                    )
                    with self.assertRaises((ValueError, RuntimeError)):
                        load_job_config(config_path)

    def test_result_and_report_expose_only_encrypted_training_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            weights = save_dir / "weights"
            weights.mkdir()
            best = weights / "best.niii-model"
            last = weights / "last.niii-model"
            best.write_bytes(b"best")
            last.write_bytes(b"last")

            result = collect_train_result(save_dir)
            self.assertEqual(str(best), result["weight_path"])
            self.assertEqual(
                [
                    str(Path("weights") / "best.niii-model"),
                    str(Path("weights") / "last.niii-model"),
                ],
                collect_artifacts(save_dir),
            )


if __name__ == "__main__":
    unittest.main()
