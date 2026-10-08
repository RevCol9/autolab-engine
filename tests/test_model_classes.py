from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from annotation.engines.yolo import YoloDetectEngine
from annotation.registry import EngineRegistry
from annotation.settings import ModelConfig, Settings
from toolkit.model_crypto import ModelCryptoError


class ModelClassesContractTest(unittest.TestCase):
    def test_crypto_load_error_exposes_stable_code(self):
        config = ModelConfig(
            key="helmet",
            name="helmet",
            task="detect",
            path="model.niii-model",
            device="cpu",
        )
        def fail_to_load() -> None:
            raise ModelCryptoError(ModelCryptoError.AUTH_FAILED, "模型认证失败")

        engine = SimpleNamespace(load=fail_to_load)
        registry = EngineRegistry(
            Settings(config_path="test", models=[config]),
            engine_factory=lambda _: engine,
        )

        with self.assertRaises(HTTPException) as caught:
            registry.load("helmet")

        self.assertEqual(503, caught.exception.status_code)
        self.assertEqual(
            {"code": ModelCryptoError.AUTH_FAILED, "message": "模型认证失败"},
            caught.exception.detail,
        )

    def test_yolo_classes_are_returned_in_class_index_order(self):
        engine = YoloDetectEngine(ModelConfig(
            key="helmet", name="helmet", task="detect", path="model.niii-model"
        ))
        engine.model = SimpleNamespace(names={2: "head_with_helmet", 0: "person", 1: "head_without_helmet"})

        self.assertEqual(
            ["person", "head_without_helmet", "head_with_helmet"],
            engine.classes(),
        )

    def test_model_load_response_exposes_loaded_yolo_classes(self):
        from api import inference

        config = ModelConfig(
            key="helmet", name="helmet", task="detect", path="model.niii-model"
        )
        engine = YoloDetectEngine(config)
        engine.model = SimpleNamespace(names={0: "head_with_helmet"})
        with patch.object(
            inference.MODEL_RUNTIME, "get_model_config", return_value=config
        ), patch.object(
            inference.MODEL_RUNTIME, "load", return_value=engine
        ):
            response = inference.load_model("helmet")

        self.assertEqual(["head_with_helmet"], response["classes"])


if __name__ == "__main__":
    unittest.main()
