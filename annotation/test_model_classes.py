import unittest
from types import SimpleNamespace
from unittest.mock import patch

from annotation.engines.yolo import YoloDetectEngine
from annotation.settings import ModelConfig


class ModelClassesContractTest(unittest.TestCase):
    def test_yolo_classes_are_returned_in_class_index_order(self):
        engine = YoloDetectEngine(ModelConfig(
            key="helmet", name="helmet", task="detect", path="model.pt"
        ))
        engine.model = SimpleNamespace(names={2: "head_with_helmet", 0: "person", 1: "head_without_helmet"})

        self.assertEqual(
            ["person", "head_without_helmet", "head_with_helmet"],
            engine.classes(),
        )

    def test_model_load_response_exposes_loaded_yolo_classes(self):
        from api import inference

        config = ModelConfig(key="helmet", name="helmet", task="detect", path="model.pt")
        engine = YoloDetectEngine(config)
        engine.model = SimpleNamespace(names={0: "head_with_helmet"})
        with patch.object(inference, "get_model_config", return_value=config), patch.object(
            inference, "ensure_model_engine", return_value=engine
        ):
            response = inference.load_model("helmet")

        self.assertEqual(["head_with_helmet"], response["classes"])


if __name__ == "__main__":
    unittest.main()
