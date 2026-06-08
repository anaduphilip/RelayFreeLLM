import os
import json
import unittest


class TestNvidiaModelsConfiguration(unittest.TestCase):

    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "src",
            "provider_model_limits.json",
        )
        with open(config_path, "r") as f:
            return json.load(f)

    def _get_nvidia(self):
        config = self._load_config()
        for provider in config["providers"]:
            if provider["name"] == "Nvidia":
                return provider
        return None

    def test_nvidia_models_in_config(self):
        nvidia = self._get_nvidia()
        self.assertIsNotNone(nvidia, "Nvidia provider not found in config")
        self.assertGreater(len(nvidia["models"]), 0, "No models configured for Nvidia")
        for model in nvidia["models"]:
            self.assertIn("name", model)
            self.assertIsInstance(model["name"], str)
            self.assertGreater(len(model["name"]), 0)

    def test_nvidia_models_no_small_scale(self):
        nvidia = self._get_nvidia()
        for model in nvidia["models"]:
            self.assertNotEqual(
                model.get("scale"), "small",
                f"Model {model['name']} has 'small' scale",
            )

    def test_nvidia_models_have_context_length(self):
        nvidia = self._get_nvidia()
        for model in nvidia["models"]:
            self.assertIn(
                "Max_Context_Length", model,
                f"Model {model['name']} missing Max_Context_Length",
            )
            self.assertGreater(
                model["Max_Context_Length"], 0,
                f"Model {model['name']} has invalid Max_Context_Length",
            )
