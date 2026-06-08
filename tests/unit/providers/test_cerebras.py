import unittest
from unittest.mock import MagicMock, patch

from ._base import BaseProviderTestMixin
from src.api_clients.cerebras_client import CerebrasClient


class TestCerebrasClient(BaseProviderTestMixin, unittest.IsolatedAsyncioTestCase):
    PROVIDER_NAME = "Cerebras"
    HAS_STREAMING = True

    def _create_client(self):
        self.mock_sdk = MagicMock()
        with patch("src.api_clients.cerebras_client.Cerebras", return_value=self.mock_sdk):
            self.client = CerebrasClient()

    @property
    def _expected_model_ids(self):
        return ["zai-glm-4.7", "gpt-oss-120b"]

    @property
    def _model_for_test(self):
        return "gpt-oss-120b"

    def _mock_list_models_success(self):
        m1 = MagicMock(); m1.id = "zai-glm-4.7"
        m2 = MagicMock(); m2.id = "gpt-oss-120b"
        self.mock_sdk.models.list.return_value = MagicMock(data=[m1, m2])

    def _mock_list_models_empty(self):
        self.mock_sdk.models.list.return_value = MagicMock(data=[])

    def _mock_list_models_error(self):
        self.mock_sdk.models.list.side_effect = Exception("API error")

    def _mock_chat_success(self, text):
        c = MagicMock(); c.message.content = text
        self.mock_sdk.chat.completions.create.return_value = MagicMock(choices=[c])

    def _mock_chat_rate_limit(self):
        self.mock_sdk.chat.completions.create.side_effect = Exception("429 rate limit")

    def _mock_chat_auth_error(self):
        self.mock_sdk.chat.completions.create.side_effect = Exception("401 api key invalid")

    def _mock_chat_provider_error(self):
        self.mock_sdk.chat.completions.create.side_effect = Exception("Server error")

    def _mock_chat_stream(self, chunks):
        items = []
        for text in chunks:
            c = MagicMock(); c.choices[0].delta.content = text
            items.append(c)
        self.mock_sdk.chat.completions.create.return_value = items
