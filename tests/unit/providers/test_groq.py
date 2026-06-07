import unittest
from unittest.mock import MagicMock, patch

from ._base import BaseProviderTestMixin
from src.api_clients.groq_client import GroqClient


class TestGroqClient(BaseProviderTestMixin, unittest.IsolatedAsyncioTestCase):
    PROVIDER_NAME = "Groq"
    HAS_STREAMING = True

    def _create_client(self):
        self.mock_sdk = MagicMock()
        with patch("src.api_clients.groq_client.Groq", return_value=self.mock_sdk):
            self.client = GroqClient()

    @property
    def _expected_model_ids(self):
        return ["llama-3.3-70b-versatile", "qwen/qwen3-32b"]

    @property
    def _model_for_test(self):
        return "llama-3.3-70b-versatile"

    def _mock_list_models_success(self):
        m1 = MagicMock(); m1.id = "llama-3.3-70b-versatile"
        m2 = MagicMock(); m2.id = "qwen/qwen3-32b"
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
