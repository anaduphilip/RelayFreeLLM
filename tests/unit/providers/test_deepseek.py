import unittest
from unittest.mock import MagicMock, patch

from ._base import BaseProviderTestMixin
from src.api_clients.deepseek_client import DeepSeekClient


class TestDeepSeekClient(BaseProviderTestMixin, unittest.IsolatedAsyncioTestCase):
    PROVIDER_NAME = "DeepSeek"

    def _create_client(self):
        self.mock_get = patch("httpx.AsyncClient.get").start()
        self.addCleanup(self.mock_get.stop)
        self.mock_post = patch("httpx.AsyncClient.post").start()
        self.addCleanup(self.mock_post.stop)
        self.client = DeepSeekClient()

    @property
    def _expected_model_ids(self):
        return ["deepseek-chat", "deepseek-reasoner"]

    @property
    def _model_for_test(self):
        return "deepseek-chat"

    def _mock_list_models_success(self):
        self.mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]},
        )

    def _mock_list_models_empty(self):
        self.mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"data": []}
        )

    def _mock_list_models_error(self):
        self.mock_get.return_value = MagicMock(status_code=500)

    def _mock_chat_success(self, text):
        self.mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": text}}]},
        )

    def _mock_chat_rate_limit(self):
        resp = MagicMock(status_code=429, text="Rate limit exceeded")
        self.mock_post.return_value = resp

    def _mock_chat_auth_error(self):
        resp = MagicMock(status_code=401, text="Unauthorized")
        self.mock_post.return_value = resp

    def _mock_chat_provider_error(self):
        resp = MagicMock(status_code=500, text="Internal error")
        self.mock_post.return_value = resp

    async def test_base_url(self):
        self.assertEqual(self.client.base_url, "https://api.deepseek.com/v1")
