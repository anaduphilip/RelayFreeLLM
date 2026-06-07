import unittest
from unittest.mock import MagicMock, patch

from ._base import BaseProviderTestMixin
from src.api_clients.cloudflare_client import CloudflareClient


class TestCloudflareClient(BaseProviderTestMixin, unittest.IsolatedAsyncioTestCase):
    PROVIDER_NAME = "Cloudflare"

    def _create_client(self):
        self.mock_get = patch("httpx.AsyncClient.get").start()
        self.addCleanup(self.mock_get.stop)
        self.mock_post = patch("httpx.AsyncClient.post").start()
        self.addCleanup(self.mock_post.stop)
        self.client = CloudflareClient()

    @property
    def _expected_model_ids(self):
        return ["@cf/meta/llama-3.1-8b-instruct", "@cf/mistral/mistral-7b-instruct"]

    @property
    def _model_for_test(self):
        return "@cf/meta/llama-3.1-8b-instruct"

    def _mock_list_models_success(self):
        self.mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "success": True,
                "result": [
                    {"name": "@cf/meta/llama-3.1-8b-instruct"},
                    {"name": "@cf/mistral/mistral-7b-instruct"},
                ],
            },
        )

    def _mock_list_models_empty(self):
        self.mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"success": True, "result": []}
        )

    def _mock_list_models_error(self):
        self.mock_get.return_value = MagicMock(status_code=500)

    def _mock_chat_success(self, text):
        self.mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": text}}]},
        )

    def _mock_chat_rate_limit(self):
        self.mock_post.return_value = MagicMock(status_code=429, text="Rate limited")

    def _mock_chat_auth_error(self):
        self.mock_post.return_value = MagicMock(status_code=401, text="Unauthorized")

    def _mock_chat_provider_error(self):
        self.mock_post.return_value = MagicMock(status_code=500, text="Server error")
