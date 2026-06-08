import unittest
from unittest.mock import MagicMock, patch

from ._base import BaseProviderTestMixin
from src.api_clients.mistral_client import MistralClient


class TestMistralClient(BaseProviderTestMixin, unittest.IsolatedAsyncioTestCase):
    PROVIDER_NAME = "Mistral"
    HAS_STREAMING = True

    def _create_client(self):
        self.mock_sdk = MagicMock()
        with patch("src.api_clients.mistral_client.Mistral", return_value=self.mock_sdk):
            self.client = MistralClient()

    @property
    def _expected_model_ids(self):
        return ["mistral-large-latest", "mistral-small-2507"]

    @property
    def _model_for_test(self):
        return "mistral-small-2507"

    def _mock_list_models_success(self):
        m1 = MagicMock(); m1.id = "mistral-large-latest"
        m2 = MagicMock(); m2.id = "mistral-small-2507"
        self.mock_sdk.models.list.return_value = MagicMock(data=[m1, m2])

    def _mock_list_models_empty(self):
        self.mock_sdk.models.list.return_value = MagicMock(data=[])

    def _mock_list_models_error(self):
        self.mock_sdk.models.list.side_effect = Exception("API error")

    def _mock_chat_success(self, text):
        c = MagicMock(); c.message.content = text
        self.mock_sdk.chat.complete.return_value = MagicMock(choices=[c])

    def _mock_chat_rate_limit(self):
        self.mock_sdk.chat.complete.side_effect = Exception("429 rate limit exceeded")

    def _mock_chat_auth_error(self):
        self.mock_sdk.chat.complete.side_effect = Exception("401 api key invalid")

    def _mock_chat_provider_error(self):
        self.mock_sdk.chat.complete.side_effect = Exception("Server error")

    def _mock_chat_stream(self, chunks):
        items = []
        for text in chunks:
            d = MagicMock(); d.content = text
            c = MagicMock(); c.data.choices[0].delta = d
            items.append(c)

        async def mock_async_iter():
            for c in items:
                yield c

        async def mock_stream_async(*args, **kwargs):
            return mock_async_iter()

        self.mock_sdk.chat.stream_async = mock_stream_async
