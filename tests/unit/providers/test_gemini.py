import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from ._base import BaseProviderTestMixin
from src.api_clients.gemini_client import GeminiClient
from src.exceptions import RateLimitError


class TestGeminiClient(BaseProviderTestMixin, unittest.IsolatedAsyncioTestCase):
    PROVIDER_NAME = "Gemini"
    HAS_STREAMING = True

    def _create_client(self):
        self.mock_genai_client = MagicMock()
        self.mock_genai_client.aio.models.generate_content = AsyncMock()
        self.mock_genai_client.aio.models.generate_content_stream = AsyncMock()
        with patch("src.api_clients.gemini_client.genai.Client",
                   return_value=self.mock_genai_client):
            self.client = GeminiClient()

    @property
    def _expected_model_ids(self):
        return ["gemini-2.5-flash", "gemini-2.5-pro"]

    @property
    def _model_for_test(self):
        return "gemini-2.5-flash"

    def _mock_list_models_success(self):
        def _m(name, actions):
            m = MagicMock(); m.name = name; m.supported_actions = actions
            return m
        self.mock_genai_client.models.list.return_value = [
            _m("models/gemini-2.5-flash", ["generateContent"]),
            _m("models/gemini-2.5-pro", ["generateContent"]),
            _m("models/other-model", []),
        ]

    def _mock_list_models_empty(self):
        self.mock_genai_client.models.list.return_value = []

    def _mock_list_models_error(self):
        self.mock_genai_client.models.list.side_effect = Exception("API error")

    def _mock_chat_success(self, text):
        resp = MagicMock(); resp.text = text
        self.mock_genai_client.aio.models.generate_content = AsyncMock(return_value=resp)

    def _mock_chat_rate_limit(self):
        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("429 rate limit")
        )

    def _mock_chat_auth_error(self):
        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("401 api key")
        )

    def _mock_chat_provider_error(self):
        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Server error")
        )

    def _mock_chat_stream(self, chunks):
        items = [MagicMock(text=t) for t in chunks]

        async def mock_stream_gen():
            for r in items:
                yield r

        self.mock_genai_client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_stream_gen()
        )

    async def test_call_model_api_rate_limit_quota(self):
        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("quota exceeded")
        )
        with self.assertRaises(RateLimitError) as ctx:
            await self.client.call_model_api(
                messages=[{"role": "user", "content": "Hi"}],
                model=self._model_for_test,
            )
        self.assertEqual(ctx.exception.provider, "Gemini")
