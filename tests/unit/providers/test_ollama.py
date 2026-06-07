import unittest
from unittest.mock import MagicMock, patch
import httpx

from ._base import BaseProviderTestMixin
from src.api_clients.ollama_client import OllamaClient
from src.exceptions import ProviderError


class TestOllamaClient(BaseProviderTestMixin, unittest.IsolatedAsyncioTestCase):
    PROVIDER_NAME = "Ollama"
    HAS_STREAMING = True

    def _create_client(self):
        self.mock_get = patch("httpx.AsyncClient.get").start()
        self.addCleanup(self.mock_get.stop)
        self.mock_post = patch("httpx.AsyncClient.post").start()
        self.addCleanup(self.mock_post.stop)
        self.mock_stream = patch("httpx.AsyncClient.stream").start()
        self.addCleanup(self.mock_stream.stop)
        self.client = OllamaClient()

    @property
    def _expected_model_ids(self):
        return ["llama3:latest", "mistral:latest"]

    @property
    def _model_for_test(self):
        return "llama3"

    def _mock_list_models_success(self):
        resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        resp.json = lambda: {"models": [{"name": "llama3:latest"}, {"name": "mistral:latest"}]}
        self.mock_get.return_value = resp

    def _mock_list_models_empty(self):
        resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        resp.json = lambda: {"models": []}
        self.mock_get.return_value = resp

    def _mock_list_models_error(self):
        resp = MagicMock(status_code=500)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=resp
        )
        self.mock_get.return_value = resp

    def _mock_chat_success(self, text):
        resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        resp.json = lambda: {"choices": [{"message": {"content": text}}]}
        self.mock_post.return_value = resp

    def _mock_chat_rate_limit(self):
        resp = MagicMock(status_code=429)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=resp
        )
        self.mock_post.return_value = resp

    def _mock_chat_auth_error(self):
        resp = MagicMock(status_code=401)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=resp
        )
        self.mock_post.return_value = resp

    def _mock_chat_provider_error(self):
        resp = MagicMock(status_code=500)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=resp
        )
        self.mock_post.return_value = resp

    def _mock_chat_stream(self, chunks):
        mock_resp = MagicMock(status_code=200)

        async def mock_iter_lines():
            for text in chunks:
                yield f'data: {{"choices": [{{"delta": {{"content": "{text}"}}}}]}}'
            yield "data: [DONE]"

        mock_resp.aiter_lines = mock_iter_lines

        class AsyncContextMock:
            async def __aenter__(self_):
                return mock_resp
            async def __aexit__(self_, *args):
                pass

        self.mock_stream.return_value = AsyncContextMock()

    async def test_call_model_api_auth_error(self):
        self._mock_chat_auth_error()
        with self.assertRaises(ProviderError) as ctx:
            await self._call_api(
                messages=[{"role": "user", "content": "Hi"}],
                model=self._model_for_test,
            )
        self.assertEqual(ctx.exception.provider, self.PROVIDER_NAME)
