import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.config import settings
from src.exceptions import ProviderError, RateLimitError, AuthenticationError


class BaseProviderTestMixin:
    PROVIDER_NAME = ""
    HAS_STREAMING = False

    def _create_client(self):
        raise NotImplementedError

    def _mock_list_models_success(self):
        raise NotImplementedError

    def _mock_list_models_empty(self):
        raise NotImplementedError

    def _mock_list_models_error(self):
        raise NotImplementedError

    def _mock_chat_success(self, response_text):
        raise NotImplementedError

    def _mock_chat_rate_limit(self):
        raise NotImplementedError

    def _mock_chat_auth_error(self):
        raise NotImplementedError

    def _mock_chat_provider_error(self):
        raise NotImplementedError

    def _mock_chat_stream(self, chunks):
        raise NotImplementedError

    @property
    def _expected_model_ids(self):
        raise NotImplementedError

    @property
    def _model_for_test(self):
        raise NotImplementedError

    def setUp(self):
        self.key_patcher = patch.object(settings, "get_api_key", return_value="test-key")
        self.key_patcher.start()
        self.addCleanup(self.key_patcher.stop)
        self._create_client()

    async def _call_api(self, messages, model, stream=False):
        kwargs = dict(messages=messages, model=model, temperature=0.7, max_tokens=100)
        if stream:
            kwargs["stream"] = True
        return await self.client.call_model_api(**kwargs)

    async def test_list_models_success(self):
        self._mock_list_models_success()
        models = await self.client.list_models()
        for m in self._expected_model_ids:
            self.assertIn(m, models)

    async def test_list_models_empty(self):
        self._mock_list_models_empty()
        models = await self.client.list_models()
        self.assertEqual(len(models), 0)

    async def test_list_models_error(self):
        self._mock_list_models_error()
        models = await self.client.list_models()
        self.assertEqual(len(models), 0)

    async def test_call_model_api_success(self):
        expected = f"Hello from {self.PROVIDER_NAME}"
        self._mock_chat_success(expected)
        response = await self._call_api(
            messages=[{"role": "user", "content": "Hi"}],
            model=self._model_for_test,
        )
        self.assertEqual(response, expected)

    async def test_call_model_api_rate_limit(self):
        self._mock_chat_rate_limit()
        with self.assertRaises(RateLimitError) as ctx:
            await self._call_api(
                messages=[{"role": "user", "content": "Hi"}],
                model=self._model_for_test,
            )
        self.assertEqual(ctx.exception.provider, self.PROVIDER_NAME)

    async def test_call_model_api_auth_error(self):
        self._mock_chat_auth_error()
        with self.assertRaises(AuthenticationError) as ctx:
            await self._call_api(
                messages=[{"role": "user", "content": "Hi"}],
                model=self._model_for_test,
            )
        self.assertEqual(ctx.exception.provider, self.PROVIDER_NAME)

    async def test_call_model_api_provider_error(self):
        self._mock_chat_provider_error()
        with self.assertRaises(ProviderError) as ctx:
            await self._call_api(
                messages=[{"role": "user", "content": "Hi"}],
                model=self._model_for_test,
            )
        self.assertEqual(ctx.exception.provider, self.PROVIDER_NAME)

    async def test_call_model_api_streaming(self):
        if not self.HAS_STREAMING:
            self.skipTest("Streaming test not supported by this provider")
        chunks = ["Hello", " from", f" {self.PROVIDER_NAME}"]
        self._mock_chat_stream(chunks)
        generator = await self._call_api(
            messages=[{"role": "user", "content": "Hi"}],
            model=self._model_for_test,
            stream=True,
        )
        collected = []
        async for chunk in generator:
            collected.append(chunk)
        self.assertEqual("".join(collected), "".join(chunks))

    async def test_provider_name(self):
        self.assertEqual(self.client.PROVIDER_NAME, self.PROVIDER_NAME)
