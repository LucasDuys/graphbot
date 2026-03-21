"""Tests for GraphBot model providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import CompletionResult
from models.base import ModelProvider
from models.errors import AuthError, ProviderError, RateLimitError
from models.openrouter import OpenRouterProvider


class TestModelProviderABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ModelProvider()  # type: ignore[abstract]


class TestOpenRouterProviderInit:
    def test_init_with_explicit_key(self) -> None:
        provider = OpenRouterProvider(api_key="test-key-123")
        assert provider.name == "openrouter"

    def test_init_with_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-456")
        provider = OpenRouterProvider()
        assert provider.name == "openrouter"

    def test_init_missing_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(AuthError) as exc_info:
            OpenRouterProvider()
        assert "openrouter" in str(exc_info.value).lower()


class TestOpenRouterProviderComplete:
    @pytest.fixture
    def provider(self) -> OpenRouterProvider:
        return OpenRouterProvider(api_key="test-key")

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "test response"
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.model = "openrouter/meta-llama/llama-3.3-70b-versatile"
        return response

    async def test_successful_completion(
        self, provider: OpenRouterProvider, mock_response: MagicMock
    ) -> None:
        with patch("models.openrouter.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            result = await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="meta-llama/llama-3.3-70b-versatile",
            )

            assert isinstance(result, CompletionResult)
            assert result.content == "test response"
            assert result.tokens_in == 10
            assert result.tokens_out == 5
            assert result.model == "openrouter/meta-llama/llama-3.3-70b-versatile"
            assert result.latency_ms > 0
            assert isinstance(result.cost, float)

            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args
            assert call_kwargs[1]["model"] == "openrouter/meta-llama/llama-3.3-70b-versatile"

    async def test_rate_limit_error(self, provider: OpenRouterProvider) -> None:
        with patch("models.openrouter.acompletion", new_callable=AsyncMock) as mock_acompletion:
            from litellm.exceptions import RateLimitError as LitellmRateLimitError

            mock_acompletion.side_effect = LitellmRateLimitError(
                message="Rate limit exceeded",
                llm_provider="openrouter",
                model="meta-llama/llama-3.3-70b-versatile",
            )

            with pytest.raises(RateLimitError) as exc_info:
                await provider.complete(
                    messages=[{"role": "user", "content": "hello"}],
                    model="meta-llama/llama-3.3-70b-versatile",
                )
            assert exc_info.value.provider == "openrouter"
            assert exc_info.value.model == "meta-llama/llama-3.3-70b-versatile"

    async def test_auth_error(self, provider: OpenRouterProvider) -> None:
        with patch("models.openrouter.acompletion", new_callable=AsyncMock) as mock_acompletion:
            from litellm.exceptions import AuthenticationError as LitellmAuthError

            mock_acompletion.side_effect = LitellmAuthError(
                message="Invalid API key",
                llm_provider="openrouter",
                model="meta-llama/llama-3.3-70b-versatile",
            )

            with pytest.raises(AuthError) as exc_info:
                await provider.complete(
                    messages=[{"role": "user", "content": "hello"}],
                    model="meta-llama/llama-3.3-70b-versatile",
                )
            assert exc_info.value.provider == "openrouter"
            assert exc_info.value.model == "meta-llama/llama-3.3-70b-versatile"

    async def test_generic_error(self, provider: OpenRouterProvider) -> None:
        with patch("models.openrouter.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = Exception("Something went wrong")

            with pytest.raises(ProviderError) as exc_info:
                await provider.complete(
                    messages=[{"role": "user", "content": "hello"}],
                    model="meta-llama/llama-3.3-70b-versatile",
                )
            assert exc_info.value.provider == "openrouter"
            assert exc_info.value.model == "meta-llama/llama-3.3-70b-versatile"

    async def test_model_prefixed_with_openrouter(
        self, provider: OpenRouterProvider, mock_response: MagicMock
    ) -> None:
        with patch("models.openrouter.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="meta-llama/llama-3.3-70b-versatile",
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert call_kwargs["model"] == "openrouter/meta-llama/llama-3.3-70b-versatile"

    async def test_kwargs_passed_through(
        self, provider: OpenRouterProvider, mock_response: MagicMock
    ) -> None:
        with patch("models.openrouter.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="meta-llama/llama-3.3-70b-versatile",
                temperature=0.5,
                max_tokens=100,
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 100
