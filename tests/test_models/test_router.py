"""Tests for the ModelRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.base import ModelProvider
from models.errors import ProviderError, RateLimitError
from models.router import DEFAULT_MODEL_MAP, ModelRouter


class FakeProvider(ModelProvider):
    """Minimal mock provider for router tests."""

    def __init__(self) -> None:
        self._mock_complete = AsyncMock()

    @property
    def name(self) -> str:
        return "fake"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


def _make_task(complexity: int = 1) -> TaskNode:
    return TaskNode(
        id="t1",
        description="test task",
        is_atomic=True,
        domain=Domain.SYSTEM,
        complexity=complexity,
        status=TaskStatus.READY,
    )


def _make_result(model: str = "some-model") -> CompletionResult:
    return CompletionResult(
        content="hello",
        model=model,
        tokens_in=10,
        tokens_out=5,
        latency_ms=42.0,
        cost=0.0,
    )


class TestModelSelection:
    def test_selects_correct_model_for_each_complexity(self) -> None:
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        for complexity in range(1, 6):
            model = router.get_model_for_complexity(complexity)
            assert model == DEFAULT_MODEL_MAP[complexity]

    def test_clamps_complexity_below_range(self) -> None:
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        assert router.get_model_for_complexity(0) == DEFAULT_MODEL_MAP[1]
        assert router.get_model_for_complexity(-5) == DEFAULT_MODEL_MAP[1]

    def test_clamps_complexity_above_range(self) -> None:
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        assert router.get_model_for_complexity(6) == DEFAULT_MODEL_MAP[5]
        assert router.get_model_for_complexity(100) == DEFAULT_MODEL_MAP[5]

    def test_custom_model_map_overrides_defaults(self) -> None:
        provider = FakeProvider()
        custom_map = {
            1: "custom/small",
            2: "custom/small",
            3: "custom/medium",
            4: "custom/large",
            5: "custom/large",
        }
        router = ModelRouter(provider=provider, model_map=custom_map)

        for complexity, expected in custom_map.items():
            assert router.get_model_for_complexity(complexity) == expected


class TestRouting:
    async def test_route_calls_provider_with_correct_model(self) -> None:
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result()
        router = ModelRouter(provider=provider)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "hi"}]

        result = await router.route(task, messages)

        assert isinstance(result, CompletionResult)
        provider._mock_complete.assert_awaited_once_with(
            messages, DEFAULT_MODEL_MAP[3],
        )

    async def test_route_forwards_kwargs_to_provider(self) -> None:
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result()
        router = ModelRouter(provider=provider)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "hi"}]

        await router.route(
            task, messages,
            response_format={"type": "json_object"},
        )

        provider._mock_complete.assert_awaited_once_with(
            messages, DEFAULT_MODEL_MAP[3],
            response_format={"type": "json_object"},
        )

    async def test_route_uses_task_complexity(self) -> None:
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result()
        router = ModelRouter(provider=provider)

        for complexity in range(1, 6):
            provider._mock_complete.reset_mock()
            task = _make_task(complexity=complexity)
            await router.route(task, [{"role": "user", "content": "test"}])

            provider._mock_complete.assert_awaited_once()
            call_args = provider._mock_complete.call_args
            assert call_args[0][1] == DEFAULT_MODEL_MAP[complexity]

    async def test_provider_error_propagates(self) -> None:
        provider = FakeProvider()
        provider._mock_complete.side_effect = ProviderError(
            "boom", provider="fake", model="x"
        )
        router = ModelRouter(provider=provider)

        with pytest.raises(ProviderError):
            await router.route(
                _make_task(complexity=1),
                [{"role": "user", "content": "hi"}],
            )

    async def test_rate_limit_error_propagates(self) -> None:
        provider = FakeProvider()
        provider._mock_complete.side_effect = RateLimitError(
            "too fast", provider="fake", model="x"
        )
        router = ModelRouter(provider=provider)

        with pytest.raises(RateLimitError):
            await router.route(
                _make_task(complexity=2),
                [{"role": "user", "content": "hi"}],
            )

    async def test_route_returns_completion_result(self) -> None:
        provider = FakeProvider()
        expected = _make_result(model="test-model")
        provider._mock_complete.return_value = expected
        router = ModelRouter(provider=provider)

        result = await router.route(
            _make_task(complexity=1),
            [{"role": "user", "content": "hi"}],
        )

        assert result is expected
