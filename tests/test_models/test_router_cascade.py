"""Tests for ModelRouter cascade mode."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.base import ModelProvider
from models.errors import AllProvidersExhaustedError, ProviderError
from models.router import CascadeConfig, DEFAULT_CASCADE_CHAIN, ModelRouter


class FakeProvider(ModelProvider):
    """Minimal mock provider for cascade tests."""

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


def _make_result(
    content: str = "hello",
    model: str = "some-model",
    tokens_in: int = 10,
    tokens_out: int = 5,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=42.0,
        cost=0.0,
    )


class TestCascadeConfig:
    def test_default_cascade_config(self) -> None:
        config = CascadeConfig()
        assert config.chain == DEFAULT_CASCADE_CHAIN
        assert config.confidence_threshold == 0.7
        assert config.max_attempts == 3

    def test_custom_cascade_config(self) -> None:
        custom_chain = ["model-a", "model-b"]
        config = CascadeConfig(
            chain=custom_chain,
            confidence_threshold=0.9,
            max_attempts=2,
        )
        assert config.chain == custom_chain
        assert config.confidence_threshold == 0.9
        assert config.max_attempts == 2

    def test_max_attempts_clamped_to_chain_length(self) -> None:
        config = CascadeConfig(
            chain=["model-a", "model-b"],
            max_attempts=10,
        )
        # max_attempts should not exceed chain length
        assert config.effective_max_attempts == 2

    def test_default_chain_has_three_models(self) -> None:
        assert len(DEFAULT_CASCADE_CHAIN) == 3


class TestCascadeModeActivation:
    def test_cascade_mode_off_by_default(self) -> None:
        provider = FakeProvider()
        router = ModelRouter(provider=provider)
        assert router.cascade_config is None

    def test_cascade_mode_enabled_via_constructor(self) -> None:
        provider = FakeProvider()
        config = CascadeConfig()
        router = ModelRouter(provider=provider, cascade_config=config)
        assert router.cascade_config is config

    def test_existing_route_still_works_with_cascade_off(self) -> None:
        """Backward compatibility: existing route() is unchanged."""
        provider = FakeProvider()
        router = ModelRouter(provider=provider)
        # cascade_config is None, but the router still functions normally
        assert router.cascade_config is None


class TestRouteCascade:
    async def test_returns_cheapest_model_result_when_confident(self) -> None:
        """If the cheapest model returns a confident result, use it."""
        provider = FakeProvider()
        cheap_result = _make_result(
            content="The answer is 42.",
            model="meta-llama/llama-3.1-8b-instruct",
            tokens_out=20,
        )
        provider._mock_complete.return_value = cheap_result

        config = CascadeConfig(confidence_threshold=0.5)
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "What is 6 * 7?"}]

        result = await router.route_cascade(task, messages)

        assert result.content == "The answer is 42."
        assert result.model == "meta-llama/llama-3.1-8b-instruct"
        # Only one call should be made (cheapest model)
        assert provider._mock_complete.await_count == 1

    async def test_escalates_when_cheap_model_gives_low_quality(self) -> None:
        """If cheapest model gives a low-quality result, escalate to next."""
        provider = FakeProvider()

        short_result = _make_result(
            content="I",
            model="meta-llama/llama-3.1-8b-instruct",
            tokens_out=1,
        )
        good_result = _make_result(
            content="The meaning of life is a philosophical question. Here is a detailed answer.",
            model="meta-llama/llama-3.3-70b-instruct",
            tokens_out=50,
        )
        provider._mock_complete.side_effect = [short_result, good_result]

        config = CascadeConfig(confidence_threshold=0.7)
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "Explain the meaning of life"}]

        result = await router.route_cascade(task, messages)

        # Should have escalated to the second model
        assert provider._mock_complete.await_count == 2
        assert result.content == good_result.content

    async def test_escalates_through_full_chain(self) -> None:
        """Escalates through all models in the chain when all give low quality."""
        provider = FakeProvider()

        bad1 = _make_result(content="", model="model-a", tokens_out=0)
        bad2 = _make_result(content="x", model="model-b", tokens_out=1)
        good = _make_result(
            content="A comprehensive and detailed answer to your question.",
            model="model-c",
            tokens_out=40,
        )
        provider._mock_complete.side_effect = [bad1, bad2, good]

        config = CascadeConfig(
            chain=["model-a", "model-b", "model-c"],
            confidence_threshold=0.7,
        )
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "complex question"}]

        result = await router.route_cascade(task, messages)

        assert provider._mock_complete.await_count == 3
        assert result.model == "model-c"

    async def test_returns_last_result_if_chain_exhausted(self) -> None:
        """If all models give low quality, return the last result anyway."""
        provider = FakeProvider()

        bad1 = _make_result(content="", model="model-a", tokens_out=0)
        bad2 = _make_result(content="x", model="model-b", tokens_out=1)
        provider._mock_complete.side_effect = [bad1, bad2]

        config = CascadeConfig(
            chain=["model-a", "model-b"],
            confidence_threshold=0.99,
        )
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "question"}]

        result = await router.route_cascade(task, messages)

        # Returns last result even if below threshold
        assert result.model == "model-b"
        assert provider._mock_complete.await_count == 2

    async def test_max_attempts_limits_escalation(self) -> None:
        """max_attempts limits how many models are tried."""
        provider = FakeProvider()

        bad1 = _make_result(content="", model="model-a", tokens_out=0)
        bad2 = _make_result(content="x", model="model-b", tokens_out=1)
        provider._mock_complete.side_effect = [bad1, bad2]

        config = CascadeConfig(
            chain=["model-a", "model-b", "model-c"],
            max_attempts=2,
        )
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "question"}]

        result = await router.route_cascade(task, messages)

        # Should stop after 2 attempts, never try model-c
        assert provider._mock_complete.await_count == 2
        assert result.model == "model-b"

    async def test_cascade_uses_first_model_in_chain(self) -> None:
        """Verify cascade starts with the first (cheapest) model in chain."""
        provider = FakeProvider()
        result = _make_result(
            content="Good response with plenty of detail for the user to understand fully.",
            model="cheapest",
            tokens_out=30,
        )
        provider._mock_complete.return_value = result

        config = CascadeConfig(chain=["cheapest", "medium", "expensive"])
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hello"}]

        await router.route_cascade(task, messages)

        call_args = provider._mock_complete.call_args
        assert call_args[0][1] == "cheapest"

    async def test_cascade_forwards_kwargs(self) -> None:
        """Verify kwargs are forwarded to provider.complete."""
        provider = FakeProvider()
        result = _make_result(
            content="A valid JSON response with enough content to be confident.",
            model="cheapest",
            tokens_out=30,
        )
        provider._mock_complete.return_value = result

        config = CascadeConfig(chain=["cheapest", "medium"])
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hello"}]

        await router.route_cascade(
            task, messages, response_format={"type": "json_object"}
        )

        provider._mock_complete.assert_awaited_once_with(
            messages, "cheapest", response_format={"type": "json_object"},
        )

    async def test_provider_error_escalates_to_next_model(self) -> None:
        """If a provider raises an error, escalate to next model."""
        provider = FakeProvider()

        error = ProviderError("rate limit", provider="fake", model="cheap")
        good_result = _make_result(
            content="A good comprehensive answer to your question.",
            model="expensive",
            tokens_out=30,
        )
        provider._mock_complete.side_effect = [error, good_result]

        config = CascadeConfig(chain=["cheap", "expensive"])
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hi"}]

        result = await router.route_cascade(task, messages)

        assert result.model == "expensive"
        assert provider._mock_complete.await_count == 2

    async def test_all_providers_error_raises_exhausted(self) -> None:
        """If all models in chain raise errors, raise AllProvidersExhaustedError."""
        provider = FakeProvider()

        error1 = ProviderError("fail1", provider="fake", model="cheap")
        error2 = ProviderError("fail2", provider="fake", model="expensive")
        provider._mock_complete.side_effect = [error1, error2]

        config = CascadeConfig(chain=["cheap", "expensive"])
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hi"}]

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.route_cascade(task, messages)

        assert len(exc_info.value.errors) == 2


class TestCascadeConfidenceMetadata:
    async def test_result_includes_confidence_metadata(self) -> None:
        """route_cascade returns a CascadeResult with confidence info."""
        from models.router import CascadeResult

        provider = FakeProvider()
        result = _make_result(
            content="A clear and complete answer to the question asked.",
            model="cheap",
            tokens_out=25,
        )
        provider._mock_complete.return_value = result

        config = CascadeConfig(chain=["cheap", "medium", "expensive"])
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hello"}]

        cascade_result = await router.route_cascade(task, messages)

        assert isinstance(cascade_result, CascadeResult)
        assert cascade_result.attempts == 1
        assert cascade_result.confidence >= 0.0
        assert cascade_result.confidence <= 1.0
        assert cascade_result.escalated is False

    async def test_escalated_result_has_correct_metadata(self) -> None:
        """When escalated, metadata reflects the escalation."""
        from models.router import CascadeResult

        provider = FakeProvider()

        short = _make_result(content="", model="cheap", tokens_out=0)
        good = _make_result(
            content="A detailed and helpful response.",
            model="expensive",
            tokens_out=25,
        )
        provider._mock_complete.side_effect = [short, good]

        config = CascadeConfig(
            chain=["cheap", "expensive"],
            confidence_threshold=0.7,
        )
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "explain quantum physics"}]

        cascade_result = await router.route_cascade(task, messages)

        assert isinstance(cascade_result, CascadeResult)
        assert cascade_result.attempts == 2
        assert cascade_result.escalated is True
        assert cascade_result.model == "expensive"
