"""Tests for multi-provider rotation and fallback in ModelRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.circuit_breaker import CircuitBreakerManager
from aiobreaker import CircuitBreakerError

from models.errors import (
    AllProvidersExhaustedError,
    ProviderError,
    RateLimitError,
)
from models.rate_limiter import RateLimiterManager
from models.router import ModelRouter


def _force_open_circuit(cb: CircuitBreakerManager, name: str) -> None:
    """Force a circuit breaker into the open state."""
    breaker = cb.get_breaker(name)
    breaker._inc_counter()
    try:
        breaker.state.on_failure(Exception("forced"))
    except CircuitBreakerError:
        pass  # Expected when threshold is reached.


def _make_provider(name: str = "fake") -> MagicMock:
    """Create a mock ModelProvider."""
    provider = MagicMock()
    type(provider).name = PropertyMock(return_value=name)
    provider.complete = AsyncMock()
    return provider


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


class TestMultiProviderInit:
    """ModelRouter accepts a list of providers."""

    def test_accepts_single_provider_backward_compat(self) -> None:
        """Single provider arg still works (backward compatibility)."""
        provider = _make_provider("openrouter")
        router = ModelRouter(provider=provider)
        assert len(router._providers) == 1

    def test_accepts_providers_list(self) -> None:
        """providers= kwarg accepts a list of providers."""
        p1 = _make_provider("openrouter")
        p2 = _make_provider("google")
        p3 = _make_provider("groq")
        router = ModelRouter(providers=[p1, p2, p3])
        assert len(router._providers) == 3

    def test_providers_list_takes_precedence_over_single(self) -> None:
        """When both provider and providers are given, providers wins."""
        single = _make_provider("single")
        p1 = _make_provider("first")
        p2 = _make_provider("second")
        router = ModelRouter(provider=single, providers=[p1, p2])
        assert len(router._providers) == 2
        assert router._providers[0].name == "first"

    def test_raises_if_no_provider_given(self) -> None:
        """Must provide at least one provider."""
        with pytest.raises(ValueError, match="At least one provider"):
            ModelRouter()


class TestFallbackOnRateLimit:
    """Primary rate limited -> falls back to secondary."""

    async def test_falls_back_on_rate_limit_error(self) -> None:
        """When primary raises RateLimitError, secondary is tried."""
        primary = _make_provider("openrouter")
        secondary = _make_provider("google")

        primary.complete.side_effect = RateLimitError(
            "rate limited", provider="openrouter", model="x"
        )
        expected = _make_result()
        secondary.complete.return_value = expected

        router = ModelRouter(providers=[primary, secondary])
        task = _make_task()
        result = await router.route(task, [{"role": "user", "content": "hi"}])

        assert result is expected
        primary.complete.assert_awaited_once()
        secondary.complete.assert_awaited_once()

    async def test_falls_back_on_provider_error(self) -> None:
        """When primary raises ProviderError, secondary is tried."""
        primary = _make_provider("openrouter")
        secondary = _make_provider("google")

        primary.complete.side_effect = ProviderError(
            "server error", provider="openrouter", model="x"
        )
        expected = _make_result()
        secondary.complete.return_value = expected

        router = ModelRouter(providers=[primary, secondary])
        task = _make_task()
        result = await router.route(task, [{"role": "user", "content": "hi"}])

        assert result is expected

    async def test_cascades_through_multiple_providers(self) -> None:
        """Falls through all failing providers to find one that works."""
        p1 = _make_provider("openrouter")
        p2 = _make_provider("google")
        p3 = _make_provider("groq")

        p1.complete.side_effect = RateLimitError(
            "rate limited", provider="openrouter", model="x"
        )
        p2.complete.side_effect = ProviderError(
            "server error", provider="google", model="x"
        )
        expected = _make_result()
        p3.complete.return_value = expected

        router = ModelRouter(providers=[p1, p2, p3])
        task = _make_task()
        result = await router.route(task, [{"role": "user", "content": "hi"}])

        assert result is expected
        p1.complete.assert_awaited_once()
        p2.complete.assert_awaited_once()
        p3.complete.assert_awaited_once()


class TestSkipOnCircuitOpen:
    """Circuit open -> skip provider entirely."""

    async def test_skips_provider_with_open_circuit(self) -> None:
        """Provider with open circuit is skipped without calling complete."""
        primary = _make_provider("openrouter")
        secondary = _make_provider("google")

        expected = _make_result()
        secondary.complete.return_value = expected

        cb = CircuitBreakerManager(fail_max=1, timeout_duration=60.0)
        _force_open_circuit(cb, "openrouter")

        router = ModelRouter(providers=[primary, secondary], circuit_breaker=cb)
        task = _make_task()
        result = await router.route(task, [{"role": "user", "content": "hi"}])

        assert result is expected
        # Primary should never be called because circuit is open.
        primary.complete.assert_not_awaited()
        secondary.complete.assert_awaited_once()

    async def test_skips_multiple_open_circuits(self) -> None:
        """Multiple providers with open circuits are all skipped."""
        p1 = _make_provider("openrouter")
        p2 = _make_provider("google")
        p3 = _make_provider("groq")

        expected = _make_result()
        p3.complete.return_value = expected

        cb = CircuitBreakerManager(fail_max=1, timeout_duration=60.0)
        _force_open_circuit(cb, "openrouter")
        _force_open_circuit(cb, "google")

        router = ModelRouter(providers=[p1, p2, p3], circuit_breaker=cb)
        task = _make_task()
        result = await router.route(task, [{"role": "user", "content": "hi"}])

        assert result is expected
        p1.complete.assert_not_awaited()
        p2.complete.assert_not_awaited()
        p3.complete.assert_awaited_once()


class TestAllProvidersDown:
    """All providers down -> graceful AllProvidersExhaustedError."""

    async def test_raises_all_providers_exhausted(self) -> None:
        """When all providers fail, raises AllProvidersExhaustedError."""
        p1 = _make_provider("openrouter")
        p2 = _make_provider("google")
        p3 = _make_provider("groq")

        p1.complete.side_effect = RateLimitError(
            "rate limited", provider="openrouter", model="x"
        )
        p2.complete.side_effect = ProviderError(
            "server error", provider="google", model="x"
        )
        p3.complete.side_effect = RateLimitError(
            "rate limited", provider="groq", model="x"
        )

        router = ModelRouter(providers=[p1, p2, p3])
        task = _make_task()

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.route(task, [{"role": "user", "content": "hi"}])

        assert len(exc_info.value.errors) == 3
        assert "openrouter" in str(exc_info.value)

    async def test_all_circuits_open_raises_exhausted(self) -> None:
        """When all circuits are open, raises AllProvidersExhaustedError."""
        p1 = _make_provider("openrouter")
        p2 = _make_provider("google")

        cb = CircuitBreakerManager(fail_max=1, timeout_duration=60.0)
        _force_open_circuit(cb, "openrouter")
        _force_open_circuit(cb, "google")

        router = ModelRouter(providers=[p1, p2], circuit_breaker=cb)
        task = _make_task()

        with pytest.raises(AllProvidersExhaustedError):
            await router.route(task, [{"role": "user", "content": "hi"}])

        p1.complete.assert_not_awaited()
        p2.complete.assert_not_awaited()

    async def test_exhausted_error_includes_all_errors(self) -> None:
        """AllProvidersExhaustedError collects errors from each provider."""
        p1 = _make_provider("openrouter")
        p2 = _make_provider("google")

        err1 = RateLimitError("rate limited", provider="openrouter", model="x")
        err2 = ProviderError("down", provider="google", model="x")
        p1.complete.side_effect = err1
        p2.complete.side_effect = err2

        router = ModelRouter(providers=[p1, p2])
        task = _make_task()

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.route(task, [{"role": "user", "content": "hi"}])

        errors = exc_info.value.errors
        assert errors[0] is err1
        assert errors[1] is err2


class TestCircuitBreakerRecording:
    """Circuit breaker records success/failure for the provider that was used."""

    async def test_success_recorded_on_fallback_provider(self) -> None:
        """When fallback succeeds, success is recorded for that provider."""
        primary = _make_provider("openrouter")
        secondary = _make_provider("google")

        primary.complete.side_effect = RateLimitError(
            "rate limited", provider="openrouter", model="x"
        )
        secondary.complete.return_value = _make_result()

        cb = MagicMock(spec=CircuitBreakerManager)
        cb.is_open.return_value = False
        mock_breaker_primary = MagicMock()
        mock_breaker_primary.state = MagicMock()
        mock_breaker_secondary = MagicMock()
        mock_breaker_secondary.state = MagicMock()
        cb.get_breaker.side_effect = lambda name: {
            "openrouter": mock_breaker_primary,
            "google": mock_breaker_secondary,
        }[name]

        router = ModelRouter(providers=[primary, secondary], circuit_breaker=cb)
        task = _make_task()
        await router.route(task, [{"role": "user", "content": "hi"}])

        # Primary failure recorded.
        mock_breaker_primary._inc_counter.assert_called_once()
        mock_breaker_primary.state.on_failure.assert_called_once()
        # Secondary success recorded.
        mock_breaker_secondary.state.on_success.assert_called_once()


class TestRateLimiterWithRotation:
    """Rate limiter acquire is called for each provider that is attempted."""

    async def test_acquire_called_for_each_attempted_provider(self) -> None:
        """Rate limiter.acquire is called for each provider tried."""
        primary = _make_provider("openrouter")
        secondary = _make_provider("google")

        primary.complete.side_effect = ProviderError(
            "fail", provider="openrouter", model="x"
        )
        secondary.complete.return_value = _make_result()

        rl = MagicMock(spec=RateLimiterManager)
        rl.acquire = AsyncMock()

        router = ModelRouter(providers=[primary, secondary], rate_limiter=rl)
        task = _make_task()
        await router.route(task, [{"role": "user", "content": "hi"}])

        assert rl.acquire.await_count == 2
        rl.acquire.assert_any_await("openrouter")
        rl.acquire.assert_any_await("google")
