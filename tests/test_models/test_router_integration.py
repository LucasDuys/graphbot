"""Integration tests for ModelRouter with rate limiter and circuit breaker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.circuit_breaker import CircuitBreakerManager
from models.errors import ProviderError
from models.rate_limiter import RateLimiterManager
from models.router import ModelRouter


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


class TestRouteWithRateLimiter:
    async def test_rate_limiter_acquire_called_before_completion(self) -> None:
        """Verify rate_limiter.acquire is called before provider.complete."""
        provider = _make_provider()
        provider.complete.return_value = _make_result()

        rate_limiter = MagicMock(spec=RateLimiterManager)
        rate_limiter.acquire = AsyncMock()

        call_order: list[str] = []
        rate_limiter.acquire.side_effect = lambda name: call_order.append("acquire")
        original_complete = provider.complete.side_effect

        async def track_complete(*args, **kwargs):
            call_order.append("complete")
            return _make_result()

        provider.complete = AsyncMock(side_effect=track_complete)

        router = ModelRouter(
            provider=provider, rate_limiter=rate_limiter
        )

        await router.route(_make_task(), [{"role": "user", "content": "hi"}])

        rate_limiter.acquire.assert_awaited_once_with("fake")
        assert call_order == ["acquire", "complete"]


class TestRouteWithCircuitBreakerClosed:
    async def test_normal_flow_with_closed_circuit(self) -> None:
        """Normal flow works when circuit breaker is closed."""
        provider = _make_provider()
        expected = _make_result()
        provider.complete.return_value = expected

        cb = MagicMock(spec=CircuitBreakerManager)
        cb.is_open.return_value = False
        mock_breaker = MagicMock()
        mock_breaker.state = MagicMock()
        cb.get_breaker.return_value = mock_breaker

        router = ModelRouter(provider=provider, circuit_breaker=cb)

        result = await router.route(
            _make_task(), [{"role": "user", "content": "hi"}]
        )

        assert result is expected
        cb.is_open.assert_called_once_with("fake")
        mock_breaker.state.on_success.assert_called_once()


class TestRouteWithCircuitBreakerOpen:
    async def test_raises_provider_error_when_circuit_open(self) -> None:
        """Raises ProviderError when the circuit is open."""
        provider = _make_provider()

        cb = MagicMock(spec=CircuitBreakerManager)
        cb.is_open.return_value = True

        router = ModelRouter(provider=provider, circuit_breaker=cb)

        with pytest.raises(ProviderError, match="Circuit open"):
            await router.route(
                _make_task(), [{"role": "user", "content": "hi"}]
            )

        # Provider should never be called when circuit is open.
        provider.complete.assert_not_awaited()


class TestRouteWithoutLimiters:
    async def test_backward_compatible_no_limiters(self) -> None:
        """Works the same as before when no rate_limiter or circuit_breaker."""
        provider = _make_provider()
        expected = _make_result()
        provider.complete.return_value = expected

        router = ModelRouter(provider=provider)

        result = await router.route(
            _make_task(complexity=3), [{"role": "user", "content": "hi"}]
        )

        assert result is expected
        provider.complete.assert_awaited_once()


class TestFailureRecordedInBreaker:
    async def test_failure_recorded_when_provider_raises(self) -> None:
        """When provider raises, circuit breaker is notified of the failure."""
        provider = _make_provider()
        error = ProviderError("boom", provider="fake", model="x")
        provider.complete.side_effect = error

        cb = MagicMock(spec=CircuitBreakerManager)
        cb.is_open.return_value = False
        mock_breaker = MagicMock()
        mock_breaker.state = MagicMock()
        cb.get_breaker.return_value = mock_breaker

        router = ModelRouter(provider=provider, circuit_breaker=cb)

        with pytest.raises(ProviderError):
            await router.route(
                _make_task(), [{"role": "user", "content": "hi"}]
            )

        mock_breaker._inc_counter.assert_called_once()
        mock_breaker.state.on_failure.assert_called_once_with(error)
        mock_breaker.state.on_success.assert_not_called()
