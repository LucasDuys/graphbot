"""Tests for per-provider circuit breaker management."""

from __future__ import annotations

import asyncio
import datetime

import pytest
from aiobreaker import CircuitBreakerError, CircuitBreakerState

from models.circuit_breaker import CircuitBreakerManager


class TestCircuitBreakerManager:
    def test_initial_state_closed(self) -> None:
        manager = CircuitBreakerManager(fail_max=3, timeout_duration=30.0)
        assert not manager.is_open("openrouter")

    async def test_trips_after_failures(self) -> None:
        manager = CircuitBreakerManager(fail_max=2, timeout_duration=30.0)
        breaker = manager.get_breaker("openrouter")

        async def failing() -> None:
            raise RuntimeError("provider down")

        for _ in range(2):
            try:
                await breaker.call_async(failing)
            except (RuntimeError, CircuitBreakerError):
                pass

        assert manager.is_open("openrouter")

    async def test_per_provider_isolation(self) -> None:
        manager = CircuitBreakerManager(fail_max=2, timeout_duration=30.0)

        async def failing() -> None:
            raise RuntimeError("down")

        breaker_a = manager.get_breaker("provider_a")
        for _ in range(2):
            try:
                await breaker_a.call_async(failing)
            except (RuntimeError, CircuitBreakerError):
                pass

        assert manager.is_open("provider_a")
        assert not manager.is_open("provider_b")

    def test_configure_custom(self) -> None:
        manager = CircuitBreakerManager(fail_max=3, timeout_duration=30.0)
        manager.configure("custom_provider", fail_max=10, timeout_duration=60.0)

        breaker = manager.get_breaker("custom_provider")
        assert breaker.fail_max == 10
        assert breaker.timeout_duration == datetime.timedelta(seconds=60.0)

    def test_get_breaker_creates_once(self) -> None:
        manager = CircuitBreakerManager(fail_max=3, timeout_duration=30.0)
        breaker1 = manager.get_breaker("openrouter")
        breaker2 = manager.get_breaker("openrouter")
        assert breaker1 is breaker2
