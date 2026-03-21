"""Tests for per-provider rate limiting."""

from __future__ import annotations

import asyncio
import time

import pytest

from models.rate_limiter import RateLimiterManager


class TestRateLimiterManager:
    async def test_acquire_default(self) -> None:
        """Acquiring once with default settings should not block."""
        manager = RateLimiterManager()
        start = time.monotonic()
        await manager.acquire("openrouter")
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

    async def test_per_provider_isolation(self) -> None:
        """Two providers should have independent limiters."""
        manager = RateLimiterManager()
        limiter_a = manager.get_limiter("provider_a")
        limiter_b = manager.get_limiter("provider_b")
        assert limiter_a is not limiter_b

    async def test_configure_custom(self) -> None:
        """Setting a custom rate should create a new limiter with those params."""
        manager = RateLimiterManager()
        # Get default limiter first
        default_limiter = manager.get_limiter("custom_provider")
        # Configure with custom values
        manager.configure("custom_provider", max_rate=10, time_period=30)
        custom_limiter = manager.get_limiter("custom_provider")
        # Should be a different limiter instance after reconfiguration
        assert default_limiter is not custom_limiter

    async def test_limiter_delays_at_capacity(self) -> None:
        """With max_rate=1 and time_period=1.0, second acquire should delay."""
        manager = RateLimiterManager()
        manager.configure("slow_provider", max_rate=1, time_period=1.0)

        # First acquire should be instant
        await manager.acquire("slow_provider")

        # Second acquire should block until the rate limit window passes
        start = time.monotonic()
        await manager.acquire("slow_provider")
        elapsed = time.monotonic() - start
        assert elapsed > 0.5, f"Expected delay >0.5s but got {elapsed:.3f}s"

    async def test_get_limiter_creates_once(self) -> None:
        """Calling get_limiter twice for the same provider returns the same instance."""
        manager = RateLimiterManager()
        first = manager.get_limiter("repeat_provider")
        second = manager.get_limiter("repeat_provider")
        assert first is second
