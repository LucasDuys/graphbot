"""Per-provider rate limiting using aiolimiter."""

from __future__ import annotations

from aiolimiter import AsyncLimiter


class RateLimiterManager:
    """Per-provider rate limiting."""

    def __init__(
        self, default_max_rate: float = 30, default_time_period: float = 60
    ) -> None:
        self._default_max_rate = default_max_rate
        self._default_time_period = default_time_period
        self._limiters: dict[str, AsyncLimiter] = {}

    def get_limiter(self, provider: str) -> AsyncLimiter:
        """Get or create a rate limiter for a provider."""
        if provider not in self._limiters:
            self._limiters[provider] = AsyncLimiter(
                self._default_max_rate, self._default_time_period
            )
        return self._limiters[provider]

    async def acquire(self, provider: str) -> None:
        """Block until rate limit allows a request for this provider."""
        limiter = self.get_limiter(provider)
        await limiter.acquire()

    def configure(self, provider: str, max_rate: float, time_period: float) -> None:
        """Set custom rate limit for a specific provider."""
        self._limiters[provider] = AsyncLimiter(max_rate, time_period)
