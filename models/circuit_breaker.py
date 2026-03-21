"""Per-provider circuit breaker management using aiobreaker."""

from __future__ import annotations

import datetime

from aiobreaker import CircuitBreaker, CircuitBreakerError, CircuitBreakerState

__all__ = ["CircuitBreakerError", "CircuitBreakerManager"]


class CircuitBreakerManager:
    """Per-provider circuit breaker management.

    Each provider gets its own CircuitBreaker instance that tracks consecutive
    failures. After ``fail_max`` failures the circuit opens and rejects calls.
    After ``timeout_duration`` seconds the circuit moves to half-open, allowing
    a single test call through.
    """

    def __init__(self, fail_max: int = 3, timeout_duration: float = 30.0) -> None:
        self._fail_max = fail_max
        self._timeout_duration = timeout_duration
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(
                fail_max=self._fail_max,
                timeout_duration=datetime.timedelta(seconds=self._timeout_duration),
                name=provider,
            )
        return self._breakers[provider]

    def is_open(self, provider: str) -> bool:
        """Check if the circuit is open (failing) for a provider."""
        if provider not in self._breakers:
            return False
        return self._breakers[provider].current_state == CircuitBreakerState.OPEN

    def configure(self, provider: str, fail_max: int, timeout_duration: float) -> None:
        """Set custom circuit breaker params for a specific provider."""
        self._breakers[provider] = CircuitBreaker(
            fail_max=fail_max,
            timeout_duration=datetime.timedelta(seconds=timeout_duration),
            name=provider,
        )
