"""Model router that selects models based on task complexity.

Supports multi-provider rotation: try primary, on rate limit or error fall
back to the next provider, skip providers whose circuit breaker is open.
"""

from __future__ import annotations

import logging

from core_gb.types import CompletionResult, TaskNode
from models.base import ModelProvider
from models.circuit_breaker import CircuitBreakerManager
from models.errors import AllProvidersExhaustedError, ProviderError
from models.rate_limiter import RateLimiterManager

logger = logging.getLogger(__name__)

DEFAULT_MODEL_MAP: dict[int, str] = {
    1: "meta-llama/llama-3.1-8b-instruct",
    2: "meta-llama/llama-4-scout",
    3: "meta-llama/llama-3.3-70b-instruct",
    4: "google/gemini-2.5-flash-preview",
    5: "google/gemini-2.5-pro-preview",
}

_MIN_COMPLEXITY = 1
_MAX_COMPLEXITY = 5


class ModelRouter:
    """Routes tasks to appropriate models based on complexity level.

    Accepts one or more providers. On failure the router rotates through
    providers in order: skip providers with an open circuit breaker, try the
    next provider on rate-limit or other provider errors, and raise
    ``AllProvidersExhaustedError`` only when every provider has been
    exhausted.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
        providers: list[ModelProvider] | None = None,
        model_map: dict[int, str] | None = None,
        rate_limiter: RateLimiterManager | None = None,
        circuit_breaker: CircuitBreakerManager | None = None,
    ) -> None:
        if providers is not None:
            self._providers: list[ModelProvider] = list(providers)
        elif provider is not None:
            self._providers = [provider]
        else:
            raise ValueError("At least one provider must be supplied")

        self._model_map: dict[int, str] = model_map or DEFAULT_MODEL_MAP
        self._rate_limiter: RateLimiterManager | None = rate_limiter
        self._circuit_breaker: CircuitBreakerManager | None = circuit_breaker

    @property
    def rate_limiter(self) -> RateLimiterManager | None:
        return self._rate_limiter

    @property
    def circuit_breaker(self) -> CircuitBreakerManager | None:
        return self._circuit_breaker

    def get_model_for_complexity(self, complexity: int) -> str:
        """Return the model name for a given complexity level, clamped to [1, 5]."""
        clamped = max(_MIN_COMPLEXITY, min(_MAX_COMPLEXITY, complexity))
        return self._model_map[clamped]

    async def route(
        self, task: TaskNode, messages: list[dict], **kwargs: object
    ) -> CompletionResult:
        """Select a model based on task complexity and try each provider in order.

        Rotation strategy:
        1. Skip providers whose circuit breaker is open.
        2. Acquire rate limiter for the provider.
        3. Call provider.complete().
        4. On success, record in circuit breaker and return.
        5. On failure (ProviderError), record in circuit breaker and try next.
        6. If all providers fail, raise AllProvidersExhaustedError.
        """
        model = self.get_model_for_complexity(task.complexity)
        errors: list[ProviderError] = []

        for provider in self._providers:
            # Circuit breaker: skip if circuit is open.
            if self._circuit_breaker is not None and self._circuit_breaker.is_open(provider.name):
                logger.debug("Skipping %s: circuit open", provider.name)
                errors.append(
                    ProviderError("Circuit open", provider=provider.name, model=model)
                )
                continue

            # Rate limiter: block until rate limit allows a request.
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire(provider.name)

            # Call provider and record outcome in circuit breaker.
            try:
                result = await provider.complete(messages, model, **kwargs)
            except ProviderError as exc:
                logger.debug(
                    "Provider %s failed: %s, trying next", provider.name, exc
                )
                if self._circuit_breaker is not None:
                    breaker = self._circuit_breaker.get_breaker(provider.name)
                    breaker._inc_counter()
                    breaker.state.on_failure(exc)
                errors.append(exc)
                continue
            except Exception as exc:
                logger.debug(
                    "Provider %s raised unexpected error: %s, trying next",
                    provider.name, exc,
                )
                if self._circuit_breaker is not None:
                    breaker = self._circuit_breaker.get_breaker(provider.name)
                    breaker._inc_counter()
                    breaker.state.on_failure(exc)
                errors.append(
                    ProviderError(str(exc), provider=provider.name, model=model)
                )
                continue
            else:
                if self._circuit_breaker is not None:
                    breaker = self._circuit_breaker.get_breaker(provider.name)
                    breaker.state.on_success()
                return result

        raise AllProvidersExhaustedError(errors)
