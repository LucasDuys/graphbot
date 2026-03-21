"""Model router that selects models based on task complexity."""

from __future__ import annotations

from core_gb.types import CompletionResult, TaskNode
from models.base import ModelProvider
from models.circuit_breaker import CircuitBreakerManager
from models.errors import ProviderError
from models.rate_limiter import RateLimiterManager

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
    """Routes tasks to appropriate models based on complexity level."""

    def __init__(
        self,
        provider: ModelProvider,
        model_map: dict[int, str] | None = None,
        rate_limiter: RateLimiterManager | None = None,
        circuit_breaker: CircuitBreakerManager | None = None,
    ) -> None:
        # Store as list for future multi-provider support (Phase 3).
        self._providers: list[ModelProvider] = [provider]
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
        """Select a model based on task complexity and run completion."""
        model = self.get_model_for_complexity(task.complexity)
        provider = self._providers[0]

        # Circuit breaker: fast-fail if circuit is open.
        if self._circuit_breaker is not None and self._circuit_breaker.is_open(provider.name):
            raise ProviderError("Circuit open", provider=provider.name, model=model)

        # Rate limiter: block until rate limit allows a request.
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(provider.name)

        # Call provider and record outcome in circuit breaker.
        try:
            result = await provider.complete(messages, model, **kwargs)
        except Exception as exc:
            if self._circuit_breaker is not None:
                breaker = self._circuit_breaker.get_breaker(provider.name)
                breaker._inc_counter()
                breaker.state.on_failure(exc)
            raise
        else:
            if self._circuit_breaker is not None:
                breaker = self._circuit_breaker.get_breaker(provider.name)
                breaker.state.on_success()
            return result
