"""Model router that selects models based on task complexity."""

from __future__ import annotations

from typing import Any

from core_gb.types import CompletionResult, TaskNode
from models.base import ModelProvider

DEFAULT_MODEL_MAP: dict[int, str] = {
    1: "meta-llama/llama-3.1-8b-instruct",
    2: "meta-llama/llama-4-scout-17b-16e-instruct",
    3: "meta-llama/llama-3.3-70b-versatile",
    4: "google/gemini-2.5-pro-preview",
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
    ) -> None:
        # Store as list for future multi-provider support (Phase 3).
        self._providers: list[ModelProvider] = [provider]
        self._model_map: dict[int, str] = model_map or DEFAULT_MODEL_MAP

        # Placeholder hooks for Phase 3: rate limiter / circuit breaker.
        self._rate_limiter: Any = None
        self._circuit_breaker: Any = None

    @property
    def rate_limiter(self) -> Any:
        """Rate limiter hook (not implemented -- Phase 3)."""
        return self._rate_limiter

    @property
    def circuit_breaker(self) -> Any:
        """Circuit breaker hook (not implemented -- Phase 3)."""
        return self._circuit_breaker

    def get_model_for_complexity(self, complexity: int) -> str:
        """Return the model name for a given complexity level, clamped to [1, 5]."""
        clamped = max(_MIN_COMPLEXITY, min(_MAX_COMPLEXITY, complexity))
        return self._model_map[clamped]

    async def route(self, task: TaskNode, messages: list[dict]) -> CompletionResult:
        """Select a model based on task complexity and run completion."""
        model = self.get_model_for_complexity(task.complexity)
        provider = self._providers[0]
        return await provider.complete(messages, model)
