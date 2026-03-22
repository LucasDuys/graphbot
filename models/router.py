"""Model router that selects models based on task complexity.

Supports multi-provider rotation: try primary, on rate limit or error fall
back to the next provider, skip providers whose circuit breaker is open.

Also supports cascade mode: try the cheapest model first, escalate to more
expensive models if the result quality is below the confidence threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core_gb.confidence import ConfidenceEstimator
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

DEFAULT_CASCADE_CHAIN: list[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-pro-preview",
]

_MIN_COMPLEXITY = 1
_MAX_COMPLEXITY = 5


DEFAULT_COMPLEXITY_MULTIPLIERS: dict[int, float] = {
    1: 1.0,
    2: 1.5,
    3: 2.0,
    4: 3.0,
    5: 4.0,
}


@dataclass
class CascadeConfig:
    """Configuration for cascade mode routing.

    Attributes:
        chain: Ordered list of model names from cheapest to most expensive.
        confidence_threshold: Minimum confidence score (0.0-1.0) to accept a
            result without escalating to the next model.
        max_attempts: Maximum number of models to try in the chain.
        base_tokens: Base token budget for leaf prompts. Multiplied by the
            complexity multiplier to produce the final max_tokens directive.
        complexity_multipliers: Mapping from complexity level (1-5) to a
            multiplier applied to base_tokens.  Higher complexity tasks
            receive larger token budgets.
    """

    chain: list[str] | None = None
    confidence_threshold: float = 0.7
    max_attempts: int = 3
    base_tokens: int = 256
    complexity_multipliers: dict[int, float] | None = None

    def __post_init__(self) -> None:
        if self.chain is None:
            self.chain = list(DEFAULT_CASCADE_CHAIN)
        if self.complexity_multipliers is None:
            self.complexity_multipliers = dict(DEFAULT_COMPLEXITY_MULTIPLIERS)

    @property
    def effective_max_attempts(self) -> int:
        """Return max_attempts clamped to chain length."""
        return min(self.max_attempts, len(self.chain))  # type: ignore[arg-type]


@dataclass(frozen=True)
class CascadeResult(CompletionResult):
    """Result from cascade routing with confidence metadata.

    Extends CompletionResult with information about the cascade process:
    how many models were tried, whether escalation occurred, and the
    estimated confidence score of the final result.
    """

    confidence: float = 0.0
    attempts: int = 1
    escalated: bool = False


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
        cascade_config: CascadeConfig | None = None,
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
        self._cascade_config: CascadeConfig | None = cascade_config
        self._confidence_estimator: ConfidenceEstimator = ConfidenceEstimator()

    @property
    def rate_limiter(self) -> RateLimiterManager | None:
        return self._rate_limiter

    @property
    def circuit_breaker(self) -> CircuitBreakerManager | None:
        return self._circuit_breaker

    @property
    def cascade_config(self) -> CascadeConfig | None:
        return self._cascade_config

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

    async def route_cascade(
        self, task: TaskNode, messages: list[dict], **kwargs: object
    ) -> CascadeResult:
        """Try the cheapest model first, escalate if result quality is low.

        Cascade strategy:
        1. Compute a token budget based on task complexity.
        2. Start with the first (cheapest) model in the chain.
        3. Call the provider with that model, injecting max_tokens.
        4. Estimate confidence using ConfidenceEstimator.
        5. If confidence >= threshold, return the result.
        6. If confidence < threshold, try the next model in the chain.
        7. If a provider error occurs, skip to the next model.
        8. If all models are exhausted (or max_attempts reached), return the
           last successful result. If no model succeeded, raise
           AllProvidersExhaustedError.
        """
        config = self._cascade_config or CascadeConfig()
        chain = config.chain or DEFAULT_CASCADE_CHAIN
        max_tries = min(config.effective_max_attempts, len(chain))

        # Compute token budget directive from task complexity.
        token_budget = self._confidence_estimator.compute_token_budget(
            task.complexity, config,
        )

        # Merge max_tokens into kwargs (caller-supplied max_tokens takes
        # precedence over the computed budget).
        call_kwargs: dict[str, object] = dict(kwargs)
        if "max_tokens" not in call_kwargs:
            call_kwargs["max_tokens"] = token_budget

        errors: list[ProviderError] = []
        last_result: CascadeResult | None = None

        for attempt_idx in range(max_tries):
            model = chain[attempt_idx]

            try:
                result = await self._providers[0].complete(
                    messages, model, **call_kwargs
                )
            except ProviderError as exc:
                logger.debug(
                    "Cascade: model %s failed: %s, escalating",
                    model, exc,
                )
                errors.append(exc)
                continue
            except Exception as exc:
                logger.debug(
                    "Cascade: model %s raised unexpected error: %s, escalating",
                    model, exc,
                )
                errors.append(
                    ProviderError(str(exc), provider=self._providers[0].name, model=model)
                )
                continue

            confidence = self._confidence_estimator.estimate(result, task)
            cascade_result = CascadeResult(
                content=result.content,
                model=result.model,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=result.latency_ms,
                cost=result.cost,
                logprobs=result.logprobs,
                confidence=confidence,
                attempts=attempt_idx + 1,
                escalated=attempt_idx > 0,
            )
            last_result = cascade_result

            if confidence >= config.confidence_threshold:
                logger.debug(
                    "Cascade: model %s accepted (confidence=%.2f, threshold=%.2f)",
                    model, confidence, config.confidence_threshold,
                )
                return cascade_result

            logger.debug(
                "Cascade: model %s below threshold (confidence=%.2f, threshold=%.2f), escalating",
                model, confidence, config.confidence_threshold,
            )

        # Return the last successful result even if below threshold.
        if last_result is not None:
            return last_result

        # All models in chain produced errors.
        raise AllProvidersExhaustedError(errors)

    @staticmethod
    def _estimate_confidence(result: CompletionResult) -> float:
        """Estimate confidence of a completion result.

        Delegates to ConfidenceEstimator with a minimal TaskNode for
        backward compatibility.  Callers that have access to the full
        TaskNode should use ``ConfidenceEstimator.estimate()`` directly.

        Returns a float between 0.0 and 1.0.
        """
        from core_gb.types import TaskNode as _TN

        estimator = ConfidenceEstimator()
        stub_task = _TN(id="_static", description="")
        return estimator.estimate(result, stub_task)
