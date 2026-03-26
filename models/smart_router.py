"""Smart model router with complexity-based selection, latency-aware routing, and cost budget enforcement.

Routes tasks to the optimal provider+model combination based on domain and complexity:
- Easy factual tasks -> Groq (fastest TTFT, free tier)
- Hard reasoning tasks -> OpenRouter 70B
- Code tasks -> Qwen3 32B on OpenRouter
- Creative/complex tasks -> Gemini Flash on OpenRouter

Tracks daily cost and downgrades to cheaper models when the budget threshold is exceeded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

from core_gb.types import CompletionResult, Domain, TaskNode
from models.base import ModelProvider
from models.errors import AllProvidersExhaustedError, ProviderError
from models.router import ModelRouter

logger = logging.getLogger(__name__)

# Provider constants
PROVIDER_GROQ = "groq"
PROVIDER_OPENROUTER = "openrouter"

# Model constants
MODEL_LLAMA_8B = "llama-3.1-8b-instruct"
MODEL_LLAMA_70B = "meta-llama/llama-3.3-70b-instruct"
MODEL_QWEN3_32B = "qwen/qwen3-32b"
MODEL_GEMINI_FLASH = "google/gemini-2.5-flash-preview"

# Default daily cost threshold in USD. When cumulative daily cost exceeds
# this value the router downgrades all requests to the cheapest model.
DEFAULT_COST_THRESHOLD: float = 0.10


@dataclass
class ModelSelection:
    """A resolved provider + model pair."""

    provider: str
    model: str


# ---------------------------------------------------------------------------
# Model selection matrix
# ---------------------------------------------------------------------------
# Each entry maps (domain_set, complexity_range) to (provider, model).
# Entries are evaluated top-to-bottom; first match wins.  The ``domains``
# value ``None`` means "any domain".
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SelectionRule:
    """A single rule in the model selection matrix."""

    domains: frozenset[Domain] | None  # None = matches any domain
    min_complexity: int
    max_complexity: int
    provider: str
    model: str


MODEL_SELECTION_MATRIX: tuple[_SelectionRule, ...] = (
    # High-complexity creative/general tasks -> Gemini Flash (strongest reasoning)
    _SelectionRule(
        domains=None,
        min_complexity=4, max_complexity=5,
        provider=PROVIDER_OPENROUTER, model=MODEL_GEMINI_FLASH,
    ),
    # Tool-use domains (WEB, FILE, BROWSER) need reasoning regardless of complexity
    _SelectionRule(
        domains=frozenset({Domain.WEB, Domain.FILE, Domain.BROWSER}),
        min_complexity=1, max_complexity=5,
        provider=PROVIDER_OPENROUTER, model=MODEL_LLAMA_70B,
    ),
    # Code tasks with moderate+ complexity -> Qwen3 32B
    _SelectionRule(
        domains=frozenset({Domain.CODE}),
        min_complexity=3, max_complexity=5,
        provider=PROVIDER_OPENROUTER, model=MODEL_QWEN3_32B,
    ),
    # Synthesis / reasoning tasks with moderate+ complexity -> 70B
    _SelectionRule(
        domains=frozenset({Domain.SYNTHESIS, Domain.COMMS, Domain.SYSTEM}),
        min_complexity=3, max_complexity=5,
        provider=PROVIDER_OPENROUTER, model=MODEL_LLAMA_70B,
    ),
    # Easy tasks (complexity 1-2, any domain not matched above) -> Groq 8B (fastest)
    _SelectionRule(
        domains=None,
        min_complexity=1, max_complexity=2,
        provider=PROVIDER_GROQ, model=MODEL_LLAMA_8B,
    ),
)

# Cheap fallback used when cost budget is exceeded
_BUDGET_FALLBACK = ModelSelection(provider=PROVIDER_GROQ, model=MODEL_LLAMA_8B)


class DailyCostTracker:
    """Tracks cumulative daily LLM cost in memory.

    Automatically resets when the date changes. Thread-safe for single-threaded
    async usage (no locks needed with cooperative multitasking).
    """

    def __init__(self, threshold: float = DEFAULT_COST_THRESHOLD) -> None:
        self._threshold = threshold
        self._total: float = 0.0
        self._date: date = date.today()

    @property
    def total(self) -> float:
        """Current cumulative cost for today."""
        self._maybe_reset()
        return self._total

    @property
    def threshold(self) -> float:
        """Daily cost threshold."""
        return self._threshold

    def record_cost(self, amount: float) -> None:
        """Record a cost amount, resetting if the date has changed."""
        self._maybe_reset()
        self._total += amount

    def should_downgrade(self) -> bool:
        """Return True if the daily cost has exceeded the threshold."""
        self._maybe_reset()
        return self._total >= self._threshold

    def reset(self) -> None:
        """Manually reset the tracker (e.g. for testing)."""
        self._total = 0.0
        self._date = date.today()

    def _maybe_reset(self) -> None:
        """Auto-reset if the calendar day has changed."""
        today = date.today()
        if today != self._date:
            self._total = 0.0
            self._date = today


def select_model(domain: Domain, complexity: int) -> ModelSelection:
    """Select the best provider + model for a given domain and complexity.

    Evaluates the MODEL_SELECTION_MATRIX top-to-bottom and returns the
    first matching rule. Falls back to Groq 8B if no rule matches (should
    not happen with a well-defined matrix).

    Args:
        domain: The task domain (e.g. Domain.CODE, Domain.SYNTHESIS).
        complexity: Task complexity level (1-5).

    Returns:
        A ModelSelection with provider name and model ID.
    """
    clamped = max(1, min(5, complexity))

    for rule in MODEL_SELECTION_MATRIX:
        # Check complexity range
        if not (rule.min_complexity <= clamped <= rule.max_complexity):
            continue
        # Check domain match (None = any)
        if rule.domains is not None and domain not in rule.domains:
            continue
        return ModelSelection(provider=rule.provider, model=rule.model)

    # Fallback: should not be reached with a complete matrix
    return ModelSelection(provider=PROVIDER_GROQ, model=MODEL_LLAMA_8B)


class SmartModelRouter:
    """Complexity- and domain-aware model router with cost budget enforcement.

    Wraps a dict of named ModelProvider instances and routes tasks to the
    optimal provider based on the MODEL_SELECTION_MATRIX.  When the daily
    cost budget is exceeded, all requests are downgraded to the cheapest
    model (Groq 8B).

    Fallback chain: if the preferred provider fails, the router tries the
    remaining providers in order.

    This class is designed to be used as a drop-in replacement wherever a
    ModelRouter is used -- the ``route()`` method has the same signature.
    """

    def __init__(
        self,
        providers: dict[str, ModelProvider],
        cost_tracker: DailyCostTracker | None = None,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider must be supplied")
        self._providers = providers
        self._cost_tracker = cost_tracker or DailyCostTracker()

    @property
    def cost_tracker(self) -> DailyCostTracker:
        """Access the daily cost tracker."""
        return self._cost_tracker

    def _get_provider(self, name: str) -> ModelProvider | None:
        """Look up a provider by name."""
        return self._providers.get(name)

    def _fallback_providers(self, exclude: str) -> list[ModelProvider]:
        """Return all providers except the one named ``exclude``."""
        return [p for name, p in self._providers.items() if name != exclude]

    async def route(
        self,
        task: TaskNode,
        messages: list[dict],
        **kwargs: object,
    ) -> CompletionResult:
        """Route a task to the best provider + model with fallback.

        Steps:
        1. Check cost budget -- downgrade to cheapest model if over threshold.
        2. Select model based on domain + complexity via MODEL_SELECTION_MATRIX.
        3. Try the preferred provider first.
        4. On failure, try remaining providers with the same model as fallback.
        5. Raise AllProvidersExhaustedError if all providers fail.

        Args:
            task: The TaskNode containing domain and complexity metadata.
            messages: Chat messages to send to the model.
            **kwargs: Extra arguments forwarded to the provider's complete().

        Returns:
            A CompletionResult from the successful provider.

        Raises:
            AllProvidersExhaustedError: When every provider has failed.
        """
        # Step 1: Cost budget check
        if self._cost_tracker.should_downgrade():
            selection = _BUDGET_FALLBACK
            logger.info(
                "Cost budget exceeded (%.4f >= %.4f), downgrading to %s/%s",
                self._cost_tracker.total,
                self._cost_tracker.threshold,
                selection.provider,
                selection.model,
            )
        else:
            # Step 2: Smart model selection
            selection = select_model(task.domain, task.complexity)

        logger.debug(
            "Smart route: domain=%s complexity=%d -> %s/%s",
            task.domain.value, task.complexity,
            selection.provider, selection.model,
        )

        # Step 3: Try preferred provider
        errors: list[ProviderError] = []
        preferred = self._get_provider(selection.provider)

        if preferred is not None:
            try:
                result = await preferred.complete(
                    messages, selection.model, **kwargs,
                )
                self._cost_tracker.record_cost(result.cost)
                return result
            except ProviderError as exc:
                logger.debug(
                    "Preferred provider %s failed: %s, trying fallbacks",
                    selection.provider, exc,
                )
                errors.append(exc)
            except Exception as exc:
                logger.debug(
                    "Preferred provider %s raised unexpected error: %s",
                    selection.provider, exc,
                )
                errors.append(
                    ProviderError(
                        str(exc),
                        provider=selection.provider,
                        model=selection.model,
                    )
                )

        # Step 4: Fallback to other providers
        for fallback in self._fallback_providers(selection.provider):
            try:
                result = await fallback.complete(
                    messages, selection.model, **kwargs,
                )
                self._cost_tracker.record_cost(result.cost)
                return result
            except ProviderError as exc:
                logger.debug(
                    "Fallback provider %s failed: %s",
                    fallback.name, exc,
                )
                errors.append(exc)
            except Exception as exc:
                logger.debug(
                    "Fallback provider %s raised unexpected error: %s",
                    fallback.name, exc,
                )
                errors.append(
                    ProviderError(
                        str(exc),
                        provider=fallback.name,
                        model=selection.model,
                    )
                )

        # Step 5: All providers exhausted
        raise AllProvidersExhaustedError(errors)
