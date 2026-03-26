"""Tests for the SmartModelRouter with domain-based selection and cost budgets."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.base import ModelProvider
from models.errors import AllProvidersExhaustedError, ProviderError
from models.smart_router import (
    DEFAULT_COST_THRESHOLD,
    MODEL_GEMINI_FLASH,
    MODEL_LLAMA_8B,
    MODEL_LLAMA_70B,
    MODEL_QWEN3_32B,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
    DailyCostTracker,
    ModelSelection,
    SmartModelRouter,
    select_model,
)


class FakeProvider(ModelProvider):
    """Minimal mock provider for smart router tests."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self._mock_complete = AsyncMock()

    @property
    def name(self) -> str:
        return self._name

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


def _make_task(
    complexity: int = 1,
    domain: Domain = Domain.SYNTHESIS,
) -> TaskNode:
    return TaskNode(
        id="t1",
        description="test task",
        is_atomic=True,
        domain=domain,
        complexity=complexity,
        status=TaskStatus.READY,
    )


def _make_result(model: str = "some-model", cost: float = 0.0) -> CompletionResult:
    return CompletionResult(
        content="hello",
        model=model,
        tokens_in=10,
        tokens_out=5,
        latency_ms=42.0,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# select_model() unit tests
# ---------------------------------------------------------------------------


class TestSelectModel:
    """Test the pure function that maps (domain, complexity) to (provider, model)."""

    def test_simple_task_routes_to_groq_8b(self) -> None:
        """Complexity 1-2 tasks should route to Groq for fastest TTFT."""
        sel = select_model(Domain.SYNTHESIS, 1)
        assert sel.provider == PROVIDER_GROQ
        assert sel.model == MODEL_LLAMA_8B

    def test_simple_task_complexity_2_routes_to_groq(self) -> None:
        sel = select_model(Domain.SYSTEM, 2)
        assert sel.provider == PROVIDER_GROQ
        assert sel.model == MODEL_LLAMA_8B

    def test_hard_synthesis_routes_to_70b(self) -> None:
        """Complexity 3+ synthesis tasks should use 70B for reasoning."""
        sel = select_model(Domain.SYNTHESIS, 3)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_LLAMA_70B

    def test_code_task_routes_to_qwen3_32b(self) -> None:
        """Code tasks with complexity >= 3 should use Qwen3 32B."""
        sel = select_model(Domain.CODE, 3)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_QWEN3_32B

    def test_code_task_high_complexity_routes_to_gemini(self) -> None:
        """Complexity 4-5 always routes to Gemini Flash regardless of domain."""
        sel = select_model(Domain.CODE, 4)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_GEMINI_FLASH

    def test_creative_high_complexity_routes_to_gemini_flash(self) -> None:
        """Complexity 4+ tasks should use Gemini Flash for best reasoning."""
        sel = select_model(Domain.SYNTHESIS, 4)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_GEMINI_FLASH

        sel = select_model(Domain.SYNTHESIS, 5)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_GEMINI_FLASH

    def test_web_domain_routes_to_70b(self) -> None:
        """WEB domain needs reasoning for tool use, even at low complexity."""
        sel = select_model(Domain.WEB, 1)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_LLAMA_70B

    def test_file_domain_routes_to_70b(self) -> None:
        """FILE domain needs reasoning for tool use."""
        sel = select_model(Domain.FILE, 2)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_LLAMA_70B

    def test_browser_domain_routes_to_70b(self) -> None:
        """BROWSER domain needs reasoning for tool use."""
        sel = select_model(Domain.BROWSER, 1)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_LLAMA_70B

    def test_complexity_clamped_below(self) -> None:
        """Complexity below 1 is clamped to 1."""
        sel = select_model(Domain.SYNTHESIS, 0)
        assert sel.provider == PROVIDER_GROQ
        assert sel.model == MODEL_LLAMA_8B

    def test_complexity_clamped_above(self) -> None:
        """Complexity above 5 is clamped to 5."""
        sel = select_model(Domain.SYNTHESIS, 10)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_GEMINI_FLASH


# ---------------------------------------------------------------------------
# DailyCostTracker tests
# ---------------------------------------------------------------------------


class TestDailyCostTracker:
    def test_initial_state(self) -> None:
        tracker = DailyCostTracker()
        assert tracker.total == 0.0
        assert not tracker.should_downgrade()

    def test_record_cost_accumulates(self) -> None:
        tracker = DailyCostTracker(threshold=0.10)
        tracker.record_cost(0.03)
        tracker.record_cost(0.04)
        assert abs(tracker.total - 0.07) < 1e-9
        assert not tracker.should_downgrade()

    def test_should_downgrade_when_over_threshold(self) -> None:
        tracker = DailyCostTracker(threshold=0.10)
        tracker.record_cost(0.11)
        assert tracker.should_downgrade()

    def test_should_downgrade_when_exactly_at_threshold(self) -> None:
        tracker = DailyCostTracker(threshold=0.10)
        tracker.record_cost(0.10)
        assert tracker.should_downgrade()

    def test_reset_clears_total(self) -> None:
        tracker = DailyCostTracker(threshold=0.05)
        tracker.record_cost(0.06)
        assert tracker.should_downgrade()
        tracker.reset()
        assert tracker.total == 0.0
        assert not tracker.should_downgrade()

    def test_auto_reset_on_date_change(self) -> None:
        tracker = DailyCostTracker(threshold=0.10)
        tracker.record_cost(0.15)
        assert tracker.should_downgrade()

        # Simulate date change
        tomorrow = date(2099, 1, 1)
        with patch("models.smart_router.date") as mock_date:
            mock_date.today.return_value = tomorrow
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            # Accessing total triggers _maybe_reset
            assert tracker.total == 0.0
            assert not tracker.should_downgrade()


# ---------------------------------------------------------------------------
# SmartModelRouter tests
# ---------------------------------------------------------------------------


class TestSmartModelRouter:
    async def test_simple_task_routes_to_groq(self) -> None:
        """Simple task (complexity=1, synthesis) should use Groq 8B."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "hi"}])

        assert result.model == MODEL_LLAMA_8B
        groq._mock_complete.assert_awaited_once()
        openrouter._mock_complete.assert_not_awaited()

    async def test_hard_task_routes_to_70b(self) -> None:
        """Hard task (complexity=3, synthesis) should use OpenRouter 70B."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")
        openrouter._mock_complete.return_value = _make_result(MODEL_LLAMA_70B)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=3, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "explain quantum"}])

        assert result.model == MODEL_LLAMA_70B
        openrouter._mock_complete.assert_awaited_once()
        groq._mock_complete.assert_not_awaited()

    async def test_code_task_routes_to_qwen3(self) -> None:
        """Code task (complexity=3) should use Qwen3 32B on OpenRouter."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")
        openrouter._mock_complete.return_value = _make_result(MODEL_QWEN3_32B)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=3, domain=Domain.CODE)
        result = await router.route(task, [{"role": "user", "content": "write a function"}])

        assert result.model == MODEL_QWEN3_32B
        openrouter._mock_complete.assert_awaited_once()

    async def test_cost_budget_enforcement_downgrades(self) -> None:
        """When daily cost exceeds threshold, all tasks downgrade to Groq 8B."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        tracker = DailyCostTracker(threshold=0.05)
        tracker.record_cost(0.06)  # Over budget

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
            cost_tracker=tracker,
        )

        # Even a hard task should downgrade to Groq 8B
        task = _make_task(complexity=5, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "complex reasoning"}])

        assert result.model == MODEL_LLAMA_8B
        groq._mock_complete.assert_awaited_once()
        openrouter._mock_complete.assert_not_awaited()

    async def test_fallback_on_preferred_provider_failure(self) -> None:
        """If the preferred provider fails, the router falls back to other providers."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")

        # Groq fails, OpenRouter succeeds
        groq._mock_complete.side_effect = ProviderError(
            "service unavailable", provider="groq", model=MODEL_LLAMA_8B,
        )
        openrouter._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "hello"}])

        # Should have fallen back to openrouter
        assert result.model == MODEL_LLAMA_8B
        groq._mock_complete.assert_awaited_once()
        openrouter._mock_complete.assert_awaited_once()

    async def test_all_providers_fail_raises_exhausted(self) -> None:
        """If all providers fail, AllProvidersExhaustedError is raised."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")

        groq._mock_complete.side_effect = ProviderError(
            "groq down", provider="groq", model=MODEL_LLAMA_8B,
        )
        openrouter._mock_complete.side_effect = ProviderError(
            "openrouter down", provider="openrouter", model=MODEL_LLAMA_8B,
        )

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.route(task, [{"role": "user", "content": "test"}])

        assert len(exc_info.value.errors) == 2

    async def test_cost_is_tracked_after_successful_call(self) -> None:
        """Successful calls should record their cost in the tracker."""
        groq = FakeProvider(name="groq")
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_8B, cost=0.002)

        tracker = DailyCostTracker(threshold=1.0)
        router = SmartModelRouter(
            providers={"groq": groq},
            cost_tracker=tracker,
        )

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        await router.route(task, [{"role": "user", "content": "hi"}])

        assert abs(tracker.total - 0.002) < 1e-9

    async def test_no_providers_raises_value_error(self) -> None:
        """Empty providers dict should raise ValueError."""
        with pytest.raises(ValueError, match="At least one provider"):
            SmartModelRouter(providers={})

    async def test_fallback_on_unexpected_exception(self) -> None:
        """Non-ProviderError exceptions should also trigger fallback."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")

        groq._mock_complete.side_effect = RuntimeError("unexpected crash")
        openrouter._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "hi"}])

        assert result.model == MODEL_LLAMA_8B
        openrouter._mock_complete.assert_awaited_once()

    async def test_kwargs_forwarded_to_provider(self) -> None:
        """Extra kwargs should be forwarded to the provider's complete()."""
        groq = FakeProvider(name="groq")
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        router = SmartModelRouter(providers={"groq": groq})

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        await router.route(
            task,
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
        )

        groq._mock_complete.assert_awaited_once_with(
            [{"role": "user", "content": "hi"}],
            MODEL_LLAMA_8B,
            temperature=0.5,
        )
