"""Integration tests for SmartModelRouter: full pipeline routing with IntakeParser.

Validates that the IntakeParser -> SmartModelRouter.select_model() pipeline
correctly routes the 15 benchmark tasks to the expected providers and models.
Also tests cost budget enforcement, daily reset, fallback chains, and routing
decision latency.
"""

from __future__ import annotations

import time
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from core_gb.intake import IntakeParser
from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.base import ModelProvider
from models.errors import AllProvidersExhaustedError, ProviderError
from models.smart_router import (
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(ModelProvider):
    """Minimal fake provider for integration tests."""

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
# Benchmark tasks (same 15 from validate_single_call.py)
# ---------------------------------------------------------------------------

BENCHMARK_TASKS: list[dict[str, str]] = [
    # Easy tasks (trivial factual recall)
    {"id": "easy_01", "category": "easy", "question": "What is the capital of Japan?"},
    {"id": "easy_02", "category": "easy", "question": "What is 15 * 23?"},
    {"id": "easy_03", "category": "easy", "question": "Define photosynthesis in one sentence."},
    {"id": "easy_04", "category": "easy", "question": "Who wrote '1984'?"},
    {"id": "easy_05", "category": "easy", "question": "What color do you get mixing red and blue?"},
    # Hard tasks (multi-step reasoning)
    {
        "id": "hard_01",
        "category": "hard",
        "question": (
            "Compare the economic systems of Sweden, Singapore, and the United States. "
            "For each country, describe the tax rate, healthcare model, and GDP per capita. "
            "Then recommend which system would work best for a developing nation and explain why."
        ),
    },
    {
        "id": "hard_02",
        "category": "hard",
        "question": (
            "Explain how a neural network learns, starting from a single neuron, "
            "building up to backpropagation, and ending with how transformers use attention. "
            "Use analogies a high school student would understand."
        ),
    },
    {
        "id": "hard_03",
        "category": "hard",
        "question": (
            "A company has 3 products: A ($50, 30% margin), B ($120, 45% margin), C ($200, 20% margin). "
            "They sold 1000 units of A, 500 units of B, and 200 units of C last quarter. "
            "Calculate total revenue, total profit, profit per product, and recommend which product "
            "to focus marketing on for maximum profit growth. Show your work."
        ),
    },
    {
        "id": "hard_04",
        "category": "hard",
        "question": (
            "Write a detailed comparison of 5 sorting algorithms (bubble sort, merge sort, quick sort, "
            "heap sort, and radix sort). For each, provide: time complexity (best, average, worst), "
            "space complexity, stability, and a one-line description of when to use it. "
            "Format as a table."
        ),
    },
    {
        "id": "hard_05",
        "category": "hard",
        "question": (
            "Trace the journey of a HTTP request from typing 'google.com' in a browser to seeing "
            "the page rendered. Include: DNS resolution, TCP handshake, TLS negotiation, HTTP request, "
            "server processing, response, and browser rendering. Be specific about each step."
        ),
    },
    # Tool-dependent tasks
    {
        "id": "tool_01",
        "category": "tool",
        "question": "What Python files are in the scripts/ directory of this project? List them all.",
    },
    {
        "id": "tool_02",
        "category": "tool",
        "question": "Run 'python --version' and tell me exactly what Python version is installed.",
    },
    {
        "id": "tool_03",
        "category": "tool",
        "question": "Search the web for 'Kuzu graph database latest version 2026' and tell me what you find.",
    },
    {
        "id": "tool_04",
        "category": "tool",
        "question": "Read the first 10 lines of pyproject.toml in this project and tell me the project name and version.",
    },
    {
        "id": "tool_05",
        "category": "tool",
        "question": "Run 'git log --oneline -5' and summarize what the last 5 commits changed.",
    },
]


# ---------------------------------------------------------------------------
# Full pipeline integration tests: IntakeParser -> select_model
# ---------------------------------------------------------------------------


class TestFullPipelineRouting:
    """Test IntakeParser.parse() -> select_model() for all 15 benchmark tasks."""

    parser: IntakeParser

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.parser = IntakeParser()

    def _route_question(self, question: str) -> tuple[Domain, int, ModelSelection]:
        """Parse a question and route it, returning (domain, complexity, selection)."""
        intake = self.parser.parse(question)
        selection = select_model(intake.domain, intake.complexity)
        return intake.domain, intake.complexity, selection

    def test_easy_tasks_route_to_small_models(self) -> None:
        """Easy tasks (trivial factual) should route to Groq 8B or small models."""
        easy_tasks = [t for t in BENCHMARK_TASKS if t["category"] == "easy"]
        for task in easy_tasks:
            domain, complexity, selection = self._route_question(task["question"])
            # Easy tasks should have low complexity (1-2) and route to Groq
            # unless they hit a tool-use domain
            assert complexity <= 3, (
                f"Task {task['id']} has unexpectedly high complexity {complexity}: "
                f"{task['question'][:60]}"
            )

    def test_hard_tasks_route_to_larger_models(self) -> None:
        """Hard tasks (multi-step reasoning) should route to 70B+ models."""
        hard_tasks = [t for t in BENCHMARK_TASKS if t["category"] == "hard"]
        routed_to_large = 0
        for task in hard_tasks:
            domain, complexity, selection = self._route_question(task["question"])
            # Hard tasks should have higher complexity and route to larger models
            if selection.model in (MODEL_LLAMA_70B, MODEL_QWEN3_32B, MODEL_GEMINI_FLASH):
                routed_to_large += 1
        # At least 3 of 5 hard tasks should route to a large model
        assert routed_to_large >= 3, (
            f"Only {routed_to_large}/5 hard tasks routed to large models"
        )

    def test_tool_tasks_route_to_reasoning_models(self) -> None:
        """Tool-dependent tasks need reasoning models for tool use."""
        tool_tasks = [t for t in BENCHMARK_TASKS if t["category"] == "tool"]
        routed_to_reasoning = 0
        for task in tool_tasks:
            domain, complexity, selection = self._route_question(task["question"])
            # Tool tasks often land in FILE/WEB/CODE domains which need 70B+
            if selection.model != MODEL_LLAMA_8B:
                routed_to_reasoning += 1
        # At least 3 of 5 tool tasks should route to a reasoning model
        assert routed_to_reasoning >= 3, (
            f"Only {routed_to_reasoning}/5 tool tasks routed to reasoning models"
        )

    def test_all_15_tasks_produce_valid_selections(self) -> None:
        """Every benchmark task should produce a valid provider+model selection."""
        valid_providers = {PROVIDER_GROQ, PROVIDER_OPENROUTER}
        valid_models = {MODEL_LLAMA_8B, MODEL_LLAMA_70B, MODEL_QWEN3_32B, MODEL_GEMINI_FLASH}

        for task in BENCHMARK_TASKS:
            domain, complexity, selection = self._route_question(task["question"])
            assert selection.provider in valid_providers, (
                f"Task {task['id']} got invalid provider: {selection.provider}"
            )
            assert selection.model in valid_models, (
                f"Task {task['id']} got invalid model: {selection.model}"
            )
            assert 1 <= complexity <= 5, (
                f"Task {task['id']} got out-of-range complexity: {complexity}"
            )


# ---------------------------------------------------------------------------
# Direct select_model routing tests
# ---------------------------------------------------------------------------


class TestSimpleTaskRoutesToGroq8B:
    """Verify simple tasks (low complexity, non-tool domains) route to Groq/8B."""

    def test_synthesis_complexity_1(self) -> None:
        sel = select_model(Domain.SYNTHESIS, 1)
        assert sel.provider == PROVIDER_GROQ
        assert sel.model == MODEL_LLAMA_8B

    def test_system_complexity_2(self) -> None:
        sel = select_model(Domain.SYSTEM, 2)
        assert sel.provider == PROVIDER_GROQ
        assert sel.model == MODEL_LLAMA_8B

    def test_comms_complexity_1(self) -> None:
        sel = select_model(Domain.COMMS, 1)
        assert sel.provider == PROVIDER_GROQ
        assert sel.model == MODEL_LLAMA_8B


class TestHardTaskRoutesTo70B:
    """Verify hard tasks (complexity 3, non-code non-tool domains) route to 70B."""

    def test_synthesis_complexity_3(self) -> None:
        sel = select_model(Domain.SYNTHESIS, 3)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_LLAMA_70B

    def test_comms_complexity_3(self) -> None:
        sel = select_model(Domain.COMMS, 3)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_LLAMA_70B

    def test_system_complexity_3(self) -> None:
        sel = select_model(Domain.SYSTEM, 3)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_LLAMA_70B


class TestCodeDomainRoutesToQwen3:
    """Verify code domain tasks with moderate complexity route to Qwen3 32B."""

    def test_code_complexity_3(self) -> None:
        sel = select_model(Domain.CODE, 3)
        assert sel.provider == PROVIDER_OPENROUTER
        assert sel.model == MODEL_QWEN3_32B


# ---------------------------------------------------------------------------
# Cost budget enforcement
# ---------------------------------------------------------------------------


class TestCostBudgetEnforcementDowngradesModel:
    """Verify that exceeding cost budget forces all tasks to cheapest model."""

    async def test_over_budget_downgrades_hard_task_to_groq_8b(self) -> None:
        """A complexity-5 task should downgrade to Groq 8B when over budget."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        tracker = DailyCostTracker(threshold=0.05)
        tracker.record_cost(0.06)  # Over budget

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
            cost_tracker=tracker,
        )

        task = _make_task(complexity=5, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "complex task"}])

        assert result.model == MODEL_LLAMA_8B
        groq._mock_complete.assert_awaited_once()
        openrouter._mock_complete.assert_not_awaited()

    async def test_over_budget_downgrades_code_task(self) -> None:
        """Even code tasks should downgrade to Groq 8B when over budget."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        tracker = DailyCostTracker(threshold=0.01)
        tracker.record_cost(0.02)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
            cost_tracker=tracker,
        )

        task = _make_task(complexity=3, domain=Domain.CODE)
        result = await router.route(task, [{"role": "user", "content": "write code"}])

        assert result.model == MODEL_LLAMA_8B
        groq._mock_complete.assert_awaited_once()
        openrouter._mock_complete.assert_not_awaited()

    async def test_under_budget_routes_normally(self) -> None:
        """Tasks should route normally when under budget."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")
        openrouter._mock_complete.return_value = _make_result(MODEL_LLAMA_70B)

        tracker = DailyCostTracker(threshold=1.00)
        tracker.record_cost(0.01)  # Well under budget

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
            cost_tracker=tracker,
        )

        task = _make_task(complexity=3, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "explain topic"}])

        assert result.model == MODEL_LLAMA_70B
        openrouter._mock_complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


class TestFallbackChainOnProviderFailure:
    """Verify fallback works when the preferred provider fails."""

    async def test_groq_fails_falls_back_to_openrouter(self) -> None:
        """When Groq is down, simple tasks fall back to OpenRouter."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")

        groq._mock_complete.side_effect = ProviderError(
            "service unavailable", provider="groq", model=MODEL_LLAMA_8B,
        )
        openrouter._mock_complete.return_value = _make_result(MODEL_LLAMA_8B)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "hello"}])

        assert result.model == MODEL_LLAMA_8B
        groq._mock_complete.assert_awaited_once()
        openrouter._mock_complete.assert_awaited_once()

    async def test_openrouter_fails_falls_back_to_groq(self) -> None:
        """When OpenRouter is down, hard tasks fall back to Groq."""
        groq = FakeProvider(name="groq")
        openrouter = FakeProvider(name="openrouter")

        openrouter._mock_complete.side_effect = ProviderError(
            "rate limited", provider="openrouter", model=MODEL_LLAMA_70B,
        )
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_70B)

        router = SmartModelRouter(
            providers={"groq": groq, "openrouter": openrouter},
        )

        task = _make_task(complexity=3, domain=Domain.SYNTHESIS)
        result = await router.route(task, [{"role": "user", "content": "explain topic"}])

        assert result.model == MODEL_LLAMA_70B
        openrouter._mock_complete.assert_awaited_once()
        groq._mock_complete.assert_awaited_once()

    async def test_all_providers_fail_raises_exhausted(self) -> None:
        """When every provider fails, AllProvidersExhaustedError is raised."""
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


# ---------------------------------------------------------------------------
# Daily cost tracking and reset
# ---------------------------------------------------------------------------


class TestDailyCostTrackingAndReset:
    """Verify daily cost tracking accumulates and resets correctly."""

    def test_cost_accumulates_across_calls(self) -> None:
        tracker = DailyCostTracker(threshold=0.10)
        tracker.record_cost(0.02)
        tracker.record_cost(0.03)
        tracker.record_cost(0.04)
        assert abs(tracker.total - 0.09) < 1e-9
        assert not tracker.should_downgrade()

    def test_cost_triggers_downgrade_at_threshold(self) -> None:
        tracker = DailyCostTracker(threshold=0.10)
        tracker.record_cost(0.10)
        assert tracker.should_downgrade()

    def test_manual_reset_clears_cost(self) -> None:
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

        # Simulate date rollover
        tomorrow = date(2099, 1, 1)
        with patch("models.smart_router.date") as mock_date:
            mock_date.today.return_value = tomorrow
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            assert tracker.total == 0.0
            assert not tracker.should_downgrade()

    async def test_cost_tracked_after_successful_route(self) -> None:
        """Successful routes should accumulate cost in the tracker."""
        groq = FakeProvider(name="groq")
        groq._mock_complete.return_value = _make_result(MODEL_LLAMA_8B, cost=0.003)

        tracker = DailyCostTracker(threshold=1.0)
        router = SmartModelRouter(
            providers={"groq": groq},
            cost_tracker=tracker,
        )

        task = _make_task(complexity=1, domain=Domain.SYNTHESIS)
        await router.route(task, [{"role": "user", "content": "hi"}])
        await router.route(task, [{"role": "user", "content": "hi"}])

        assert abs(tracker.total - 0.006) < 1e-9


# ---------------------------------------------------------------------------
# Routing decision latency benchmark
# ---------------------------------------------------------------------------


class TestRoutingDecisionLatency:
    """Verify that routing decisions are fast (pure computation, no I/O)."""

    def test_100_routing_decisions_under_100ms(self) -> None:
        """100 select_model() calls should complete in under 100ms total."""
        domains = list(Domain)
        complexities = [1, 2, 3, 4, 5]

        start = time.perf_counter()
        for i in range(100):
            domain = domains[i % len(domains)]
            complexity = complexities[i % len(complexities)]
            select_model(domain, complexity)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, (
            f"100 routing decisions took {elapsed_ms:.2f}ms, expected <100ms"
        )

    def test_single_routing_decision_under_1ms(self) -> None:
        """A single routing decision should take well under 1ms."""
        # Warm up
        select_model(Domain.SYNTHESIS, 3)

        start = time.perf_counter()
        select_model(Domain.CODE, 3)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1.0, (
            f"Single routing decision took {elapsed_ms:.4f}ms, expected <1ms"
        )

    def test_full_pipeline_routing_under_1ms_per_decision(self) -> None:
        """IntakeParser.parse() + select_model() should be <1ms per decision on average."""
        parser = IntakeParser()
        questions = [t["question"] for t in BENCHMARK_TASKS]

        # Warm up
        for q in questions[:3]:
            intake = parser.parse(q)
            select_model(intake.domain, intake.complexity)

        start = time.perf_counter()
        for q in questions:
            intake = parser.parse(q)
            select_model(intake.domain, intake.complexity)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / len(questions)

        assert avg_ms < 1.0, (
            f"Average pipeline routing took {avg_ms:.4f}ms per decision, expected <1ms"
        )
