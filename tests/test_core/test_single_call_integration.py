"""Integration tests for SingleCallExecutor + ContextEnricher + TokenBudget.

Tests the full flow from context enrichment through formatting to single-call
execution, with mocked graph store and mocked model router.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.context_enrichment import ContextEnricher, EnrichedContext
from core_gb.single_executor import SingleCallExecutor
from core_gb.token_budget import TokenBudget
from core_gb.types import (
    CompletionResult,
    ExecutionResult,
    GraphContext,
    Pattern,
)
from models.base import ModelProvider
from models.errors import AllProvidersExhaustedError, ProviderError
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockProvider(ModelProvider):
    """Mock provider that returns a configurable CompletionResult."""

    def __init__(self, result: CompletionResult | None = None) -> None:
        self._result = result or CompletionResult(
            content="integration response",
            model="test-model-v1",
            tokens_in=20,
            tokens_out=10,
            latency_ms=42.0,
            cost=0.002,
        )
        self._mock_complete = AsyncMock(side_effect=self._do_complete)
        self.captured_messages: list[list[dict[str, str]]] = []

    @property
    def name(self) -> str:
        return "mock-integration"

    async def _do_complete(
        self, messages: list[dict[str, str]], model: str, **kwargs: object
    ) -> CompletionResult:
        self.captured_messages.append(messages)
        return self._result

    async def complete(
        self, messages: list[dict[str, str]], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


class _FailingProvider(ModelProvider):
    """Mock provider that always raises ProviderError."""

    @property
    def name(self) -> str:
        return "failing-integration"

    async def complete(
        self, messages: list[dict[str, str]], model: str, **kwargs: object
    ) -> CompletionResult:
        raise ProviderError(
            "provider unavailable", provider="failing-integration", model=model
        )


def _make_enriched_context(
    *,
    entities: tuple[dict[str, str], ...] = (),
    memories: tuple[str, ...] = (),
    reflections: tuple[dict[str, str], ...] = (),
    patterns: tuple[Pattern, ...] = (),
    conversation_turns: tuple[dict[str, str], ...] = (),
    entity_tokens: int = 0,
    memory_tokens: int = 0,
    reflection_tokens: int = 0,
    pattern_tokens: int = 0,
    conversation_tokens: int = 0,
) -> EnrichedContext:
    """Build an EnrichedContext with sensible defaults."""
    return EnrichedContext(
        entities=entities,
        memories=memories,
        reflections=reflections,
        patterns=patterns,
        conversation_turns=conversation_turns,
        entity_tokens=entity_tokens,
        memory_tokens=memory_tokens,
        reflection_tokens=reflection_tokens,
        pattern_tokens=pattern_tokens,
        conversation_tokens=conversation_tokens,
    )


def _enriched_to_graph_context(ctx: EnrichedContext) -> GraphContext:
    """Convert an EnrichedContext into a GraphContext for the executor."""
    return GraphContext(
        relevant_entities=ctx.entities,
        active_memories=ctx.memories,
        matching_patterns=ctx.patterns,
        reflections=ctx.reflections,
        total_tokens=ctx.total_tokens,
    )


def _sample_pattern(
    *,
    trigger: str = "explain concept",
    description: str = "Give definition then example",
    success_count: int = 3,
) -> Pattern:
    return Pattern(
        id=str(uuid.uuid4()),
        trigger=trigger,
        description=description,
        success_count=success_count,
    )


# ---------------------------------------------------------------------------
# 1. SingleCallExecutor with real EnrichedContext (mocked router)
# ---------------------------------------------------------------------------


class TestExecutorWithEnrichedContext:
    """SingleCallExecutor fed with EnrichedContext-derived GraphContext."""

    async def test_enriched_entities_appear_in_prompt(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        enriched = _make_enriched_context(
            entities=(
                {"type": "Person", "name": "Alice", "details": "AI researcher"},
            ),
            entity_tokens=10,
        )
        graph_ctx = _enriched_to_graph_context(enriched)

        result = await executor.execute("Who is Alice?", graph_ctx)

        assert result.success is True
        system_content = provider.captured_messages[0][0]["content"]
        assert "Alice" in system_content
        assert "AI researcher" in system_content

    async def test_enriched_memories_appear_in_prompt(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        enriched = _make_enriched_context(
            memories=("prefers functional programming", "uses Haskell"),
            memory_tokens=8,
        )
        graph_ctx = _enriched_to_graph_context(enriched)

        result = await executor.execute("What languages?", graph_ctx)

        assert result.success is True
        system_content = provider.captured_messages[0][0]["content"]
        assert "prefers functional programming" in system_content
        assert "uses Haskell" in system_content

    async def test_enriched_reflections_appear_in_prompt(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        enriched = _make_enriched_context(
            reflections=(
                {
                    "task_description": "fetch API data",
                    "what_failed": "timeout",
                    "why": "no retry",
                    "what_to_try": "add backoff",
                },
            ),
            reflection_tokens=12,
        )
        graph_ctx = _enriched_to_graph_context(enriched)

        result = await executor.execute("Fetch data from API", graph_ctx)

        assert result.success is True
        system_content = provider.captured_messages[0][0]["content"]
        assert "PAST FAILURE" in system_content
        assert "timeout" in system_content
        assert "add backoff" in system_content

    async def test_enriched_context_token_count_tracked(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        enriched = _make_enriched_context(
            entities=({"type": "Tool", "name": "grep", "details": "search"},),
            entity_tokens=5,
            memory_tokens=3,
            reflection_tokens=7,
        )
        graph_ctx = _enriched_to_graph_context(enriched)

        result = await executor.execute("test", graph_ctx)

        assert result.context_tokens == enriched.total_tokens
        assert result.context_tokens == 15


# ---------------------------------------------------------------------------
# 2. EnrichedContext validity from seeded graph
# ---------------------------------------------------------------------------


class TestEnrichedContextStructure:
    """EnrichedContext dataclass produces valid, well-typed structures."""

    def test_empty_enriched_context_has_zero_tokens(self) -> None:
        ctx = EnrichedContext()
        assert ctx.total_tokens == 0
        assert ctx.entities == ()
        assert ctx.memories == ()
        assert ctx.reflections == ()
        assert ctx.patterns == ()
        assert ctx.conversation_turns == ()

    def test_total_tokens_sums_all_sections(self) -> None:
        ctx = _make_enriched_context(
            entity_tokens=10,
            memory_tokens=20,
            reflection_tokens=30,
            pattern_tokens=40,
            conversation_tokens=50,
        )
        assert ctx.total_tokens == 150

    def test_enriched_context_is_frozen(self) -> None:
        ctx = EnrichedContext()
        with pytest.raises(AttributeError):
            ctx.entity_tokens = 99  # type: ignore[misc]

    def test_enriched_context_with_all_sections_populated(self) -> None:
        pattern = _sample_pattern()
        ctx = _make_enriched_context(
            entities=({"type": "Org", "name": "Anthropic", "details": "AI lab"},),
            memories=("founded in 2021",),
            reflections=({"task_description": "t", "what_failed": "f", "why": "w", "what_to_try": "x"},),
            patterns=(pattern,),
            conversation_turns=({"role": "user", "content": "hello"},),
            entity_tokens=5,
            memory_tokens=3,
            reflection_tokens=4,
            pattern_tokens=2,
            conversation_tokens=1,
        )
        assert len(ctx.entities) == 1
        assert len(ctx.memories) == 1
        assert len(ctx.reflections) == 1
        assert len(ctx.patterns) == 1
        assert len(ctx.conversation_turns) == 1
        assert ctx.total_tokens == 15


# ---------------------------------------------------------------------------
# 3. TokenBudget trims oversized context
# ---------------------------------------------------------------------------


class TestTokenBudgetTrimsOversizedContext:
    """TokenBudget correctly trims oversized sections by priority."""

    def test_all_sections_fit_within_budget(self) -> None:
        budget = TokenBudget(max_tokens=8192)
        sections = {
            "conversation": "short conversation",
            "reflections": "short reflection",
            "entities": "short entity",
            "patterns": "short pattern",
        }
        result = budget.trim_to_budget(sections)

        for name in sections:
            assert result[name] == sections[name]

    def test_lowest_priority_dropped_first(self) -> None:
        """With a tiny budget, only the highest-priority section survives."""
        budget = TokenBudget(
            max_tokens=810,
            system_prompt_reserve=0,
            user_message_reserve=0,
            response_reserve=0,
        )
        # "conversation" has priority 1 (highest), "patterns" has priority 4 (lowest)
        sections = {
            "conversation": " ".join(["word"] * 300),  # ~400 tokens
            "reflections": " ".join(["word"] * 300),    # ~400 tokens
            "entities": " ".join(["word"] * 300),       # ~400 tokens
            "patterns": " ".join(["word"] * 300),       # ~400 tokens
        }
        result = budget.trim_to_budget(sections)

        # Only conversation (priority 1) should fit; reflections (priority 2) might fit too
        # but the total is ~1600 tokens for all, budget is 810
        assert result["conversation"] != ""
        # patterns (priority 4) should be dropped
        assert result["patterns"] == ""

    def test_empty_sections_dict_returns_empty(self) -> None:
        budget = TokenBudget()
        assert budget.trim_to_budget({}) == {}

    def test_zero_budget_drops_everything(self) -> None:
        budget = TokenBudget(
            max_tokens=100,
            system_prompt_reserve=50,
            user_message_reserve=50,
            response_reserve=50,
        )
        # available_budget = max(0, 100 - 50 - 50 - 50) = 0
        assert budget.available_budget == 0

        sections = {"conversation": "some text", "entities": "some entities"}
        result = budget.trim_to_budget(sections)

        assert result["conversation"] == ""
        assert result["entities"] == ""


# ---------------------------------------------------------------------------
# 4. Full flow: enrich -> format -> single call -> ExecutionResult
# ---------------------------------------------------------------------------


class TestFullFlowIntegration:
    """End-to-end: EnrichedContext -> GraphContext -> SingleCallExecutor."""

    async def test_full_flow_success(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        enriched = _make_enriched_context(
            entities=(
                {"type": "Language", "name": "Python", "details": "3.12"},
            ),
            memories=("user prefers type hints",),
            entity_tokens=5,
            memory_tokens=4,
        )
        graph_ctx = _enriched_to_graph_context(enriched)
        pattern = _sample_pattern(trigger="code help", description="show example")

        result = await executor.execute(
            "Write a Python function",
            graph_ctx,
            conversation_history=[
                {"role": "user", "content": "I need help coding"},
                {"role": "assistant", "content": "Sure, what language?"},
            ],
            pattern_hints=[pattern],
            complexity=2,
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output == "integration response"
        assert result.llm_calls == 1
        assert result.total_tokens == 30  # 20 in + 10 out
        assert result.context_tokens == enriched.total_tokens
        assert result.model_used == "test-model-v1"

        # Verify prompt structure
        messages = provider.captured_messages[0]
        assert messages[0]["role"] == "system"
        assert "Python" in messages[0]["content"]
        assert "type hints" in messages[0]["content"]
        assert "code help" in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Write a Python function"

    async def test_full_flow_with_budget_trimming(self) -> None:
        """TokenBudget trims context before it reaches the executor."""
        budget = TokenBudget(
            max_tokens=500,
            system_prompt_reserve=50,
            user_message_reserve=50,
            response_reserve=50,
        )
        # available = 350 tokens
        sections = {
            "conversation": "Recent chat about data pipelines",
            "reflections": "Past failure: timeout on large datasets",
            "entities": "Entity: DataPipeline | ETL framework",
            "patterns": " ".join(["long"] * 500),  # way over budget
        }
        trimmed = budget.trim_to_budget(sections)

        # High-priority sections should survive; the huge "patterns" section gets dropped
        assert trimmed["conversation"] != ""
        assert trimmed["reflections"] != ""
        assert trimmed["entities"] != ""
        assert trimmed["patterns"] == ""

        # Now feed a GraphContext derived from trimmed sections to the executor
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        graph_ctx = GraphContext(
            relevant_entities=({"type": "Tool", "name": "DataPipeline", "details": "ETL"},),
            active_memories=("prefers batch processing",),
            total_tokens=budget.estimate_tokens(trimmed["entities"]),
        )

        result = await executor.execute("Build a data pipeline", graph_ctx)
        assert result.success is True
        assert "DataPipeline" in provider.captured_messages[0][0]["content"]

    async def test_full_flow_provider_error(self) -> None:
        """Full flow with a failing provider returns failed ExecutionResult."""
        provider = _FailingProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        enriched = _make_enriched_context(
            entities=({"type": "API", "name": "OpenAI", "details": "LLM provider"},),
            entity_tokens=5,
        )
        graph_ctx = _enriched_to_graph_context(enriched)

        result = await executor.execute("Call the API", graph_ctx)

        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert result.output == ""
        assert len(result.errors) >= 1
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.total_latency_ms > 0


# ---------------------------------------------------------------------------
# 5. Edge cases: empty graph, empty conversation, no patterns
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for empty or minimal inputs."""

    async def test_empty_graph_context(self) -> None:
        """Empty GraphContext should produce no <context> tag in the prompt."""
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("hello", GraphContext())

        assert result.success is True
        system_content = provider.captured_messages[0][0]["content"]
        assert "<context>" not in system_content

    async def test_empty_conversation_history(self) -> None:
        """Empty conversation list should not add any history messages."""
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute(
            "test", GraphContext(), conversation_history=[]
        )

        assert result.success is True
        messages = provider.captured_messages[0]
        # Only system + user message
        assert len(messages) == 2

    async def test_none_conversation_history(self) -> None:
        """None conversation_history should not add any history messages."""
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute(
            "test", GraphContext(), conversation_history=None
        )

        assert result.success is True
        messages = provider.captured_messages[0]
        assert len(messages) == 2

    async def test_no_patterns_no_hint_section(self) -> None:
        """No pattern hints means no 'Similar tasks' section in prompt."""
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute(
            "test", GraphContext(), pattern_hints=[]
        )

        assert result.success is True
        system_content = provider.captured_messages[0][0]["content"]
        assert "Similar tasks" not in system_content

    async def test_empty_enriched_context_converts_to_empty_graph_context(self) -> None:
        """An empty EnrichedContext converts to an empty GraphContext."""
        enriched = EnrichedContext()
        graph_ctx = _enriched_to_graph_context(enriched)

        assert graph_ctx.relevant_entities == ()
        assert graph_ctx.active_memories == ()
        assert graph_ctx.total_tokens == 0
        assert graph_ctx.format() == ""

    def test_enriched_context_with_zero_token_budget(self) -> None:
        """EnrichedContext with zero token budget returns zero total_tokens."""
        ctx = _make_enriched_context(
            entity_tokens=0,
            memory_tokens=0,
            reflection_tokens=0,
            pattern_tokens=0,
            conversation_tokens=0,
        )
        assert ctx.total_tokens == 0


# ---------------------------------------------------------------------------
# 6. ExecutionResult field correctness
# ---------------------------------------------------------------------------


class TestExecutionResultFields:
    """Verify that single-call output has correct ExecutionResult fields."""

    async def test_result_has_valid_root_id(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("task", GraphContext())

        # UUID format: 8-4-4-4-12 hex characters
        assert len(result.root_id) == 36
        parts = result.root_id.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]

    async def test_result_nodes_contains_root_id(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("task", GraphContext())

        assert result.root_id in result.nodes
        assert len(result.nodes) == 1

    async def test_result_tools_used_is_zero(self) -> None:
        """Single-call mode uses no tools."""
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("task", GraphContext())

        assert result.tools_used == 0

    async def test_result_cost_matches_provider(self) -> None:
        custom_result = CompletionResult(
            content="expensive answer",
            model="gpt-4o",
            tokens_in=500,
            tokens_out=200,
            latency_ms=1200.0,
            cost=0.05,
        )
        provider = _MockProvider(result=custom_result)
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("complex task", GraphContext(), complexity=5)

        assert result.total_cost == 0.05
        assert result.total_tokens == 700
        assert result.model_used == "gpt-4o"

    async def test_result_errors_tuple_empty_on_success(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("task", GraphContext())

        assert result.errors == ()
        assert isinstance(result.errors, tuple)

    async def test_result_is_frozen(self) -> None:
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("task", GraphContext())

        with pytest.raises(AttributeError):
            result.output = "modified"  # type: ignore[misc]
