"""Tests for frozen/immutable core types and extended fields."""

import pytest

from core_gb.types import (
    CompletionResult,
    ExecutionResult,
    GraphContext,
    Pattern,
    TaskNode,
    TaskStatus,
)


class TestFrozenTypes:
    """Verify that output types are frozen (immutable)."""

    def test_execution_result_is_frozen(self) -> None:
        result = ExecutionResult(
            root_id="r1",
            output="test",
            success=True,
        )
        with pytest.raises(AttributeError):
            result.output = "changed"  # type: ignore[misc]

    def test_graph_context_is_frozen(self) -> None:
        ctx = GraphContext()
        with pytest.raises(AttributeError):
            ctx.user_summary = "changed"  # type: ignore[misc]

    def test_pattern_is_frozen(self) -> None:
        p = Pattern(id="p1", trigger="test", description="test")
        with pytest.raises(AttributeError):
            p.trigger = "changed"  # type: ignore[misc]

    def test_completion_result_is_frozen(self) -> None:
        cr = CompletionResult(
            content="hello",
            model="test-model",
            tokens_in=10,
            tokens_out=5,
            latency_ms=100.0,
            cost=0.0,
        )
        with pytest.raises(AttributeError):
            cr.content = "changed"  # type: ignore[misc]


class TestExtendedFields:
    """Verify extended fields on types match spec."""

    def test_task_node_complexity_range(self) -> None:
        for c in range(1, 6):
            node = TaskNode(id=f"t{c}", description="test", complexity=c)
            assert node.complexity == c

    def test_execution_result_extended_fields(self) -> None:
        result = ExecutionResult(
            root_id="r1",
            output="test",
            success=True,
            total_cost=0.0,
            context_tokens=500,
            model_used="llama-3.3-70b",
        )
        assert result.context_tokens == 500
        assert result.model_used == "llama-3.3-70b"
        assert result.total_cost == 0.0

    def test_completion_result_fields(self) -> None:
        cr = CompletionResult(
            content="42",
            model="openrouter/meta-llama/llama-3.3-70b-versatile",
            tokens_in=50,
            tokens_out=10,
            latency_ms=250.5,
            cost=0.001,
        )
        assert cr.content == "42"
        assert cr.tokens_in == 50
        assert cr.tokens_out == 10
        assert cr.latency_ms == 250.5

    def test_graph_context_total_tokens(self) -> None:
        ctx = GraphContext(
            user_summary="test user",
            total_tokens=2500,
        )
        assert ctx.total_tokens == 2500
