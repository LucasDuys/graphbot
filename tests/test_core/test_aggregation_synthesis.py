"""Tests for LLM synthesis aggregation and JSON artifact stripping.

Covers:
- JSON artifact stripping from subtask outputs
- LLM synthesis for complex decomposed tasks (complexity >= 3)
- Simple tasks (1-2 subtasks) skip synthesis, only clean up output
- Configurable synthesis threshold
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.aggregator import Aggregator, strip_json_artifacts
from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConfigurableExecutor:
    """Mock executor that returns configurable outputs per task description."""

    def __init__(self, outputs: dict[str, str] | None = None) -> None:
        self._outputs = outputs or {}

    async def execute(
        self, task: str, complexity: int = 1, **kwargs: Any
    ) -> ExecutionResult:
        output = ""
        for key, value in self._outputs.items():
            if key in task:
                output = value
                break
        if not output:
            output = f"Result: {task}"
        return ExecutionResult(
            root_id=str(uuid.uuid4()),
            output=output,
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=1.0,
            total_cost=0.001,
        )


def _make_leaf(
    node_id: str,
    description: str,
    requires: list[str] | None = None,
    provides: list[str] | None = None,
    consumes: list[str] | None = None,
    complexity: int = 1,
) -> TaskNode:
    """Helper to create an atomic leaf TaskNode."""
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=complexity,
        status=TaskStatus.READY,
        requires=requires or [],
        provides=provides or [],
        consumes=consumes or [],
    )


# ---------------------------------------------------------------------------
# strip_json_artifacts tests
# ---------------------------------------------------------------------------

class TestStripJsonArtifacts:
    """Test that raw JSON artifacts are removed from subtask outputs."""

    def test_strips_full_json_object(self) -> None:
        text = '{"answer": "Paris is the capital of France", "confidence": 0.95}'
        result = strip_json_artifacts(text)
        assert "{" not in result
        assert "}" not in result
        assert '"answer"' not in result
        assert '"confidence"' not in result
        assert "Paris is the capital of France" in result

    def test_strips_nested_json(self) -> None:
        text = '{"result": {"city": "Amsterdam", "temp": 22}, "status": "ok"}'
        result = strip_json_artifacts(text)
        assert "{" not in result
        assert "}" not in result
        assert "Amsterdam" in result

    def test_preserves_plain_text(self) -> None:
        text = "The weather in Amsterdam is sunny with 22 degrees Celsius."
        result = strip_json_artifacts(text)
        assert result == text

    def test_strips_json_prefix_in_mixed_content(self) -> None:
        text = 'Here is the data: {"key": "value"} and more text.'
        result = strip_json_artifacts(text)
        assert "{" not in result
        assert "}" not in result
        assert "more text" in result

    def test_strips_quoted_keys(self) -> None:
        text = '"answer": "The result is 42"'
        result = strip_json_artifacts(text)
        # Should not have JSON-style quoted key patterns
        assert '"answer":' not in result
        assert "42" in result

    def test_empty_string(self) -> None:
        assert strip_json_artifacts("") == ""

    def test_json_array_stripped(self) -> None:
        text = '["item1", "item2", "item3"]'
        result = strip_json_artifacts(text)
        assert "[" not in result or "item1" in result
        # Items should still be readable
        assert "item1" in result


# ---------------------------------------------------------------------------
# Aggregator.synthesize tests
# ---------------------------------------------------------------------------

class TestAggregatorSynthesize:
    """Test LLM synthesis aggregation for complex tasks."""

    async def test_synthesis_called_for_complex_tasks(self) -> None:
        """Synthesis should be invoked when leaf count >= synthesis_threshold."""
        mock_router = MagicMock()
        mock_completion = CompletionResult(
            content="Amsterdam is sunny at 22C while London is rainy at 15C and Berlin is cloudy at 18C.",
            model="test-model",
            tokens_in=50,
            tokens_out=30,
            latency_ms=100.0,
            cost=0.001,
        )
        mock_router.route = AsyncMock(return_value=mock_completion)

        agg = Aggregator(router=mock_router, synthesis_threshold=3)

        leaf_outputs = {
            "weather_ams": "Sunny, 22C",
            "weather_lon": "Rainy, 15C",
            "weather_ber": "Cloudy, 18C",
        }

        result = await agg.synthesize(
            original_question="What is the weather in Amsterdam, London, and Berlin?",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        assert mock_router.route.called
        assert "sunny" in result.lower() or "Amsterdam" in result

    async def test_synthesis_skipped_below_threshold(self) -> None:
        """Synthesis should NOT be called when leaf count < synthesis_threshold."""
        mock_router = MagicMock()
        mock_router.route = AsyncMock()

        agg = Aggregator(router=mock_router, synthesis_threshold=3)

        leaf_outputs = {
            "weather_ams": '{"answer": "Sunny, 22C"}',
            "weather_lon": "Rainy, 15C",
        }

        result = await agg.synthesize(
            original_question="Compare weather in Amsterdam and London",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        # LLM should NOT have been called
        mock_router.route.assert_not_called()
        # Output should still be cleaned (JSON artifacts stripped)
        assert "{" not in result
        assert "Sunny, 22C" in result

    async def test_synthesis_with_custom_threshold(self) -> None:
        """Custom threshold of 2 should trigger synthesis for 2 outputs."""
        mock_router = MagicMock()
        mock_completion = CompletionResult(
            content="Both cities have pleasant weather.",
            model="test-model",
            tokens_in=30,
            tokens_out=15,
            latency_ms=80.0,
            cost=0.001,
        )
        mock_router.route = AsyncMock(return_value=mock_completion)

        agg = Aggregator(router=mock_router, synthesis_threshold=2)

        leaf_outputs = {
            "weather_ams": "Sunny, 22C",
            "weather_lon": "Rainy, 15C",
        }

        result = await agg.synthesize(
            original_question="Compare weather",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        mock_router.route.assert_called_once()
        assert "pleasant" in result.lower() or "weather" in result.lower()

    async def test_synthesis_strips_json_before_llm_call(self) -> None:
        """JSON artifacts should be stripped from leaf outputs before synthesis."""
        mock_router = MagicMock()
        mock_completion = CompletionResult(
            content="A coherent synthesized response.",
            model="test-model",
            tokens_in=40,
            tokens_out=20,
            latency_ms=90.0,
            cost=0.001,
        )
        mock_router.route = AsyncMock(return_value=mock_completion)

        agg = Aggregator(router=mock_router, synthesis_threshold=3)

        leaf_outputs = {
            "a": '{"answer": "Result A", "confidence": 0.9}',
            "b": '{"answer": "Result B", "confidence": 0.8}',
            "c": "Plain text result C",
        }

        await agg.synthesize(
            original_question="Test question",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        # Check that the messages sent to the router have cleaned outputs
        call_args = mock_router.route.call_args
        messages = call_args[0][1]  # second positional arg
        user_msg = next(m for m in messages if m["role"] == "user")
        assert '{"answer"' not in user_msg["content"]
        assert '"confidence"' not in user_msg["content"]

    async def test_synthesis_fallback_on_router_failure(self) -> None:
        """If the LLM synthesis call fails, fall back to deterministic aggregation."""
        mock_router = MagicMock()
        mock_router.route = AsyncMock(side_effect=Exception("LLM unavailable"))

        agg = Aggregator(router=mock_router, synthesis_threshold=3)

        leaf_outputs = {
            "a": "Result A",
            "b": "Result B",
            "c": "Result C",
        }

        result = await agg.synthesize(
            original_question="Test question",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        # Should fall back to deterministic aggregation, not raise
        assert "Result A" in result
        assert "Result B" in result
        assert "Result C" in result

    async def test_synthesis_without_router_uses_deterministic(self) -> None:
        """Aggregator without router always uses deterministic aggregation."""
        agg = Aggregator()  # No router

        leaf_outputs = {
            "a": "Result A",
            "b": "Result B",
            "c": "Result C",
        }

        result = await agg.synthesize(
            original_question="Test question",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        # Deterministic fallback
        assert "Result A" in result


# ---------------------------------------------------------------------------
# DAGExecutor integration: synthesis wired into _aggregate_results
# ---------------------------------------------------------------------------

class TestDAGExecutorSynthesis:
    """Test that DAGExecutor uses synthesis for decomposed tasks."""

    async def test_decomposed_comparison_produces_clean_prose(self) -> None:
        """A decomposed comparison task with 3+ subtasks should produce
        clean prose (no JSON artifacts) via LLM synthesis."""
        mock_exec = ConfigurableExecutor(outputs={
            "Amsterdam weather": '{"answer": "Sunny, 22C", "confidence": 0.9}',
            "London weather": '{"answer": "Rainy, 15C", "confidence": 0.85}',
            "Berlin weather": '{"answer": "Cloudy, 18C", "confidence": 0.88}',
        })

        mock_router = MagicMock()
        mock_completion = CompletionResult(
            content=(
                "Amsterdam is sunny at 22C. London is rainy at 15C. "
                "Berlin is cloudy at 18C."
            ),
            model="test-model",
            tokens_in=50,
            tokens_out=30,
            latency_ms=100.0,
            cost=0.001,
        )
        mock_router.route = AsyncMock(return_value=mock_completion)

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            router=mock_router,
        )
        dag.original_question = "What is the weather in Amsterdam, London, and Berlin?"

        nodes = [
            _make_leaf("a", "Amsterdam weather", provides=["weather_ams"], complexity=3),
            _make_leaf("b", "London weather", provides=["weather_lon"], complexity=3),
            _make_leaf("c", "Berlin weather", provides=["weather_ber"], complexity=3),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # Output should be clean prose, not raw JSON
        assert "{" not in result.output
        assert "}" not in result.output
        assert '"answer"' not in result.output

    async def test_simple_task_skips_synthesis(self) -> None:
        """A simple task with only 2 subtasks should skip LLM synthesis
        and just clean up the output."""
        mock_exec = ConfigurableExecutor(outputs={
            "Fetch A": '{"result": "Value A"}',
            "Fetch B": "Value B",
        })

        mock_router = MagicMock()
        mock_router.route = AsyncMock()  # Should NOT be called

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            router=mock_router,
        )

        nodes = [
            _make_leaf("a", "Fetch A", provides=["data_a"], complexity=1),
            _make_leaf("b", "Fetch B", provides=["data_b"], complexity=1),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # Router.route should NOT have been called for synthesis
        mock_router.route.assert_not_called()
        # JSON artifacts should still be stripped from output
        assert '{"result"' not in result.output
