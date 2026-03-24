"""Tests for deterministic aggregation wired into DAGExecutor."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.aggregator import Aggregator
from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)


class ConfigurableExecutor:
    """Mock executor that returns configurable outputs per task description."""

    def __init__(self, outputs: dict[str, str] | None = None) -> None:
        self._outputs = outputs or {}

    async def execute(self, task: str, complexity: int = 1, **kwargs: Any) -> ExecutionResult:
        # Match by checking if any configured key is a substring of the task
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
) -> TaskNode:
    """Helper to create an atomic leaf TaskNode."""
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=1,
        status=TaskStatus.READY,
        requires=requires or [],
        provides=provides or [],
        consumes=consumes or [],
    )


class TestTemplateFillAggregation:
    async def test_template_fill_aggregation(self) -> None:
        """3 parallel leaves with provides keys, template with slots -> aggregated output fills slots."""
        mock = ConfigurableExecutor(outputs={
            "Get Amsterdam weather": "Sunny, 22C",
            "Get London weather": "Rainy, 15C",
            "Get Berlin weather": "Cloudy, 18C",
        })
        dag = DAGExecutor(executor=mock, max_concurrency=10)
        dag.aggregation_template = {
            "aggregation_type": "template_fill",
            "template": "Amsterdam: {weather_ams}\nLondon: {weather_lon}\nBerlin: {weather_ber}",
            "slot_definitions": {
                "weather_ams": "Amsterdam weather",
                "weather_lon": "London weather",
                "weather_ber": "Berlin weather",
            },
        }

        nodes = [
            _make_leaf("a", "Get Amsterdam weather", provides=["weather_ams"]),
            _make_leaf("b", "Get London weather", provides=["weather_lon"]),
            _make_leaf("c", "Get Berlin weather", provides=["weather_ber"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert result.output == "Amsterdam: Sunny, 22C\nLondon: Rainy, 15C\nBerlin: Cloudy, 18C"


class TestConcatenateAggregation:
    async def test_concatenate_aggregation(self) -> None:
        """No template -> outputs concatenated with headers."""
        mock = ConfigurableExecutor(outputs={
            "Task A": "Alpha output",
            "Task B": "Beta output",
        })
        dag = DAGExecutor(executor=mock, max_concurrency=10)
        # No template set (None by default)

        nodes = [
            _make_leaf("a", "Task A", provides=["result_a"]),
            _make_leaf("b", "Task B", provides=["result_b"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # Concatenation mode uses headers derived from keys
        assert "## Result A" in result.output
        assert "Alpha output" in result.output
        assert "## Result B" in result.output
        assert "Beta output" in result.output


class TestAggregationUsesStructuredOutput:
    async def test_aggregation_uses_structured_output(self) -> None:
        """Leaves return JSON -> aggregator gets parsed values for matching keys."""
        json_a = json.dumps({"weather_ams": "Sunny, 22C", "extra": "ignored"})
        json_b = json.dumps({"weather_lon": "Rainy, 15C"})
        mock = ConfigurableExecutor(outputs={
            "Get Amsterdam": json_a,
            "Get London": json_b,
        })
        dag = DAGExecutor(executor=mock, max_concurrency=10)
        dag.aggregation_template = {
            "aggregation_type": "template_fill",
            "template": "AMS: {weather_ams} | LON: {weather_lon}",
        }

        nodes = [
            _make_leaf("a", "Get Amsterdam", provides=["weather_ams"]),
            _make_leaf("b", "Get London", provides=["weather_lon"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # JSON was parsed; the matching key value is extracted
        assert "Sunny, 22C" in result.output
        assert "Rainy, 15C" in result.output
        assert result.output == "AMS: Sunny, 22C | LON: Rainy, 15C"


class TestAggregationFallbackPlainText:
    async def test_aggregation_fallback_plain_text(self) -> None:
        """Leaves return plain text (not JSON) -> still aggregated correctly."""
        mock = ConfigurableExecutor(outputs={
            "Fetch data": "plain text result",
            "Fetch more": "another plain result",
        })
        dag = DAGExecutor(executor=mock, max_concurrency=10)
        dag.aggregation_template = {
            "aggregation_type": "template_fill",
            "template": "First: {data_one}\nSecond: {data_two}",
        }

        nodes = [
            _make_leaf("a", "Fetch data", provides=["data_one"]),
            _make_leaf("b", "Fetch more", provides=["data_two"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # Plain text cannot be JSON-parsed, so full output is used as the value
        assert result.output == "First: plain text result\nSecond: another plain result"


class TestOrchestratorPassesTemplate:
    async def test_orchestrator_passes_template(self) -> None:
        """Verify orchestrator sets aggregation_template on DAGExecutor before execute."""
        from core_gb.orchestrator import Orchestrator

        mock_store = MagicMock()
        mock_store.get_context.return_value = None
        mock_router = MagicMock()

        orchestrator = Orchestrator(store=mock_store, router=mock_router, force_decompose=True)

        # Set up decomposer to return nodes with a template
        test_template = {
            "aggregation_type": "template_fill",
            "template": "Result: {data_a} and {data_b}",
        }

        test_nodes = [
            TaskNode(
                id="root",
                description="Root",
                is_atomic=False,
                children=["leaf_a", "leaf_b"],
                status=TaskStatus.CREATED,
            ),
            _make_leaf("leaf_a", "Leaf A", provides=["data_a"]),
            _make_leaf("leaf_b", "Leaf B", provides=["data_b"]),
        ]

        # Mock decomposer
        async def mock_decompose(msg: str, context: Any = None) -> list[TaskNode]:
            orchestrator._decomposer.last_template = test_template
            return test_nodes

        orchestrator._decomposer.decompose = mock_decompose
        orchestrator._decomposer.last_template = None

        # Mock intake to return complex
        mock_intake_result = MagicMock()
        mock_intake_result.is_simple = False
        mock_intake_result.complexity = 3
        mock_intake_result.entities = []
        orchestrator._intake.parse = MagicMock(return_value=mock_intake_result)

        # Mock pattern store
        orchestrator._pattern_store.load_all = MagicMock(return_value=[])

        # Mock DAG executor to capture the template and return a result
        captured_template = None
        original_execute = orchestrator._dag_executor.execute

        async def capture_execute(nodes: list[TaskNode]) -> ExecutionResult:
            nonlocal captured_template
            captured_template = orchestrator._dag_executor.aggregation_template
            return ExecutionResult(
                root_id="test",
                output="test output",
                success=True,
                total_nodes=2,
                total_tokens=20,
                total_latency_ms=1.0,
                total_cost=0.002,
            )

        orchestrator._dag_executor.execute = capture_execute

        # Mock graph updater
        orchestrator._graph_updater.update = MagicMock()

        await orchestrator.process("Compare weather in two cities")

        assert captured_template is not None
        assert captured_template == test_template
