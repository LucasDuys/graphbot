"""Tests for the Orchestrator -- intake -> decompose -> execute pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from core_gb.orchestrator import Orchestrator
from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _weather_tree_json() -> dict:
    """Valid weather-3-cities decomposition tree."""
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Get weather for 3 cities",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 2,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["w1", "w2", "w3", "agg"],
            },
            {
                "id": "w1",
                "description": "Get Amsterdam weather",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_ams"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "w2",
                "description": "Get London weather",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_lon"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "w3",
                "description": "Get Berlin weather",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_ber"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "agg",
                "description": "Summarize weather",
                "domain": "synthesis",
                "task_type": "WRITE",
                "complexity": 1,
                "depends_on": ["w1", "w2", "w3"],
                "provides": ["summary"],
                "consumes": ["weather_ams", "weather_lon", "weather_ber"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


class SequentialMockProvider(ModelProvider):
    """Mock provider that returns different responses per call.

    First call is typically the decomposition JSON, subsequent calls are leaf
    execution responses.
    """

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = responses
        self._call_count = 0
        self.call_log: list[list[dict]] = []

    @property
    def name(self) -> str:
        return "mock"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        self.call_log.append(messages)
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def _simple_completion(content: str, tokens: int = 10, cost: float = 0.001) -> CompletionResult:
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens,
        tokens_out=tokens,
        latency_ms=50.0,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimpleTaskDirectExecution:
    """Simple tasks (is_simple=True) route directly to SimpleExecutor."""

    async def test_simple_task_direct_execution(self) -> None:
        store = _make_store()
        provider = SequentialMockProvider([_simple_completion("4")])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("What is 2+2?")

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output == "4"
        # Simple task: only one LLM call (executor), no decomposition
        assert len(provider.call_log) == 1

        store.close()


class TestComplexTaskDecomposition:
    """Complex tasks route through decomposer then execute leaves."""

    async def test_complex_task_decomposition(self) -> None:
        store = _make_store()
        decomposition_json = json.dumps(_weather_tree_json())

        # First call: decomposition. Subsequent calls: leaf execution + synthesis.
        # Some leaf nodes (web domain) may route through tools rather than the
        # provider, so we provide enough responses for any execution path.
        # Final call: LLM synthesis aggregation (triggered for >= 3 subtasks).
        responses = [
            _simple_completion(decomposition_json, tokens=50, cost=0.01),
            _simple_completion("Amsterdam: 15C sunny", tokens=10, cost=0.001),
            _simple_completion("London: 12C rainy", tokens=10, cost=0.001),
            _simple_completion("Berlin: 10C cloudy", tokens=10, cost=0.001),
            _simple_completion("Summary of weather", tokens=10, cost=0.001),
            _simple_completion("Amsterdam is 15C and sunny, London is 12C and rainy, Berlin is 10C and cloudy.", tokens=10, cost=0.001),
            _simple_completion("Synthesized weather comparison.", tokens=10, cost=0.001),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Compare the weather in Amsterdam, London, and Berlin"
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        # Output should be non-empty (synthesis or aggregated leaf outputs)
        assert len(result.output) > 0
        # Should have called provider multiple times (1 decompose + leaves + synthesis)
        assert len(provider.call_log) >= 2

        store.close()


class TestSequentialExecutionOrder:
    """3-node sequential chain: A -> B -> C. Verify topological order."""

    async def test_sequential_execution_order(self) -> None:
        store = _make_store()

        sequential_tree = {
            "nodes": [
                {
                    "id": "root",
                    "description": "Sequential pipeline",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 2,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["a", "b", "c"],
                },
                {
                    "id": "a",
                    "description": "Step A",
                    "domain": "system",
                    "task_type": "RETRIEVE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["data_a"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "b",
                    "description": "Step B",
                    "domain": "system",
                    "task_type": "THINK",
                    "complexity": 1,
                    "depends_on": ["a"],
                    "provides": ["data_b"],
                    "consumes": ["data_a"],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "c",
                    "description": "Step C",
                    "domain": "synthesis",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": ["b"],
                    "provides": ["data_c"],
                    "consumes": ["data_b"],
                    "is_atomic": True,
                    "children": [],
                },
            ]
        }

        # 1 decompose + 3 leaf executions + 1 synthesis aggregation = 5 total
        responses = [
            _simple_completion(json.dumps(sequential_tree), tokens=50, cost=0.01),
            _simple_completion("Result A", tokens=10, cost=0.001),
            _simple_completion("Result B", tokens=10, cost=0.001),
            _simple_completion("Result C", tokens=10, cost=0.001),
            _simple_completion("Combined: Result A then B then C", tokens=10, cost=0.001),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Do step A then step B then step C and summarize"
        )

        assert result.success is True
        # 1 decompose call + 3 leaf execution calls + 1 synthesis = 5 total
        assert len(provider.call_log) == 5

        # Verify order: call_log[1] = A, call_log[2] = B, call_log[3] = C
        # Each leaf call has a user message containing the step description
        leaf_descriptions = []
        for call_messages in provider.call_log[1:4]:
            user_msg = [m for m in call_messages if m["role"] == "user"][0]
            leaf_descriptions.append(user_msg["content"])

        assert "Step A" in leaf_descriptions[0]
        assert "Step B" in leaf_descriptions[1]
        assert "Step C" in leaf_descriptions[2]

        store.close()


class TestAggregatedResult:
    """Verify tokens/latency/cost are aggregated correctly across leaves."""

    async def test_aggregated_result(self) -> None:
        store = _make_store()

        simple_tree = {
            "nodes": [
                {
                    "id": "root",
                    "description": "Two tasks",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 2,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["a", "b"],
                },
                {
                    "id": "a",
                    "description": "Task A",
                    "domain": "system",
                    "task_type": "RETRIEVE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["data_a"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "b",
                    "description": "Task B",
                    "domain": "system",
                    "task_type": "RETRIEVE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["data_b"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
            ]
        }

        responses = [
            _simple_completion(json.dumps(simple_tree), tokens=50, cost=0.01),
            CompletionResult(
                content="Output A",
                model="mock-model",
                tokens_in=20,
                tokens_out=10,
                latency_ms=100.0,
                cost=0.005,
            ),
            CompletionResult(
                content="Output B",
                model="mock-model",
                tokens_in=30,
                tokens_out=15,
                latency_ms=200.0,
                cost=0.010,
            ),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Compare two things and also analyze them together"
        )

        assert result.success is True
        # Tokens: (20+10) + (30+15) = 75
        assert result.total_tokens == 75
        # Cost: 0.005 + 0.010 = 0.015
        assert abs(result.total_cost - 0.015) < 1e-9
        # Latency: max of leaf latencies (wall-clock captured, but at least > 0)
        assert result.total_latency_ms > 0
        # Output should contain both leaf outputs
        assert "Output A" in result.output
        assert "Output B" in result.output

        store.close()


class TestDataForwarding:
    """Node A provides data, node B consumes it. Verify B receives A's output."""

    async def test_data_forwarding(self) -> None:
        store = _make_store()

        chain_tree = {
            "nodes": [
                {
                    "id": "root",
                    "description": "Data chain",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 2,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["producer", "consumer"],
                },
                {
                    "id": "producer",
                    "description": "Produce data",
                    "domain": "system",
                    "task_type": "THINK",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["weather_ams"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "consumer",
                    "description": "Consume data",
                    "domain": "synthesis",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": ["producer"],
                    "provides": ["summary"],
                    "consumes": ["weather_ams"],
                    "is_atomic": True,
                    "children": [],
                },
            ]
        }

        responses = [
            _simple_completion(json.dumps(chain_tree), tokens=50, cost=0.01),
            _simple_completion("Amsterdam is 15C and sunny", tokens=10, cost=0.001),
            _simple_completion("Summary: Amsterdam is warm", tokens=10, cost=0.001),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Get weather for Amsterdam and then summarize it"
        )

        assert result.success is True

        # The consumer's LLM call should have received the producer's output
        # via input_data injection. Check that the consumer's user message
        # or system message contains the producer output.
        assert len(provider.call_log) == 3  # decompose + 2 leaves
        consumer_messages = provider.call_log[2]
        # The forwarded data is injected into the user message via <forwarded_data> tags
        consumer_user = [m for m in consumer_messages if m["role"] == "user"][0]
        assert "Amsterdam is 15C and sunny" in consumer_user["content"]

        store.close()


class TestDecomposerFallback:
    """Decomposer falls back to single node -- still executes correctly."""

    async def test_decomposer_fallback(self) -> None:
        store = _make_store()

        # Decomposer gets garbage twice, falls back to single node.
        # Then executor runs that single node.
        responses = [
            _simple_completion("garbage!!!", tokens=5, cost=0.001),
            _simple_completion("more garbage!!!", tokens=5, cost=0.001),
            _simple_completion("The answer is 42", tokens=10, cost=0.001),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Compare the weather in Amsterdam, London, and Berlin"
        )

        assert result.success is True
        assert "42" in result.output

        store.close()


class TestProcessReturnsExecutionResult:
    """Verify return type and required fields on ExecutionResult."""

    async def test_process_returns_execution_result(self) -> None:
        store = _make_store()
        provider = SequentialMockProvider([_simple_completion("hello")])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("What is 2+2?")

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.root_id, str)
        assert len(result.root_id) > 0
        assert isinstance(result.output, str)
        assert isinstance(result.success, bool)
        assert isinstance(result.total_nodes, int)
        assert isinstance(result.total_tokens, int)
        assert isinstance(result.total_latency_ms, float)
        assert isinstance(result.total_cost, float)
        assert isinstance(result.errors, tuple)

        store.close()
