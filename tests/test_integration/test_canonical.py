"""Canonical integration tests with mocked provider.

Verifies the full Orchestrator pipeline (intake -> decompose -> execute) against
five canonical task types, using a MockProvider that returns predictable responses.
"""

from __future__ import annotations

import json

import pytest

from core_gb.orchestrator import Orchestrator
from core_gb.types import CompletionResult, ExecutionResult
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockProvider(ModelProvider):
    """Provider that returns configurable responses based on call order.

    Each call pops the next response from the queue. If the queue is exhausted,
    the last response is reused for all subsequent calls.
    """

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = list(responses)
        self._call_count: int = 0
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(
    content: str,
    tokens_in: int = 10,
    tokens_out: int = 10,
    cost: float = 0.001,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=50.0,
        cost=cost,
    )


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _weather_decomposition() -> str:
    """JSON decomposition tree for weather-in-3-cities."""
    tree = {
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
                "description": "Summarize weather for all cities",
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
    return json.dumps(tree)


def _sequential_decomposition() -> str:
    """JSON decomposition tree for read-then-parse sequential task."""
    tree = {
        "nodes": [
            {
                "id": "root",
                "description": "Read README and find TODOs",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 2,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["read", "parse"],
            },
            {
                "id": "read",
                "description": "Read README.md file contents",
                "domain": "system",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": ["file_content"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "parse",
                "description": "Find TODO items in the file content",
                "domain": "system",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": ["read"],
                "provides": ["todo_list"],
                "consumes": ["file_content"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }
    return json.dumps(tree)


def _research_decomposition() -> str:
    """JSON decomposition tree for TU/e vs TUM comparison."""
    tree = {
        "nodes": [
            {
                "id": "root",
                "description": "Compare TU/e vs TUM for AI masters",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 3,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["r1", "r2", "agg"],
            },
            {
                "id": "r1",
                "description": "Research TU/e AI masters programs",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 2,
                "depends_on": [],
                "provides": ["tue_info"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "r2",
                "description": "Research TUM AI masters programs",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 2,
                "depends_on": [],
                "provides": ["tum_info"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "agg",
                "description": "Compare and summarize findings",
                "domain": "synthesis",
                "task_type": "WRITE",
                "complexity": 2,
                "depends_on": ["r1", "r2"],
                "provides": ["comparison"],
                "consumes": ["tue_info", "tum_info"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }
    return json.dumps(tree)


# ---------------------------------------------------------------------------
# Canonical tests
# ---------------------------------------------------------------------------


class TestCanonical1SimpleArithmetic:
    """Test 1: Simple arithmetic -- single node, no decomposition."""

    async def test_simple_arithmetic(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("9386")])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("What's 247 * 38?")

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output != ""
        assert "9386" in result.output
        # Simple task: 1 LLM call, 1 node
        assert len(provider.call_log) == 1
        assert result.total_nodes >= 1

        store.close()


class TestCanonical2WeatherDecomposition:
    """Test 2: Weather in 3 cities -- complex, decomposition attempted."""

    async def test_weather_decomposition(self) -> None:
        store = _make_store()
        responses = [
            _completion(_weather_decomposition(), tokens_in=50, tokens_out=100),
            _completion("Amsterdam: 15C, sunny"),
            _completion("London: 12C, rainy"),
            _completion("Berlin: 10C, cloudy"),
            _completion("Summary: Amsterdam 15C, London 12C, Berlin 10C"),
        ]
        provider = MockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Weather in Amsterdam, London, Berlin"
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output != ""
        # Multiple LLM calls (decompose + leaves)
        assert len(provider.call_log) >= 2
        assert result.total_nodes >= 1

        store.close()


class TestCanonical3SequentialTask:
    """Test 3: Read README.md, find TODOs -- sequential dependency chain."""

    async def test_sequential_task(self) -> None:
        store = _make_store()
        responses = [
            _completion(_sequential_decomposition(), tokens_in=40, tokens_out=80),
            _completion("# README\n\n- TODO: fix tests\n- TODO: add docs"),
            _completion("Found 2 TODOs:\n1. fix tests\n2. add docs"),
        ]
        provider = MockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("Read README.md, find TODOs")

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output != ""
        assert result.total_nodes >= 1

        store.close()


class TestCanonical4ComplexResearch:
    """Test 4: Compare TU/e vs TUM for AI masters -- complex research task."""

    async def test_complex_research(self) -> None:
        store = _make_store()
        responses = [
            _completion(_research_decomposition(), tokens_in=50, tokens_out=120),
            _completion("TU/e offers AI track in CSE masters, strong in systems"),
            _completion("TUM offers Informatics with AI specialization, strong in ML"),
            _completion(
                "Comparison: TU/e focuses on systems, TUM focuses on ML. "
                "Both strong for AI masters."
            ),
        ]
        provider = MockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Compare TU/e vs TUM for AI masters"
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output != ""
        assert result.total_nodes >= 1

        store.close()


class TestCanonical5RepeatedTaskPatternCheck:
    """Test 5: Run weather task twice with same orchestrator.

    Second run checks whether pattern matching or caching behavior is stable.
    The orchestrator should handle repeated identical tasks gracefully.
    """

    async def test_repeated_task_pattern_check(self) -> None:
        store = _make_store()
        # Provide enough responses for two full pipeline runs.
        # Each run: 1 decompose + 4 leaves = 5 calls. Two runs = 10 total.
        leaf_responses = [
            _completion("Amsterdam: 15C"),
            _completion("London: 12C"),
            _completion("Berlin: 10C"),
            _completion("Summary: 3 cities weather"),
        ]
        responses = [
            _completion(_weather_decomposition(), tokens_in=50, tokens_out=100),
            *leaf_responses,
            _completion(_weather_decomposition(), tokens_in=50, tokens_out=100),
            *leaf_responses,
        ]
        provider = MockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        task = "Weather in Amsterdam, London, Berlin"

        result_1 = await orchestrator.process(task)
        assert result_1.success is True
        assert result_1.output != ""

        calls_after_first = len(provider.call_log)

        result_2 = await orchestrator.process(task)
        assert result_2.success is True
        assert result_2.output != ""

        calls_after_second = len(provider.call_log)

        # Second run should also produce valid output. It may use fewer calls
        # if pattern cache kicks in, or the same number if not.
        assert calls_after_second >= calls_after_first
        assert result_2.total_nodes >= 1

        store.close()
