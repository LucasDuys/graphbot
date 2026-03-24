"""Tests for pattern matching and graph updating wired into the Orchestrator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from core_gb.orchestrator import Orchestrator
from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    Pattern,
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


def _simple_completion(content: str, tokens: int = 10, cost: float = 0.001) -> CompletionResult:
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens,
        tokens_out=tokens,
        latency_ms=50.0,
        cost=cost,
    )


class SequentialMockProvider(ModelProvider):
    """Mock provider that returns different responses per call."""

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


def _weather_pattern() -> Pattern:
    """A pattern for weather queries with a tree template."""
    tree_template = json.dumps([
        {
            "description": "Get weather for 3 cities",
            "domain": "synthesis",
            "is_atomic": False,
            "complexity": 2,
            "provides": [],
            "consumes": [],
        },
        {
            "description": "Get {slot_0} weather",
            "domain": "web",
            "is_atomic": True,
            "complexity": 1,
            "provides": ["weather_0"],
            "consumes": [],
        },
        {
            "description": "Get {slot_1} weather",
            "domain": "web",
            "is_atomic": True,
            "complexity": 1,
            "provides": ["weather_1"],
            "consumes": [],
        },
        {
            "description": "Summarize weather",
            "domain": "synthesis",
            "is_atomic": True,
            "complexity": 1,
            "provides": ["summary"],
            "consumes": ["weather_0", "weather_1"],
        },
    ])
    return Pattern(
        id="pat-001",
        trigger="Weather in {slot_0} and {slot_1}",
        description="Pattern for weather comparison",
        variable_slots=("slot_0", "slot_1"),
        tree_template=tree_template,
        success_count=3,
        avg_tokens=100.0,
        avg_latency_ms=500.0,
    )


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
                "children": ["w1", "w2", "agg"],
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
                "id": "agg",
                "description": "Summarize weather",
                "domain": "synthesis",
                "task_type": "WRITE",
                "complexity": 1,
                "depends_on": ["w1", "w2"],
                "provides": ["summary"],
                "consumes": ["weather_ams", "weather_lon"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPatternCacheHit:
    """When a matching pattern exists, decomposer should be skipped."""

    async def test_pattern_cache_hit(self) -> None:
        store = _make_store()
        pattern = _weather_pattern()

        # Responses for the 3 leaf executions (pattern hit skips decomposition)
        responses = [
            _simple_completion("Amsterdam: 15C sunny"),
            _simple_completion("London: 12C rainy"),
            _simple_completion("Summary of weather"),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        # Patch pattern_store.load_all to return our pattern
        with patch.object(
            orchestrator._pattern_store, "load_all", return_value=[pattern]
        ) as mock_load, patch.object(
            orchestrator._pattern_store, "increment_usage"
        ) as mock_increment, patch.object(
            orchestrator._decomposer, "decompose", new_callable=AsyncMock
        ) as mock_decompose:
            result = await orchestrator.process("Weather in Amsterdam and London")

        assert result.success is True
        # Decomposer should NOT have been called (pattern cache hit)
        mock_decompose.assert_not_called()
        # Pattern usage should have been incremented
        mock_increment.assert_called_once_with("pat-001")

        store.close()


class TestNoPatternFallsThrough:
    """When no patterns exist, decomposer should be called."""

    async def test_no_pattern_falls_through(self) -> None:
        store = _make_store()
        decomposition_json = json.dumps(_weather_tree_json())

        responses = [
            _simple_completion(decomposition_json, tokens=50, cost=0.01),
            _simple_completion("Amsterdam: 15C sunny"),
            _simple_completion("London: 12C rainy"),
            _simple_completion("Summary of weather"),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router, force_decompose=True)

        # Patch load_all to return empty list (no patterns)
        with patch.object(
            orchestrator._pattern_store, "load_all", return_value=[]
        ):
            result = await orchestrator.process(
                "Compare the weather in Amsterdam, London, and Berlin"
            )

        assert result.success is True
        # Decomposer was called (provider got at least 1 decompose call + leaf calls)
        assert len(provider.call_log) >= 2

        store.close()


class TestGraphUpdatedAfterExecution:
    """Graph should be updated after simple task execution."""

    async def test_graph_updated_after_execution(self) -> None:
        store = _make_store()
        provider = SequentialMockProvider([_simple_completion("4")])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        # Patch load_all to return empty (no patterns)
        with patch.object(
            orchestrator._pattern_store, "load_all", return_value=[]
        ):
            result = await orchestrator.process("What is 2+2?")

        assert result.success is True

        # Verify that a Task node exists in the graph after execution
        rows = store.query("MATCH (t:Task) RETURN t.id, t.status")
        assert len(rows) >= 1
        task_row = rows[0]
        assert task_row.get("t.status") == "completed"

        # Verify an ExecutionTree node was also created
        tree_rows = store.query("MATCH (et:ExecutionTree) RETURN et.id")
        assert len(tree_rows) >= 1

        store.close()


class TestPatternExtractedAfterComplexTask:
    """After a complex task with multiple leaves, a PatternNode should exist in the graph."""

    async def test_pattern_extracted_after_complex_task(self) -> None:
        store = _make_store()
        decomposition_json = json.dumps(_weather_tree_json())

        responses = [
            _simple_completion(decomposition_json, tokens=50, cost=0.01),
            _simple_completion("Amsterdam: 15C sunny"),
            _simple_completion("London: 12C rainy"),
            _simple_completion("Summary of weather"),
        ]
        provider = SequentialMockProvider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router, force_decompose=True)

        # Patch load_all to return empty (no patterns)
        with patch.object(
            orchestrator._pattern_store, "load_all", return_value=[]
        ):
            result = await orchestrator.process(
                "Compare the weather in Amsterdam, London, and Berlin"
            )

        assert result.success is True

        # Verify a PatternNode was created in the graph
        pattern_rows = store.query(
            "MATCH (p:PatternNode) RETURN p.id, p.trigger_template"
        )
        assert len(pattern_rows) >= 1

        store.close()
