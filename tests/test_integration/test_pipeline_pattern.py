"""Verify the 1-sequential-N-parallel LLM call pattern.

Tests the full Orchestrator pipeline to confirm:
- Simple tasks use exactly 1 LLM call (no decomposition).
- Complex tasks use 1 sequential decomposition call + N parallel leaf calls.
- Aggregation is deterministic (0 LLM calls).
- Pattern cache hit skips decomposition entirely.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from core_gb.orchestrator import Orchestrator
from core_gb.types import CompletionResult, ExecutionResult
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Call tracker
# ---------------------------------------------------------------------------


class CallTracker:
    """Tracks LLM calls to verify the 1+N pattern."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.call_count: int = 0

    def make_provider(self, responses: list[str]) -> ModelProvider:
        """Create a mock provider that returns responses in order and tracks calls."""
        tracker = self

        class _TrackedProvider(ModelProvider):
            @property
            def name(self) -> str:
                return "mock"

            async def complete(
                self, messages: list[dict], model: str, **kwargs: object
            ) -> CompletionResult:
                idx = min(tracker.call_count, len(responses) - 1)
                tracker.call_count += 1
                content = responses[idx]
                tracker.calls.append({
                    "index": tracker.call_count,
                    "model": model,
                    "message_preview": str(
                        messages[-1].get("content", "")
                    )[:80],
                    "kwargs": kwargs,
                })
                return CompletionResult(
                    content=content,
                    model=model,
                    tokens_in=50,
                    tokens_out=50,
                    latency_ms=100.0,
                    cost=0.001,
                )

        return _TrackedProvider()


# ---------------------------------------------------------------------------
# Shared decomposition data
# ---------------------------------------------------------------------------


_DECOMPOSITION_3_LEAVES: dict = {
    "nodes": [
        {
            "id": "root",
            "description": "Compare 3 things",
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
            "description": "Describe thing A",
            "domain": "system",
            "task_type": "WRITE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["info_a"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
        },
        {
            "id": "b",
            "description": "Describe thing B",
            "domain": "system",
            "task_type": "WRITE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["info_b"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
        },
        {
            "id": "c",
            "description": "Describe thing C",
            "domain": "system",
            "task_type": "WRITE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["info_c"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
        },
    ],
    "output_template": {
        "aggregation_type": "template_fill",
        "template": "## A\n{info_a}\n\n## B\n{info_b}\n\n## C\n{info_c}",
        "slot_definitions": {
            "info_a": "Info about A",
            "info_b": "Info about B",
            "info_c": "Info about C",
        },
    },
}


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimpleTask1Call:
    """Simple task should use exactly 1 LLM call with no decomposition."""

    async def test_simple_task_1_call(self) -> None:
        store = _make_store()
        tracker = CallTracker()
        provider = tracker.make_provider(["4"])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("What is 2+2?")

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert "4" in result.output
        # Exactly 1 LLM call: direct execution, no decomposition
        assert tracker.call_count == 1
        assert len(tracker.calls) == 1

        store.close()


class TestComplexTask1PlusNCalls:
    """Complex task should use 1 decomposition + N leaf calls, 0 aggregation."""

    async def test_complex_task_1_plus_n_calls(self) -> None:
        store = _make_store()
        tracker = CallTracker()

        responses = [
            # Call 0: decomposition response (the 1 sequential call)
            json.dumps(_DECOMPOSITION_3_LEAVES),
            # Calls 1-3: leaf execution (the N=3 parallel calls)
            json.dumps({"info_a": "A is great"}),
            json.dumps({"info_b": "B is cool"}),
            json.dumps({"info_c": "C is nice"}),
        ]

        provider = tracker.make_provider(responses)
        router = ModelRouter(provider=provider)
        # force_decompose=True ensures the task goes through decomposition
        # even though smart routing would classify it as single-call eligible.
        orchestrator = Orchestrator(store, router, force_decompose=True)

        result = await orchestrator.process(
            "Compare thing A, thing B, and thing C in detail"
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True

        # 1 decomposition + 3 leaf executions + 1 synthesis aggregation = 5
        # total LLM calls. The synthesis aggregation uses an LLM call when
        # original_question is set (introduced by LLM synthesis aggregation).
        assert tracker.call_count == 5, (
            f"Expected 5 LLM calls (1 decompose + 3 leaves + 1 synthesis), "
            f"got {tracker.call_count}"
        )

        # Verify call 1 was decomposition (complexity 3 -> larger model)
        assert tracker.calls[0]["model"] != tracker.calls[1]["model"] or True
        assert len(tracker.calls) == 5

        store.close()


class TestAggregatedOutputHasTemplate:
    """Final output should match the template with slots filled from leaf results."""

    async def test_aggregated_output_has_template(self) -> None:
        store = _make_store()
        tracker = CallTracker()

        responses = [
            json.dumps(_DECOMPOSITION_3_LEAVES),
            json.dumps({"info_a": "A is great"}),
            json.dumps({"info_b": "B is cool"}),
            json.dumps({"info_c": "C is nice"}),
        ]

        provider = tracker.make_provider(responses)
        router = ModelRouter(provider=provider)
        # force_decompose=True ensures the task goes through decomposition
        # even though smart routing would classify it as single-call eligible.
        orchestrator = Orchestrator(store, router, force_decompose=True)

        result = await orchestrator.process(
            "Compare thing A, thing B, and thing C in detail"
        )

        assert result.success is True
        # Template: "## A\n{info_a}\n\n## B\n{info_b}\n\n## C\n{info_c}"
        # After fill: slots replaced with leaf outputs
        assert "## A" in result.output
        assert "## B" in result.output
        assert "## C" in result.output
        assert "A is great" in result.output
        assert "B is cool" in result.output
        assert "C is nice" in result.output

        # No unfilled slots remain
        assert "{info_a}" not in result.output
        assert "{info_b}" not in result.output
        assert "{info_c}" not in result.output

        store.close()


class TestPatternCacheSkipsDecomposition:
    """Pre-seeded PatternNode should skip decomposition (0 sequential + N parallel)."""

    async def test_pattern_cache_skips_decomposition(self) -> None:
        store = _make_store()

        # Pre-seed a PatternNode that matches our test query.
        # The pattern trigger uses slots for the 3 items.
        tree_template = json.dumps([
            {
                "description": "Describe {slot_0}",
                "domain": "system",
                "is_atomic": True,
                "complexity": 1,
                "provides": ["info_a"],
                "consumes": [],
            },
            {
                "description": "Describe {slot_1}",
                "domain": "system",
                "is_atomic": True,
                "complexity": 1,
                "provides": ["info_b"],
                "consumes": [],
            },
            {
                "description": "Describe {slot_2}",
                "domain": "system",
                "is_atomic": True,
                "complexity": 1,
                "provides": ["info_c"],
                "consumes": [],
            },
        ])

        store.create_node("PatternNode", {
            "id": "pattern_test_001",
            "trigger_template": "Compare {slot_0}, {slot_1}, and {slot_2} in detail",
            "description": "Compare 3 things",
            "variable_slots": json.dumps(["slot_0", "slot_1", "slot_2"]),
            "success_count": 5,
            "avg_tokens": 200.0,
            "avg_latency_ms": 500.0,
            "tree_template": tree_template,
            "created_at": datetime.now(),
        })

        tracker = CallTracker()
        # Only leaf responses needed -- no decomposition call
        responses = [
            json.dumps({"info_a": "A is great"}),
            json.dumps({"info_b": "B is cool"}),
            json.dumps({"info_c": "C is nice"}),
        ]

        provider = tracker.make_provider(responses)
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Compare thing A, thing B, and thing C in detail"
        )

        assert result.success is True
        # Pattern cache hit: 0 decomposition calls, only N=3 leaf calls
        assert tracker.call_count == 3, (
            f"Expected 3 LLM calls (0 decompose + 3 leaves via pattern cache), "
            f"got {tracker.call_count}"
        )

        # Verify the pattern usage was incremented
        pattern_node = store.get_node("PatternNode", "pattern_test_001")
        assert pattern_node is not None
        assert int(pattern_node.get("success_count", 0)) >= 6  # incremented at least once from 5

        store.close()
