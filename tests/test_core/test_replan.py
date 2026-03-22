"""Tests for intermediate result feedback with optional re-planning (T149, R004 AC 2-4).

Validates that:
- Orchestrator hooks into wave_complete to trigger re-planning
- Re-planning is off by default, enabled via enable_replan config
- On re-plan: remaining unexecuted nodes replaced with new decomposition
- Completed results are preserved across re-planning
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)
from core_gb.wave_event import WaveCompleteEvent


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockExecutorForReplan:
    """Mock executor that returns configurable results per task description.

    Records call order for assertions. Optionally delays execution.
    """

    def __init__(
        self,
        results: dict[str, str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.call_order: list[str] = []
        self._results = results or {}
        self._delay = delay
        self._lock = asyncio.Lock()

    async def execute(
        self,
        task: str,
        complexity: int = 1,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        async with self._lock:
            self.call_order.append(task)
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        # Look up by substring match (task text may have forwarded data prepended)
        output = f"Result: {task[:60]}"
        for key, val in self._results.items():
            if key in task:
                output = val
                break

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReplanDisabledByDefault:
    """Re-planning is off by default -- no re-plan callback fires."""

    async def test_no_replan_when_disabled(self) -> None:
        """With no on_replan set, waves execute normally without re-planning."""
        mock_exec = MockExecutorForReplan()
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)

        # Wave 1: a, b (parallel). Wave 2: c (depends on a, b).
        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", provides=["data_b"]),
            _make_leaf(
                "c", "Task C",
                requires=["a", "b"],
                consumes=["data_a", "data_b"],
            ),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert len(mock_exec.call_order) == 3
        # on_replan is None by default
        assert dag._on_replan is None


class TestReplanCallbackInvoked:
    """When on_replan is set and returns None, execution proceeds normally."""

    async def test_replan_callback_called_when_remaining_nodes_exist(self) -> None:
        """The on_replan callback is invoked after waves that have remaining nodes."""
        wave_events: list[WaveCompleteEvent] = []

        async def track_replan(event: WaveCompleteEvent) -> list[TaskNode] | None:
            wave_events.append(event)
            return None  # No re-planning, just track

        mock_exec = MockExecutorForReplan()
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)
        dag.set_replan_callback(track_replan)

        # Wave 0: a, b (parallel). Wave 1: c (depends on a, b).
        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", provides=["data_b"]),
            _make_leaf(
                "c", "Task C",
                requires=["a", "b"],
                consumes=["data_a", "data_b"],
            ),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # Callback fires after wave 0 (a, b) because c is still remaining.
        # After wave 1 (c), no remaining nodes, so callback is not invoked.
        assert len(wave_events) == 1
        assert wave_events[0].wave_index == 0
        assert len(wave_events[0].remaining_nodes) > 0
        assert "c" in wave_events[0].remaining_nodes


class TestReplanReplacesRemainingNodes:
    """When on_replan returns new nodes, they replace the remaining unexecuted ones."""

    async def test_replan_replaces_remaining_with_new_decomposition(self) -> None:
        """After wave 1 completes, re-planning replaces wave 2 nodes with new ones.

        Original DAG: a -> b -> c (3 waves sequential)
        After wave 1 (a completes), re-plan replaces b and c with d and e (parallel).
        Final execution: a, then d and e in parallel.
        """
        replan_called = False

        async def replan_callback(
            event: WaveCompleteEvent,
        ) -> list[TaskNode] | None:
            nonlocal replan_called
            # Only re-plan after wave 0 (first wave)
            if event.wave_index != 0:
                return None
            replan_called = True

            # Return replacement nodes: d and e, both depend on a's output
            return [
                _make_leaf(
                    "d", "Replanned Task D",
                    provides=["data_d"],
                ),
                _make_leaf(
                    "e", "Replanned Task E",
                    provides=["data_e"],
                ),
            ]

        mock_exec = MockExecutorForReplan()
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)
        dag.set_replan_callback(replan_callback)

        # Original: a -> b -> c (sequential chain)
        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf(
                "b", "Task B",
                requires=["a"],
                consumes=["data_a"],
                provides=["data_b"],
            ),
            _make_leaf(
                "c", "Task C",
                requires=["b"],
                consumes=["data_b"],
            ),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert replan_called is True

        # Task A should have executed
        assert any("Task A" in call for call in mock_exec.call_order)
        # Original Task B and Task C should NOT have executed
        assert not any(
            "Task B" in call
            for call in mock_exec.call_order
            if "forwarded_data" not in call
        )
        assert not any(
            "Task C" in call
            for call in mock_exec.call_order
            if "forwarded_data" not in call
        )
        # Replanned tasks D and E should have executed
        assert any("Replanned Task D" in call for call in mock_exec.call_order)
        assert any("Replanned Task E" in call for call in mock_exec.call_order)


class TestReplanPreservesCompletedResults:
    """Completed results from earlier waves are preserved after re-planning."""

    async def test_completed_results_preserved_after_replan(self) -> None:
        """Results from wave 0 nodes appear in the final aggregated output."""
        replan_count = 0

        async def replan_callback(
            event: WaveCompleteEvent,
        ) -> list[TaskNode] | None:
            nonlocal replan_count
            if event.wave_index != 0:
                return None
            replan_count += 1
            return [
                _make_leaf("new_agg", "New aggregation step"),
            ]

        mock_exec = MockExecutorForReplan(
            results={
                "Task A": "Alpha output",
                "Task B": "Beta output",
                "New aggregation": "Final summary",
            },
        )
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)
        dag.set_replan_callback(replan_callback)

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", provides=["data_b"]),
            _make_leaf(
                "c", "Task C",
                requires=["a", "b"],
                consumes=["data_a", "data_b"],
            ),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert replan_count == 1
        # Alpha and Beta outputs from wave 0 must be in the final result
        assert "Alpha output" in result.output
        assert "Beta output" in result.output


class TestReplanCallbackReceivesAccumulatedResults:
    """The re-plan callback receives accumulated results from completed waves."""

    async def test_accumulated_results_in_callback(self) -> None:
        """The WaveCompleteEvent passed to on_replan contains outputs from all completed nodes."""
        captured_event: list[WaveCompleteEvent] = []

        async def replan_callback(
            event: WaveCompleteEvent,
        ) -> list[TaskNode] | None:
            captured_event.append(event)
            return None  # No actual re-planning

        mock_exec = MockExecutorForReplan(
            results={
                "Task A": "Result from A",
                "Task B": "Result from B",
            },
        )
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)
        dag.set_replan_callback(replan_callback)

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", provides=["data_b"]),
            _make_leaf(
                "c", "Task C",
                requires=["a", "b"],
                consumes=["data_a", "data_b"],
            ),
        ]

        await dag.execute(nodes)

        # First wave event should have accumulated results for a and b
        assert len(captured_event) >= 1
        first_event = captured_event[0]
        assert len(first_event.accumulated_results) == 2
        # The values should contain actual outputs
        values = list(first_event.accumulated_results.values())
        assert any("Result from A" in v for v in values)
        assert any("Result from B" in v for v in values)


class TestReplanCallbackReturnsNone:
    """When on_replan returns None, execution continues unchanged."""

    async def test_none_return_means_no_replan(self) -> None:
        """Returning None from on_replan preserves original execution plan."""
        async def no_replan(
            event: WaveCompleteEvent,
        ) -> list[TaskNode] | None:
            return None

        mock_exec = MockExecutorForReplan()
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)
        dag.set_replan_callback(no_replan)

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf(
                "b", "Task B",
                requires=["a"],
                consumes=["data_a"],
            ),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert len(mock_exec.call_order) == 2
        assert mock_exec.call_order[0] == "Task A"
        assert "Task B" in mock_exec.call_order[1]


class TestReplanCallbackExceptionHandled:
    """If the on_replan callback raises, execution continues without re-planning."""

    async def test_exception_in_replan_handled_gracefully(self) -> None:
        """A failing on_replan callback does not crash the DAG execution."""
        async def failing_replan(
            event: WaveCompleteEvent,
        ) -> list[TaskNode] | None:
            raise RuntimeError("Replan callback crashed")

        mock_exec = MockExecutorForReplan()
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)
        dag.set_replan_callback(failing_replan)

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf(
                "b", "Task B",
                requires=["a"],
                consumes=["data_a"],
            ),
        ]

        result = await dag.execute(nodes)

        # Execution should complete successfully despite the callback error
        assert result.success is True
        assert len(mock_exec.call_order) == 2


class TestOrchestratorReplanWiring:
    """Orchestrator wires up re-plan callback when enable_replan is True."""

    async def test_orchestrator_enable_replan_false_by_default(self) -> None:
        """By default, Orchestrator does not set an on_replan callback."""
        from core_gb.orchestrator import Orchestrator
        from graph.store import GraphStore
        from models.router import ModelRouter
        from core_gb.types import CompletionResult
        from tests.test_core.test_orchestrator import (
            SequentialMockProvider,
            _simple_completion,
        )

        store = GraphStore(db_path=None)
        store.initialize()
        provider = SequentialMockProvider([_simple_completion("hello")])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        assert orchestrator._dag_executor._on_replan is None

        store.close()

    async def test_orchestrator_enable_replan_true_wires_callback(self) -> None:
        """When enable_replan=True, Orchestrator sets on_replan on the DAGExecutor."""
        from core_gb.orchestrator import Orchestrator
        from graph.store import GraphStore
        from models.router import ModelRouter
        from core_gb.types import CompletionResult
        from tests.test_core.test_orchestrator import (
            SequentialMockProvider,
            _simple_completion,
        )

        store = GraphStore(db_path=None)
        store.initialize()
        provider = SequentialMockProvider([_simple_completion("hello")])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router, enable_replan=True)

        assert orchestrator._dag_executor._on_replan is not None

        store.close()
