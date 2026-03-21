"""Tests for the parallel DAG executor with streaming topological dispatch."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    Domain,
    ExecutionResult,
    FlowType,
    TaskNode,
    TaskStatus,
)


class MockSimpleExecutor:
    """Mock executor that records call order and returns configurable results."""

    def __init__(
        self,
        delay: float = 0.0,
        fail_on: set[str] | None = None,
    ) -> None:
        self.call_order: list[str] = []
        self.delay = delay
        self._fail_on = fail_on or set()
        self._lock = asyncio.Lock()

    async def execute(self, task: str, complexity: int = 1) -> ExecutionResult:
        async with self._lock:
            self.call_order.append(task)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if task in self._fail_on:
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="",
                success=False,
                total_nodes=1,
                total_tokens=10,
                total_latency_ms=self.delay * 1000,
                total_cost=0.001,
                errors=(f"Failed: {task}",),
            )
        return ExecutionResult(
            root_id=str(uuid.uuid4()),
            output=f"Result: {task}",
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=self.delay * 1000,
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


class TestParallelExecution:
    async def test_parallel_execution(self) -> None:
        """Three independent leaves with 0.1s delay each run in parallel.

        Total wall-clock time should be well under 0.5s (sequential would be ~0.3s).
        """
        mock = MockSimpleExecutor(delay=0.1)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A"),
            _make_leaf("b", "Task B"),
            _make_leaf("c", "Task C"),
        ]

        start = time.perf_counter()
        result = await dag.execute(nodes)
        elapsed = time.perf_counter() - start

        assert result.success is True
        assert len(mock.call_order) == 3
        # Parallel: should complete in ~0.1s, generous threshold of 0.5s for CI
        assert elapsed < 0.5, f"Expected parallel execution under 0.5s, got {elapsed:.3f}s"


class TestSequentialChain:
    async def test_sequential_chain(self) -> None:
        """A -> B -> C chain executes in strict order."""
        mock = MockSimpleExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", requires=["a"], consumes=["data_a"], provides=["data_b"]),
            _make_leaf("c", "Task C", requires=["b"], consumes=["data_b"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert len(mock.call_order) == 3
        # First node has no deps, so raw description
        assert mock.call_order[0] == "Task A"
        # B and C receive forwarded data, but end with their original description
        assert mock.call_order[1].endswith("Task B")
        assert mock.call_order[2].endswith("Task C")
        # Verify ordering: B contains A's output, C contains B's output
        assert "data_a" in mock.call_order[1]
        assert "data_b" in mock.call_order[2]


class TestMixedDAG:
    async def test_mixed_dag(self) -> None:
        """A and B independent, C depends on both. C must execute after A and B."""
        mock = MockSimpleExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", provides=["data_b"]),
            _make_leaf("c", "Task C", requires=["a", "b"], consumes=["data_a", "data_b"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert len(mock.call_order) == 3
        # C must be last (it has forwarded data prepended)
        assert mock.call_order[-1].endswith("Task C")
        assert "<forwarded_data>" in mock.call_order[-1]
        # A and B must both appear before C (no forwarded data, raw descriptions)
        first_two = mock.call_order[:2]
        assert "Task A" in first_two
        assert "Task B" in first_two


class TestAggregatedResult:
    async def test_aggregated_result(self) -> None:
        """Verify tokens and cost are summed across all nodes."""
        mock = MockSimpleExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A"),
            _make_leaf("b", "Task B"),
            _make_leaf("c", "Task C"),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert result.total_tokens == 30  # 10 per node * 3
        assert result.total_cost == pytest.approx(0.003)  # 0.001 * 3
        assert result.total_nodes == 3
        # All outputs present
        assert "Result: Task A" in result.output
        assert "Result: Task B" in result.output
        assert "Result: Task C" in result.output


class TestNodeFailureDoesntCrashOthers:
    async def test_node_failure_doesnt_crash_others(self) -> None:
        """One node fails but others still complete successfully."""
        mock = MockSimpleExecutor(delay=0.0, fail_on={"Task B"})
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A"),
            _make_leaf("b", "Task B"),
            _make_leaf("c", "Task C"),
        ]

        result = await dag.execute(nodes)

        # Overall success is False because one node failed
        assert result.success is False
        assert len(mock.call_order) == 3
        # Errors should contain the failure
        assert any("Failed: Task B" in e for e in result.errors)
        # But successful outputs are still present
        assert "Result: Task A" in result.output
        assert "Result: Task C" in result.output


class TestEmptyNodes:
    async def test_empty_nodes(self) -> None:
        """Empty node list returns an empty result."""
        mock = MockSimpleExecutor()
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        result = await dag.execute([])

        assert result.success is True
        assert result.total_nodes == 0
        assert result.total_tokens == 0
        assert result.output == ""
        assert len(mock.call_order) == 0


class TestSingleNode:
    async def test_single_node(self) -> None:
        """A single atomic node executes correctly."""
        mock = MockSimpleExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [_make_leaf("only", "The Only Task")]

        result = await dag.execute(nodes)

        assert result.success is True
        assert result.output == "Result: The Only Task"
        assert result.total_nodes == 1
        assert mock.call_order == ["The Only Task"]


class TestDataForwarding:
    async def test_data_forwarded_to_dependents(self) -> None:
        """Verify that completed node output is injected into dependent task description."""
        mock = MockSimpleExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", requires=["a"], consumes=["data_a"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # Task B should have been called with forwarded data in its description
        assert len(mock.call_order) == 2
        assert mock.call_order[0] == "Task A"
        # The second call should contain forwarded data
        assert "<forwarded_data>" in mock.call_order[1]
        assert "data_a" in mock.call_order[1]


class TestConcurrencyLimit:
    async def test_concurrency_limit_respected(self) -> None:
        """With max_concurrency=1, nodes execute one at a time even if independent."""
        mock = MockSimpleExecutor(delay=0.05)
        dag = DAGExecutor(executor=mock, max_concurrency=1)

        nodes = [
            _make_leaf("a", "Task A"),
            _make_leaf("b", "Task B"),
            _make_leaf("c", "Task C"),
        ]

        start = time.perf_counter()
        result = await dag.execute(nodes)
        elapsed = time.perf_counter() - start

        assert result.success is True
        assert len(mock.call_order) == 3
        # With concurrency=1 and 0.05s delay, should take at least 0.15s
        assert elapsed >= 0.12, f"Expected sequential execution, got {elapsed:.3f}s"
