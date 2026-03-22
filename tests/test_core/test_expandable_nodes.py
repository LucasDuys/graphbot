"""Tests for expandable node flag and lazy DAG expansion in DAGExecutor."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    Domain,
    ExecutionResult,
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

    async def execute(
        self,
        task: str,
        complexity: int = 1,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
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
    expandable: bool = False,
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
        expandable=expandable,
    )


class TestExpandableFieldDefaults:
    """Verify the expandable field on TaskNode."""

    def test_expandable_defaults_to_false(self) -> None:
        """TaskNode.expandable defaults to False."""
        node = TaskNode(id="t1", description="test")
        assert node.expandable is False

    def test_expandable_can_be_set_true(self) -> None:
        """TaskNode.expandable can be explicitly set to True."""
        node = TaskNode(id="t1", description="test", expandable=True)
        assert node.expandable is True

    def test_expandable_false_is_explicit(self) -> None:
        """TaskNode.expandable=False is explicit and stays False."""
        node = TaskNode(id="t1", description="test", expandable=False)
        assert node.expandable is False


class TestExpandableFalseExecutesDirectly:
    """When expandable=False, nodes execute directly without decomposition."""

    async def test_non_expandable_executes_directly(self) -> None:
        """A node with expandable=False is executed as-is by the executor."""
        mock = MockSimpleExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A", expandable=False),
            _make_leaf("b", "Task B", expandable=False),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert len(mock.call_order) == 2
        assert "Task A" in mock.call_order
        assert "Task B" in mock.call_order


class TestExpandableTrueDecomposesOnDemand:
    """When expandable=True, the node is decomposed into a sub-DAG before execution."""

    async def test_expandable_node_triggers_decomposition(self) -> None:
        """An expandable node should be decomposed into sub-nodes and those
        sub-nodes should be executed instead of the original node."""
        mock = MockSimpleExecutor(delay=0.0)

        # Sub-DAG returned by decomposer: two leaf nodes
        sub_leaf_1 = TaskNode(
            id="sub_1",
            description="Sub Task 1",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
            provides=["sub_data_1"],
        )
        sub_leaf_2 = TaskNode(
            id="sub_2",
            description="Sub Task 2",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
            requires=["sub_1"],
            consumes=["sub_data_1"],
        )

        mock_decomposer = AsyncMock()
        mock_decomposer.decompose.return_value = [sub_leaf_1, sub_leaf_2]

        dag = DAGExecutor(executor=mock, max_concurrency=10)
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("a", "Complex expandable task", expandable=True),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # The decomposer should have been called with the node's description
        mock_decomposer.decompose.assert_called_once_with("Complex expandable task")
        # The original node should NOT have been executed directly
        assert "Complex expandable task" not in mock.call_order
        # The sub-nodes should have been executed
        assert any("Sub Task 1" in call for call in mock.call_order)
        assert any("Sub Task 2" in call for call in mock.call_order)

    async def test_expandable_mixed_with_non_expandable(self) -> None:
        """Mix of expandable and non-expandable nodes: only expandable ones decompose."""
        mock = MockSimpleExecutor(delay=0.0)

        sub_leaf = TaskNode(
            id="sub_x",
            description="Expanded Sub X",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
        )

        mock_decomposer = AsyncMock()
        mock_decomposer.decompose.return_value = [sub_leaf]

        dag = DAGExecutor(executor=mock, max_concurrency=10)
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("a", "Normal task", expandable=False),
            _make_leaf("b", "Expandable task", expandable=True),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # Decomposer called only once for the expandable node
        mock_decomposer.decompose.assert_called_once_with("Expandable task")
        # Normal task executed directly
        assert "Normal task" in mock.call_order
        # Expanded sub-node executed
        assert any("Expanded Sub X" in call for call in mock.call_order)
        # Original expandable node NOT executed
        assert "Expandable task" not in mock.call_order


class TestSubDAGInheritsDependencyPosition:
    """Sub-DAG nodes inherit the parent's dependency position in the graph."""

    async def test_sub_dag_respects_upstream_dependency(self) -> None:
        """If A -> B(expandable) -> C, then the sub-DAG from B should:
        - Have its entry nodes depend on A
        - Have C depend on the sub-DAG's terminal nodes
        """
        mock = MockSimpleExecutor(delay=0.0)

        # B expands into sub_b1 -> sub_b2
        sub_b1 = TaskNode(
            id="sub_b1",
            description="Sub B Step 1",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
            provides=["sub_b_data"],
        )
        sub_b2 = TaskNode(
            id="sub_b2",
            description="Sub B Step 2",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
            requires=["sub_b1"],
            consumes=["sub_b_data"],
            provides=["data_b"],
        )

        mock_decomposer = AsyncMock()
        mock_decomposer.decompose.return_value = [sub_b1, sub_b2]

        dag = DAGExecutor(executor=mock, max_concurrency=10)
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf(
                "b", "Expandable B",
                requires=["a"],
                consumes=["data_a"],
                provides=["data_b"],
                expandable=True,
            ),
            _make_leaf("c", "Task C", requires=["b"], consumes=["data_b"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # A must execute before any sub-B nodes
        a_idx = next(i for i, c in enumerate(mock.call_order) if "Task A" in c)
        # Match sub-B calls by checking the call STARTS with or contains
        # the sub-node description as the primary task (not in forwarded data)
        sub_b_indices = [
            i for i, c in enumerate(mock.call_order)
            if c.endswith("Sub B Step 1") or c.endswith("Sub B Step 2")
        ]
        assert len(sub_b_indices) == 2, f"Expected 2 sub-B calls, got {sub_b_indices}"
        for idx in sub_b_indices:
            assert idx > a_idx, "Sub-B nodes must execute after A"

        # C must execute after all sub-B nodes
        c_idx = next(i for i, c in enumerate(mock.call_order) if c.endswith("Task C"))
        for idx in sub_b_indices:
            assert c_idx > idx, "C must execute after all sub-B nodes"

    async def test_sub_dag_provides_keys_forwarded_to_downstream(self) -> None:
        """The terminal sub-node's provides keys are forwarded to downstream consumers."""
        mock = MockSimpleExecutor(delay=0.0)

        # B expands into a single sub-node that provides data_b
        sub_b = TaskNode(
            id="sub_b_only",
            description="Sub B produces data",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
            provides=["data_b"],
        )

        mock_decomposer = AsyncMock()
        mock_decomposer.decompose.return_value = [sub_b]

        dag = DAGExecutor(executor=mock, max_concurrency=10)
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf(
                "b", "Expandable B",
                provides=["data_b"],
                expandable=True,
            ),
            _make_leaf("c", "Task C", requires=["b"], consumes=["data_b"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # C should have received forwarded data from sub_b
        c_call = next(c for c in mock.call_order if "Task C" in c)
        assert "<forwarded_data>" in c_call
        assert "data_b" in c_call


class TestExpandableWithoutDecomposer:
    """When no decomposer is available, expandable nodes fall back to direct execution."""

    async def test_expandable_without_decomposer_executes_directly(self) -> None:
        """If no decomposer is set, expandable nodes should still execute directly."""
        mock = MockSimpleExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)
        # Do not set dag._decomposer

        nodes = [
            _make_leaf("a", "Expandable but no decomposer", expandable=True),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert "Expandable but no decomposer" in mock.call_order
