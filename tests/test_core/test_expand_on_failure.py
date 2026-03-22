"""Tests for DAGExecutor.expand_node() re-decomposition on leaf failure."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.decomposer import Decomposer
from core_gb.types import (
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockExecutorForExpansion:
    """Mock executor that fails specific task descriptions, then succeeds on sub-tasks.

    Tracks call_order for assertion. ``fail_on`` is a set of description
    substrings -- if any substring appears in the task text, the execution
    fails.
    """

    def __init__(
        self,
        fail_on: set[str] | None = None,
    ) -> None:
        self.call_order: list[str] = []
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

        should_fail = any(pat in task for pat in self._fail_on)
        if should_fail:
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="",
                success=False,
                total_nodes=1,
                total_tokens=10,
                total_latency_ms=1.0,
                total_cost=0.001,
                errors=(f"Failed: {task[:60]}",),
            )
        return ExecutionResult(
            root_id=str(uuid.uuid4()),
            output=f"Result: {task[:60]}",
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=1.0,
            total_cost=0.001,
        )


class MockDecomposerForExpansion:
    """Mock decomposer that returns a configurable sub-DAG for any task.

    ``sub_dag_factory`` is called with (task_description, failure_context)
    and must return list[TaskNode].  ``call_count`` tracks how many times
    decompose was invoked.
    """

    def __init__(
        self,
        sub_dag_factory: callable | None = None,
    ) -> None:
        self._factory = sub_dag_factory or self._default_factory
        self.call_count: int = 0
        self.call_args: list[tuple[str, str | None]] = []

    async def decompose(
        self,
        task: str,
        context: object | None = None,
        max_depth: int = 3,
        failure_context: str | None = None,
    ) -> list[TaskNode]:
        self.call_count += 1
        self.call_args.append((task, failure_context))
        return self._factory(task, failure_context)

    @staticmethod
    def _default_factory(
        task: str, failure_context: str | None
    ) -> list[TaskNode]:
        """Return two simple sub-leaves that always succeed."""
        sub_a_id = str(uuid.uuid4())
        sub_b_id = str(uuid.uuid4())
        return [
            TaskNode(
                id=sub_a_id,
                description=f"sub-step-1 of: {task[:40]}",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.READY,
                provides=["sub_result_1"],
            ),
            TaskNode(
                id=sub_b_id,
                description=f"sub-step-2 of: {task[:40]}",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.READY,
                requires=[sub_a_id],
                consumes=["sub_result_1"],
                provides=["sub_result_2"],
            ),
        ]


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
# Tests: leaf fails then re-decomposes into sub-leaves that succeed
# ---------------------------------------------------------------------------


class TestExpandNodeOnFailure:
    """Verify that a failed leaf triggers re-decomposition via expand_node."""

    async def test_failed_leaf_redecomposes_and_succeeds(self) -> None:
        """A leaf fails, gets re-decomposed into sub-leaves, sub-leaves succeed.

        The overall DAG should still succeed because the sub-leaves replace
        the failed node and complete successfully.
        """
        # The original leaf "hard_task" will fail; the sub-tasks will not
        mock_exec = MockExecutorForExpansion(fail_on={"hard_task"})
        mock_decomposer = MockDecomposerForExpansion()

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("a", "easy_task", provides=["data_a"]),
            _make_leaf("b", "hard_task", requires=["a"], consumes=["data_a"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # The decomposer should have been called exactly once (for the
        # failed "hard_task" node)
        assert mock_decomposer.call_count == 1
        # expansion_count should be tracked
        assert result.expansion_count == 1

    async def test_expansion_count_tracked_in_result(self) -> None:
        """ExecutionResult.expansion_count reflects how many nodes were expanded."""
        mock_exec = MockExecutorForExpansion(fail_on={"fail_me"})
        mock_decomposer = MockDecomposerForExpansion()

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("x", "fail_me"),
        ]

        result = await dag.execute(nodes)

        assert result.expansion_count == 1

    async def test_no_expansion_when_no_decomposer(self) -> None:
        """Without a decomposer, failed leaves stay failed (no expansion)."""
        mock_exec = MockExecutorForExpansion(fail_on={"hard_task"})

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        # No decomposer set

        nodes = [
            _make_leaf("a", "hard_task"),
        ]

        result = await dag.execute(nodes)

        assert result.success is False
        assert result.expansion_count == 0

    async def test_no_expansion_on_success(self) -> None:
        """Successful leaves should never trigger expansion."""
        mock_exec = MockExecutorForExpansion()
        mock_decomposer = MockDecomposerForExpansion()

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("a", "easy_task_1"),
            _make_leaf("b", "easy_task_2"),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert mock_decomposer.call_count == 0
        assert result.expansion_count == 0

    async def test_failure_context_passed_to_decomposer(self) -> None:
        """The failure error message is forwarded to the decomposer."""
        mock_exec = MockExecutorForExpansion(fail_on={"hard_task"})
        mock_decomposer = MockDecomposerForExpansion()

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("a", "hard_task"),
        ]

        await dag.execute(nodes)

        assert mock_decomposer.call_count == 1
        task_desc, failure_ctx = mock_decomposer.call_args[0]
        assert "hard_task" in task_desc
        assert failure_ctx is not None
        assert "Failed" in failure_ctx


# ---------------------------------------------------------------------------
# Tests: max expansion depth enforcement
# ---------------------------------------------------------------------------


class TestMaxExpansionDepth:
    """Verify that max_expansion_depth is enforced."""

    async def test_max_depth_zero_disables_expansion(self) -> None:
        """With max_expansion_depth=0, no expansion is attempted."""
        mock_exec = MockExecutorForExpansion(fail_on={"hard_task"})
        mock_decomposer = MockDecomposerForExpansion()

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=0,
        )
        dag._decomposer = mock_decomposer

        nodes = [_make_leaf("a", "hard_task")]

        result = await dag.execute(nodes)

        assert result.success is False
        assert mock_decomposer.call_count == 0
        assert result.expansion_count == 0

    async def test_max_depth_prevents_recursive_expansion(self) -> None:
        """When sub-leaves also fail, expansion stops at max_expansion_depth.

        Setup: max_expansion_depth=1, original leaf fails, sub-leaves also
        fail. Since depth limit is 1, no further expansion is attempted on
        the sub-leaves. The result should be a failure.
        """
        # Both original and sub-tasks fail
        mock_exec = MockExecutorForExpansion(
            fail_on={"hard_task", "sub-step-1", "sub-step-2"}
        )
        mock_decomposer = MockDecomposerForExpansion()

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=1,
        )
        dag._decomposer = mock_decomposer

        nodes = [_make_leaf("a", "hard_task")]

        result = await dag.execute(nodes)

        # Should fail because sub-leaves also fail and depth limit reached
        assert result.success is False
        # Decomposer called once (for first expansion), not again for sub-leaves
        assert mock_decomposer.call_count == 1
        assert result.expansion_count == 1

    async def test_depth_two_allows_nested_expansion(self) -> None:
        """With max_expansion_depth=2, a second level of expansion is allowed.

        Setup: original leaf fails -> sub-leaves generated -> one sub-leaf
        fails -> second expansion produces leaves that succeed.
        """
        call_count = {"n": 0}

        def cascading_factory(
            task: str, failure_context: str | None
        ) -> list[TaskNode]:
            call_count["n"] += 1
            level = call_count["n"]
            sub_a_id = str(uuid.uuid4())
            sub_b_id = str(uuid.uuid4())
            if level == 1:
                # First expansion: sub-step-1 will also fail
                return [
                    TaskNode(
                        id=sub_a_id,
                        description="level1_fails",
                        is_atomic=True,
                        domain=Domain.SYNTHESIS,
                        complexity=1,
                        status=TaskStatus.READY,
                        provides=["l1_data"],
                    ),
                    TaskNode(
                        id=sub_b_id,
                        description="level1_ok",
                        is_atomic=True,
                        domain=Domain.SYNTHESIS,
                        complexity=1,
                        status=TaskStatus.READY,
                        requires=[sub_a_id],
                        consumes=["l1_data"],
                    ),
                ]
            else:
                # Second expansion: all succeed
                return [
                    TaskNode(
                        id=sub_a_id,
                        description="level2_ok_a",
                        is_atomic=True,
                        domain=Domain.SYNTHESIS,
                        complexity=1,
                        status=TaskStatus.READY,
                        provides=["l2_data"],
                    ),
                    TaskNode(
                        id=sub_b_id,
                        description="level2_ok_b",
                        is_atomic=True,
                        domain=Domain.SYNTHESIS,
                        complexity=1,
                        status=TaskStatus.READY,
                        requires=[sub_a_id],
                        consumes=["l2_data"],
                    ),
                ]

        mock_exec = MockExecutorForExpansion(
            fail_on={"hard_task", "level1_fails"}
        )
        mock_decomposer = MockDecomposerForExpansion(
            sub_dag_factory=cascading_factory
        )

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [_make_leaf("a", "hard_task")]

        result = await dag.execute(nodes)

        assert result.success is True
        # Two expansions: original -> level1, level1_fails -> level2
        assert mock_decomposer.call_count == 2
        assert result.expansion_count == 2

    async def test_default_max_expansion_depth_is_two(self) -> None:
        """The default max_expansion_depth should be 2."""
        mock_exec = MockExecutorForExpansion()
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)
        assert dag._max_expansion_depth == 2


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestExpandNodeEdgeCases:
    """Edge cases for the expand_node mechanism."""

    async def test_decomposer_returns_empty_sub_dag(self) -> None:
        """If the decomposer returns empty sub-DAG, the original failure stands."""

        def empty_factory(
            task: str, failure_context: str | None
        ) -> list[TaskNode]:
            return []

        mock_exec = MockExecutorForExpansion(fail_on={"hard_task"})
        mock_decomposer = MockDecomposerForExpansion(
            sub_dag_factory=empty_factory
        )

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [_make_leaf("a", "hard_task")]

        result = await dag.execute(nodes)

        assert result.success is False
        assert mock_decomposer.call_count == 1

    async def test_decomposer_raises_exception(self) -> None:
        """If the decomposer raises, the original failure stands gracefully."""

        def raising_factory(
            task: str, failure_context: str | None
        ) -> list[TaskNode]:
            raise RuntimeError("Decomposer crashed")

        mock_exec = MockExecutorForExpansion(fail_on={"hard_task"})
        mock_decomposer = MockDecomposerForExpansion(
            sub_dag_factory=raising_factory
        )

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [_make_leaf("a", "hard_task")]

        result = await dag.execute(nodes)

        assert result.success is False
        assert mock_decomposer.call_count == 1

    async def test_expansion_preserves_dependency_wiring(self) -> None:
        """After expansion, downstream nodes still receive data from the sub-DAG.

        Setup: A -> B (fails, expanded) -> C
        After expansion: A -> [sub1, sub2] -> C
        C should still execute successfully with data from sub2.
        """
        def wired_factory(
            task: str, failure_context: str | None
        ) -> list[TaskNode]:
            sub_a_id = str(uuid.uuid4())
            sub_b_id = str(uuid.uuid4())
            return [
                TaskNode(
                    id=sub_a_id,
                    description="sub_step_ok_1",
                    is_atomic=True,
                    domain=Domain.SYNTHESIS,
                    complexity=1,
                    status=TaskStatus.READY,
                    provides=["sub_data_1"],
                ),
                TaskNode(
                    id=sub_b_id,
                    description="sub_step_ok_2",
                    is_atomic=True,
                    domain=Domain.SYNTHESIS,
                    complexity=1,
                    status=TaskStatus.READY,
                    requires=[sub_a_id],
                    consumes=["sub_data_1"],
                    provides=["data_b"],
                ),
            ]

        mock_exec = MockExecutorForExpansion(fail_on={"hard_task"})
        mock_decomposer = MockDecomposerForExpansion(
            sub_dag_factory=wired_factory
        )

        dag = DAGExecutor(
            executor=mock_exec,
            max_concurrency=10,
            max_expansion_depth=2,
        )
        dag._decomposer = mock_decomposer

        nodes = [
            _make_leaf("a", "easy_a", provides=["data_a"]),
            _make_leaf(
                "b", "hard_task",
                requires=["a"], consumes=["data_a"], provides=["data_b"],
            ),
            _make_leaf(
                "c", "easy_c",
                requires=["b"], consumes=["data_b"],
            ),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        # C should have executed (it depends on the sub-DAG terminal)
        assert any("easy_c" in call for call in mock_exec.call_order)

    async def test_expansion_count_zero_when_no_failures(self) -> None:
        """expansion_count defaults to 0 for a fully successful run."""
        mock_exec = MockExecutorForExpansion()
        dag = DAGExecutor(executor=mock_exec, max_concurrency=10)

        nodes = [
            _make_leaf("a", "ok_1"),
            _make_leaf("b", "ok_2"),
        ]

        result = await dag.execute(nodes)
        assert result.expansion_count == 0
