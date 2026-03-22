"""Tests for LoopNode type and retry-with-context iteration in DAG executor."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    Domain,
    ExecutionResult,
    LoopNode,
    TaskNode,
    TaskStatus,
)


class MockLoopExecutor:
    """Mock executor that returns configurable results per call count.

    Tracks call history and supports per-invocation output control so tests
    can simulate exit-condition satisfaction on a specific iteration.
    """

    def __init__(
        self,
        results_sequence: list[tuple[bool, str]] | None = None,
    ) -> None:
        """Initialise mock executor.

        Args:
            results_sequence: Optional ordered list of (success, output) pairs.
                Each call to execute() pops the next pair. If exhausted, the
                last pair is reused.
        """
        self.call_count: int = 0
        self.call_texts: list[str] = []
        self._results_sequence = results_sequence or [(True, "default output")]
        self._lock = asyncio.Lock()

    async def execute(
        self,
        task: str,
        complexity: int = 1,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        async with self._lock:
            idx = min(self.call_count, len(self._results_sequence) - 1)
            success, output = self._results_sequence[idx]
            self.call_count += 1
            self.call_texts.append(task)

        return ExecutionResult(
            root_id=str(uuid.uuid4()),
            output=output,
            success=success,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=1.0,
            total_cost=0.001,
            errors=() if success else (f"Failed: {task}",),
        )


def _make_body_leaf(
    node_id: str,
    description: str,
    requires: list[str] | None = None,
    provides: list[str] | None = None,
    consumes: list[str] | None = None,
) -> TaskNode:
    """Create an atomic leaf TaskNode for use inside a loop body."""
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


class TestLoopNodeType:
    """Tests for the LoopNode dataclass structure and defaults."""

    def test_loop_node_extends_task_node(self) -> None:
        """LoopNode is a subclass of TaskNode."""
        loop = LoopNode(id="loop1", description="Retry loop")
        assert isinstance(loop, TaskNode)

    def test_default_max_iterations(self) -> None:
        """Default max_iterations is 3."""
        loop = LoopNode(id="loop1", description="Retry loop")
        assert loop.max_iterations == 3

    def test_custom_max_iterations(self) -> None:
        """max_iterations can be set to a custom value."""
        loop = LoopNode(id="loop1", description="Retry loop", max_iterations=5)
        assert loop.max_iterations == 5

    def test_exit_condition_field(self) -> None:
        """exit_condition stores a string condition expression."""
        loop = LoopNode(
            id="loop1",
            description="Retry loop",
            exit_condition="contains:DONE",
        )
        assert loop.exit_condition == "contains:DONE"

    def test_default_exit_condition(self) -> None:
        """Default exit_condition is an empty string."""
        loop = LoopNode(id="loop1", description="Retry loop")
        assert loop.exit_condition == ""

    def test_body_nodes_field(self) -> None:
        """body_nodes stores a list of node IDs."""
        loop = LoopNode(
            id="loop1",
            description="Retry loop",
            body_nodes=["step_a", "step_b"],
        )
        assert loop.body_nodes == ["step_a", "step_b"]

    def test_default_body_nodes(self) -> None:
        """Default body_nodes is an empty list."""
        loop = LoopNode(id="loop1", description="Retry loop")
        assert loop.body_nodes == []

    def test_inherits_task_node_fields(self) -> None:
        """LoopNode retains all TaskNode fields."""
        loop = LoopNode(
            id="loop1",
            description="Retry loop",
            domain=Domain.CODE,
            complexity=3,
            provides=["refined_output"],
        )
        assert loop.domain == Domain.CODE
        assert loop.complexity == 3
        assert loop.provides == ["refined_output"]


class TestLoopExecution:
    """Tests for loop execution within the DAG executor."""

    async def test_loop_succeeds_on_iteration_2(self) -> None:
        """Loop body runs twice: first iteration output does not meet exit
        condition, second iteration does. Total body executions = 2.
        """
        mock = MockLoopExecutor(
            results_sequence=[
                (True, "partial result"),       # iteration 1: no match
                (True, "DONE: final result"),   # iteration 2: matches
            ],
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_leaf = _make_body_leaf(
            "body1", "Refine the output", provides=["refined"],
        )
        loop = LoopNode(
            id="loop1",
            description="Iterate until done",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body1"],
            is_atomic=False,
            children=["body1"],
            provides=["refined"],
        )

        result = await dag.execute_loop(loop, [body_leaf])

        assert result.success is True
        assert "DONE" in result.output
        # Body ran exactly 2 times
        assert mock.call_count == 2

    async def test_loop_exits_at_max_iterations(self) -> None:
        """When exit condition is never met, loop stops at max_iterations."""
        mock = MockLoopExecutor(
            results_sequence=[
                (True, "still not done"),
            ],
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_leaf = _make_body_leaf("body1", "Try again")
        loop = LoopNode(
            id="loop1",
            description="Iterate until done",
            max_iterations=3,
            exit_condition="contains:DONE",
            body_nodes=["body1"],
            is_atomic=False,
            children=["body1"],
        )

        result = await dag.execute_loop(loop, [body_leaf])

        # Should have run exactly max_iterations times
        assert mock.call_count == 3
        # Still returns a result (last iteration output)
        assert result.output == "still not done"
        # Success is True because the body itself did not fail
        assert result.success is True

    async def test_exit_condition_checked_each_iteration(self) -> None:
        """Exit condition is evaluated after every iteration, not just at end."""
        mock = MockLoopExecutor(
            results_sequence=[
                (True, "attempt 1"),
                (True, "attempt 2"),
                (True, "DONE on 3"),
                (True, "should not reach"),
            ],
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_leaf = _make_body_leaf("body1", "Attempt task")
        loop = LoopNode(
            id="loop1",
            description="Iterate until done",
            max_iterations=10,
            exit_condition="contains:DONE",
            body_nodes=["body1"],
            is_atomic=False,
            children=["body1"],
        )

        result = await dag.execute_loop(loop, [body_leaf])

        # Should stop at iteration 3 (exit condition met)
        assert mock.call_count == 3
        assert "DONE" in result.output

    async def test_previous_iteration_context_in_prompt(self) -> None:
        """Each iteration after the first includes previous iteration output
        in the task description, enabling retry-with-context.
        """
        mock = MockLoopExecutor(
            results_sequence=[
                (True, "first attempt output"),
                (True, "DONE: improved"),
            ],
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_leaf = _make_body_leaf("body1", "Refine the answer")
        loop = LoopNode(
            id="loop1",
            description="Iterate until done",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body1"],
            is_atomic=False,
            children=["body1"],
        )

        await dag.execute_loop(loop, [body_leaf])

        # First call has no previous context
        assert "previous_iteration" not in mock.call_texts[0].lower()
        # Second call includes previous iteration output
        assert "first attempt output" in mock.call_texts[1]
        assert "previous_iteration" in mock.call_texts[1].lower()

    async def test_loop_body_failure_stops_iteration(self) -> None:
        """If a body node fails, the loop stops and returns a failed result."""
        mock = MockLoopExecutor(
            results_sequence=[
                (True, "ok first time"),
                (False, ""),   # body fails on iteration 2
            ],
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_leaf = _make_body_leaf("body1", "Do something risky")
        loop = LoopNode(
            id="loop1",
            description="Iterate until done",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body1"],
            is_atomic=False,
            children=["body1"],
        )

        result = await dag.execute_loop(loop, [body_leaf])

        assert result.success is False
        assert mock.call_count == 2

    async def test_loop_with_multiple_body_nodes(self) -> None:
        """Loop with two body nodes runs both each iteration."""
        call_count = 0

        class MultiNodeMock:
            def __init__(self) -> None:
                self.total_calls: int = 0
                self.call_texts: list[str] = []

            async def execute(
                self,
                task: str,
                complexity: int = 1,
                provides_keys: list[str] | None = None,
            ) -> ExecutionResult:
                self.total_calls += 1
                self.call_texts.append(task)
                # Second body node in iteration 2 returns DONE
                if self.total_calls == 4:
                    output = "DONE: complete"
                else:
                    output = f"output_{self.total_calls}"
                return ExecutionResult(
                    root_id=str(uuid.uuid4()),
                    output=output,
                    success=True,
                    total_nodes=1,
                    total_tokens=10,
                    total_latency_ms=1.0,
                    total_cost=0.001,
                )

        mock = MultiNodeMock()
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_a = _make_body_leaf("body_a", "Step A", provides=["data_a"])
        body_b = _make_body_leaf(
            "body_b", "Step B",
            requires=["body_a"], consumes=["data_a"], provides=["data_b"],
        )
        loop = LoopNode(
            id="loop1",
            description="Multi-step loop",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body_a", "body_b"],
            is_atomic=False,
            children=["body_a", "body_b"],
        )

        result = await dag.execute_loop(loop, [body_a, body_b])

        assert result.success is True
        assert "DONE" in result.output
        # 2 body nodes * 2 iterations = 4 calls
        assert mock.total_calls == 4

    async def test_loop_with_zero_max_iterations(self) -> None:
        """A loop with max_iterations=0 returns immediately with empty result."""
        mock = MockLoopExecutor()
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_leaf = _make_body_leaf("body1", "Never runs")
        loop = LoopNode(
            id="loop1",
            description="Zero iteration loop",
            max_iterations=0,
            exit_condition="contains:DONE",
            body_nodes=["body1"],
            is_atomic=False,
            children=["body1"],
        )

        result = await dag.execute_loop(loop, [body_leaf])

        assert result.success is True
        assert mock.call_count == 0
