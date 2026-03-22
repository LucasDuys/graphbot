"""Tests for ConditionalNode type and then/else branch routing in DAGExecutor."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    ConditionalNode,
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)


class MockConditionalExecutor:
    """Mock executor that returns configurable output per task description."""

    def __init__(
        self,
        outputs: dict[str, str] | None = None,
    ) -> None:
        self.call_order: list[str] = []
        self._outputs = outputs or {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        task: str,
        complexity: int = 1,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        async with self._lock:
            self.call_order.append(task)
        # Match output by checking if any key is a substring of the task text
        output = "default output"
        for key, val in self._outputs.items():
            if key in task:
                output = val
                break
        return ExecutionResult(
            root_id=str(uuid.uuid4()),
            output=output,
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=0.0,
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


class TestConditionalNodeType:
    """Tests for ConditionalNode dataclass structure."""

    def test_conditional_node_extends_task_node(self) -> None:
        node = ConditionalNode(
            id="cond_1",
            description="Check if data is valid",
            condition="contains 'valid'",
            then_branch=["then_a", "then_b"],
            else_branch=["else_a"],
        )
        assert isinstance(node, TaskNode)

    def test_conditional_node_fields(self) -> None:
        node = ConditionalNode(
            id="cond_1",
            description="Check result",
            condition="contains 'success'",
            then_branch=["t1", "t2"],
            else_branch=["e1"],
        )
        assert node.condition == "contains 'success'"
        assert node.then_branch == ["t1", "t2"]
        assert node.else_branch == ["e1"]

    def test_conditional_node_default_branches(self) -> None:
        node = ConditionalNode(
            id="cond_1",
            description="Check result",
            condition="contains 'yes'",
        )
        assert node.then_branch == []
        assert node.else_branch == []

    def test_conditional_node_inherits_task_node_fields(self) -> None:
        node = ConditionalNode(
            id="cond_1",
            description="Conditional task",
            condition="test",
            then_branch=["t1"],
            else_branch=["e1"],
            domain=Domain.CODE,
            complexity=3,
            requires=["predecessor"],
            consumes=["pred_output"],
        )
        assert node.domain == Domain.CODE
        assert node.complexity == 3
        assert node.requires == ["predecessor"]
        assert node.consumes == ["pred_output"]


class TestSkippedStatus:
    """Tests for the SKIPPED TaskStatus value."""

    def test_skipped_status_exists(self) -> None:
        assert TaskStatus.SKIPPED == "skipped"

    def test_skipped_status_is_string(self) -> None:
        assert isinstance(TaskStatus.SKIPPED, str)


class TestConditionalTrueRouting:
    """When condition is true, then_branch executes and else_branch is SKIPPED."""

    async def test_true_condition_routes_to_then_branch(self) -> None:
        """Predecessor outputs 'valid', condition matches, then_branch runs."""
        mock = MockConditionalExecutor(
            outputs={"Produce data": "valid data here"}
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        predecessor = _make_leaf(
            "pred", "Produce data", provides=["pred_output"]
        )
        cond = ConditionalNode(
            id="cond_1",
            description="Check validity",
            condition="contains 'valid'",
            then_branch=["then_task"],
            else_branch=["else_task"],
            requires=["pred"],
            consumes=["pred_output"],
            is_atomic=False,
        )
        then_task = _make_leaf(
            "then_task", "Then branch work",
            requires=["cond_1"],
        )
        else_task = _make_leaf(
            "else_task", "Else branch work",
            requires=["cond_1"],
        )

        nodes = [predecessor, cond, then_task, else_task]
        result = await dag.execute(nodes)

        assert result.success is True
        # then_task should have been called
        executed_descs = " ".join(mock.call_order)
        assert "Then branch work" in executed_descs
        # else_task should NOT have been called
        assert "Else branch work" not in executed_descs
        # else_task should be marked SKIPPED
        assert else_task.status == TaskStatus.SKIPPED

    async def test_then_branch_multiple_nodes(self) -> None:
        """Multiple then_branch nodes all execute when condition is true."""
        mock = MockConditionalExecutor(
            outputs={"Source": "yes confirmed"}
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        pred = _make_leaf("pred", "Source", provides=["data"])
        cond = ConditionalNode(
            id="cond_1",
            description="Route check",
            condition="contains 'yes'",
            then_branch=["t1", "t2"],
            else_branch=["e1"],
            requires=["pred"],
            consumes=["data"],
            is_atomic=False,
        )
        t1 = _make_leaf("t1", "Then task 1", requires=["cond_1"])
        t2 = _make_leaf("t2", "Then task 2", requires=["cond_1"])
        e1 = _make_leaf("e1", "Else task 1", requires=["cond_1"])

        result = await dag.execute([pred, cond, t1, t2, e1])

        assert result.success is True
        executed_descs = " ".join(mock.call_order)
        assert "Then task 1" in executed_descs
        assert "Then task 2" in executed_descs
        assert "Else task 1" not in executed_descs
        assert e1.status == TaskStatus.SKIPPED


class TestConditionalFalseRouting:
    """When condition is false, else_branch executes and then_branch is SKIPPED."""

    async def test_false_condition_routes_to_else_branch(self) -> None:
        """Predecessor outputs 'error', condition does not match, else_branch runs."""
        mock = MockConditionalExecutor(
            outputs={"Produce data": "error in processing"}
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        predecessor = _make_leaf(
            "pred", "Produce data", provides=["pred_output"]
        )
        cond = ConditionalNode(
            id="cond_1",
            description="Check validity",
            condition="contains 'valid'",
            then_branch=["then_task"],
            else_branch=["else_task"],
            requires=["pred"],
            consumes=["pred_output"],
            is_atomic=False,
        )
        then_task = _make_leaf(
            "then_task", "Then branch work",
            requires=["cond_1"],
        )
        else_task = _make_leaf(
            "else_task", "Else branch work",
            requires=["cond_1"],
        )

        nodes = [predecessor, cond, then_task, else_task]
        result = await dag.execute(nodes)

        assert result.success is True
        executed_descs = " ".join(mock.call_order)
        # else_task should have been called
        assert "Else branch work" in executed_descs
        # then_task should NOT have been called
        assert "Then branch work" not in executed_descs
        # then_task should be marked SKIPPED
        assert then_task.status == TaskStatus.SKIPPED

    async def test_else_branch_multiple_nodes(self) -> None:
        """Multiple else_branch nodes all execute when condition is false."""
        mock = MockConditionalExecutor(
            outputs={"Source": "nope nothing"}
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        pred = _make_leaf("pred", "Source", provides=["data"])
        cond = ConditionalNode(
            id="cond_1",
            description="Route check",
            condition="contains 'yes'",
            then_branch=["t1"],
            else_branch=["e1", "e2"],
            requires=["pred"],
            consumes=["data"],
            is_atomic=False,
        )
        t1 = _make_leaf("t1", "Then task 1", requires=["cond_1"])
        e1 = _make_leaf("e1", "Else task 1", requires=["cond_1"])
        e2 = _make_leaf("e2", "Else task 2", requires=["cond_1"])

        result = await dag.execute([pred, cond, t1, e1, e2])

        assert result.success is True
        executed_descs = " ".join(mock.call_order)
        assert "Then task 1" not in executed_descs
        assert "Else task 1" in executed_descs
        assert "Else task 2" in executed_descs
        assert t1.status == TaskStatus.SKIPPED


class TestPredecessorOutputUsed:
    """The condition evaluation uses the predecessor node's output data."""

    async def test_predecessor_output_drives_condition(self) -> None:
        """Condition evaluated against predecessor output, not static text."""
        # First run: predecessor says "valid" -> then_branch
        mock_true = MockConditionalExecutor(
            outputs={"Generate": "valid result"}
        )
        dag_true = DAGExecutor(executor=mock_true, max_concurrency=10)

        pred = _make_leaf("pred", "Generate", provides=["result"])
        cond = ConditionalNode(
            id="cond_1",
            description="Evaluate",
            condition="contains 'valid'",
            then_branch=["then_node"],
            else_branch=["else_node"],
            requires=["pred"],
            consumes=["result"],
            is_atomic=False,
        )
        then_node = _make_leaf("then_node", "Then work", requires=["cond_1"])
        else_node = _make_leaf("else_node", "Else work", requires=["cond_1"])

        await dag_true.execute([pred, cond, then_node, else_node])
        assert "Then work" in " ".join(mock_true.call_order)

        # Second run: predecessor says "invalid" -> else_branch
        mock_false = MockConditionalExecutor(
            outputs={"Generate": "total failure"}
        )
        dag_false = DAGExecutor(executor=mock_false, max_concurrency=10)

        pred2 = _make_leaf("pred", "Generate", provides=["result"])
        cond2 = ConditionalNode(
            id="cond_1",
            description="Evaluate",
            condition="contains 'valid'",
            then_branch=["then_node"],
            else_branch=["else_node"],
            requires=["pred"],
            consumes=["result"],
            is_atomic=False,
        )
        then_node2 = _make_leaf("then_node", "Then work", requires=["cond_1"])
        else_node2 = _make_leaf("else_node", "Else work", requires=["cond_1"])

        await dag_false.execute([pred2, cond2, then_node2, else_node2])
        assert "Else work" in " ".join(mock_false.call_order)

    async def test_skipped_nodes_not_in_results(self) -> None:
        """Skipped branch nodes should not contribute to execution results output."""
        mock = MockConditionalExecutor(
            outputs={"Source": "valid data"}
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        pred = _make_leaf("pred", "Source", provides=["data"])
        cond = ConditionalNode(
            id="cond_1",
            description="Check",
            condition="contains 'valid'",
            then_branch=["then_t"],
            else_branch=["else_t"],
            requires=["pred"],
            consumes=["data"],
            is_atomic=False,
        )
        then_t = _make_leaf("then_t", "Active work", requires=["cond_1"])
        else_t = _make_leaf("else_t", "Skipped work", requires=["cond_1"])

        result = await dag.execute([pred, cond, then_t, else_t])

        assert result.success is True
        # The skipped node's output should not appear in the final result
        assert "Skipped work" not in result.output
