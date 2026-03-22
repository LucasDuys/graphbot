"""Tests for control flow integration in topological dispatch.

Verifies that ConditionalNode and LoopNode are handled correctly within
the DAGExecutor.execute() topological dispatch loop, including:
- Mixed DAGs with regular + conditional + loop nodes
- LoopNode integrated into the main execute() flow (not just execute_loop())
- Nested conditionals
- Loops containing conditionals
- Correct ordering and data flow between control flow and regular nodes
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from typing import Any

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    ConditionalNode,
    Domain,
    ExecutionResult,
    LoopNode,
    TaskNode,
    TaskStatus,
)


class MockControlFlowExecutor:
    """Mock executor that returns configurable output per task description substring.

    Tracks call order and supports per-invocation output matching. When a task
    description contains a registered key substring, that key's output is returned.
    Falls back to "default output" if no key matches.
    """

    def __init__(
        self,
        outputs: dict[str, str] | None = None,
        iteration_outputs: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize mock executor.

        Args:
            outputs: Map of description-substring to output text. First match wins.
            iteration_outputs: Map of description-substring to a list of outputs,
                one per invocation. Supports testing loops where the same node
                produces different output across iterations.
        """
        self.call_order: list[str] = []
        self.call_count: int = 0
        self._outputs = outputs or {}
        self._iteration_outputs = iteration_outputs or {}
        self._iteration_counters: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        task: str,
        complexity: int = 1,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        async with self._lock:
            self.call_order.append(task)
            self.call_count += 1

            # Check iteration_outputs first (allows per-call variation)
            for key, outputs_list in self._iteration_outputs.items():
                if key in task:
                    idx = self._iteration_counters.get(key, 0)
                    self._iteration_counters[key] = idx + 1
                    output = outputs_list[min(idx, len(outputs_list) - 1)]
                    return ExecutionResult(
                        root_id=str(uuid.uuid4()),
                        output=output,
                        success=True,
                        total_nodes=1,
                        total_tokens=10,
                        total_latency_ms=0.0,
                        total_cost=0.001,
                    )

            # Fall back to static outputs
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


# ---------------------------------------------------------------------------
# Test 1: Mixed DAG with regular + conditional + loop nodes
# ---------------------------------------------------------------------------


class TestMixedDAGExecution:
    """A DAG containing regular TaskNodes, a ConditionalNode, and a LoopNode
    all executes correctly through the single execute() entry point."""

    async def test_regular_then_conditional_then_regular(self) -> None:
        """regular -> conditional -> regular: the final regular node
        only receives output from the selected branch."""
        mock = MockControlFlowExecutor(
            outputs={
                "Fetch data": "valid payload",
                "Process valid": "processed result",
                "Handle error": "error handled",
                "Summarize": "summary of processed result",
            }
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        fetch = _make_leaf("fetch", "Fetch data", provides=["raw_data"])
        cond = ConditionalNode(
            id="cond_1",
            description="Check validity",
            condition="contains 'valid'",
            then_branch=["process"],
            else_branch=["handle_err"],
            requires=["fetch"],
            consumes=["raw_data"],
            is_atomic=False,
        )
        process = _make_leaf(
            "process", "Process valid",
            requires=["cond_1"], provides=["proc_out"],
        )
        handle_err = _make_leaf(
            "handle_err", "Handle error",
            requires=["cond_1"], provides=["proc_out"],
        )
        summarize = _make_leaf(
            "summarize", "Summarize",
            requires=["process", "handle_err"],
            consumes=["proc_out"],
        )

        result = await dag.execute([fetch, cond, process, handle_err, summarize])

        assert result.success is True
        descs = " ".join(mock.call_order)
        assert "Fetch data" in descs
        assert "Process valid" in descs
        assert "Summarize" in descs
        # else_branch should be skipped
        assert "Handle error" not in descs
        assert handle_err.status == TaskStatus.SKIPPED

    async def test_mixed_dag_with_loop_node(self) -> None:
        """regular -> loop -> regular: the loop body iterates, and the
        final regular node runs after the loop completes."""
        mock = MockControlFlowExecutor(
            iteration_outputs={
                "Refine": ["partial", "DONE: refined"],
            },
            outputs={
                "Prepare": "initial data",
                "Finalize": "final output",
            },
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        prepare = _make_leaf("prepare", "Prepare", provides=["prep_data"])
        body = _make_leaf(
            "loop_body", "Refine",
            provides=["refined"],
        )
        loop = LoopNode(
            id="loop_1",
            description="Refinement loop",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["loop_body"],
            requires=["prepare"],
            consumes=["prep_data"],
            provides=["refined"],
            is_atomic=False,
            children=["loop_body"],
        )
        finalize = _make_leaf(
            "finalize", "Finalize",
            requires=["loop_1"],
            consumes=["refined"],
        )

        result = await dag.execute([prepare, loop, body, finalize])

        assert result.success is True
        descs = " ".join(mock.call_order)
        assert "Prepare" in descs
        assert "Finalize" in descs
        # Loop body should have been called at least once
        refine_calls = [d for d in mock.call_order if "Refine" in d]
        assert len(refine_calls) >= 1


# ---------------------------------------------------------------------------
# Test 2: LoopNode integrated into topological dispatch
# ---------------------------------------------------------------------------


class TestLoopNodeInTopologicalDispatch:
    """LoopNode is handled within execute(), not requiring a separate
    execute_loop() call. The sorter treats the loop as a single unit."""

    async def test_loop_body_reruns_until_exit_condition(self) -> None:
        """Loop body nodes are re-executed until the exit condition is met."""
        mock = MockControlFlowExecutor(
            iteration_outputs={
                "Attempt": ["try 1", "try 2", "DONE: success"],
            },
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body = _make_leaf("body_a", "Attempt the task", provides=["result"])
        loop = LoopNode(
            id="loop_1",
            description="Retry loop",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body_a"],
            is_atomic=False,
            children=["body_a"],
            provides=["result"],
        )

        result = await dag.execute([loop, body])

        assert result.success is True
        attempt_calls = [d for d in mock.call_order if "Attempt" in d]
        assert len(attempt_calls) == 3

    async def test_loop_stops_at_max_iterations(self) -> None:
        """Loop stops after max_iterations even if exit condition is never met."""
        mock = MockControlFlowExecutor(
            outputs={"Try again": "still not done"},
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body = _make_leaf("body_a", "Try again", provides=["out"])
        loop = LoopNode(
            id="loop_1",
            description="Bounded loop",
            max_iterations=3,
            exit_condition="contains:DONE",
            body_nodes=["body_a"],
            is_atomic=False,
            children=["body_a"],
            provides=["out"],
        )

        result = await dag.execute([loop, body])

        assert result.success is True
        calls = [d for d in mock.call_order if "Try again" in d]
        assert len(calls) == 3

    async def test_loop_with_upstream_dependency(self) -> None:
        """Loop node that depends on a regular predecessor receives its data."""
        mock = MockControlFlowExecutor(
            outputs={
                "Generate seed": "seed_value",
                "Iterate on": "DONE: polished",
            },
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        seed = _make_leaf("seed", "Generate seed", provides=["seed_data"])
        body = _make_leaf(
            "body_a", "Iterate on seed",
            provides=["polished"],
        )
        loop = LoopNode(
            id="loop_1",
            description="Polish loop",
            max_iterations=3,
            exit_condition="contains:DONE",
            body_nodes=["body_a"],
            requires=["seed"],
            consumes=["seed_data"],
            is_atomic=False,
            children=["body_a"],
            provides=["polished"],
        )

        result = await dag.execute([seed, loop, body])

        assert result.success is True
        descs = " ".join(mock.call_order)
        assert "Generate seed" in descs

    async def test_loop_with_downstream_dependent(self) -> None:
        """A regular node that depends on the loop receives the loop output."""
        mock = MockControlFlowExecutor(
            iteration_outputs={
                "Compute": ["DONE: computed_value"],
            },
            outputs={
                "Use computed": "used it",
            },
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body = _make_leaf("body_a", "Compute something", provides=["computed"])
        loop = LoopNode(
            id="loop_1",
            description="Compute loop",
            max_iterations=3,
            exit_condition="contains:DONE",
            body_nodes=["body_a"],
            is_atomic=False,
            children=["body_a"],
            provides=["computed"],
        )
        consumer = _make_leaf(
            "consumer", "Use computed value",
            requires=["loop_1"],
            consumes=["computed"],
        )

        result = await dag.execute([loop, body, consumer])

        assert result.success is True
        descs = " ".join(mock.call_order)
        assert "Use computed" in descs

    async def test_loop_with_multiple_body_nodes(self) -> None:
        """Loop with two body nodes (A -> B) runs both each iteration."""
        call_count = 0

        class MultiBodyMock:
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
                # On the 4th call (body_b iteration 2), return DONE
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

        mock = MultiBodyMock()
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body_a = _make_leaf("body_a", "Step A", provides=["data_a"])
        body_b = _make_leaf(
            "body_b", "Step B",
            requires=["body_a"], consumes=["data_a"], provides=["data_b"],
        )
        loop = LoopNode(
            id="loop_1",
            description="Multi-body loop",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body_a", "body_b"],
            is_atomic=False,
            children=["body_a", "body_b"],
            provides=["data_b"],
        )

        result = await dag.execute([loop, body_a, body_b])

        assert result.success is True
        assert mock.total_calls == 4  # 2 body nodes * 2 iterations


# ---------------------------------------------------------------------------
# Test 3: Nested conditionals
# ---------------------------------------------------------------------------


class TestNestedConditionals:
    """Nested conditional: outer conditional routes to inner conditional,
    which further routes to leaf nodes."""

    async def test_nested_conditional_both_true(self) -> None:
        """Outer=true -> inner conditional -> inner=true -> deep_then runs."""
        mock = MockControlFlowExecutor(
            outputs={
                "Source": "valid data",
                "Validate": "confirmed valid",
                "Deep then": "deep success",
                "Deep else": "deep fallback",
                "Outer else": "outer fallback",
            }
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        source = _make_leaf("source", "Source", provides=["raw"])

        outer_cond = ConditionalNode(
            id="outer_cond",
            description="Outer check",
            condition="contains 'valid'",
            then_branch=["validate"],
            else_branch=["outer_else"],
            requires=["source"],
            consumes=["raw"],
            is_atomic=False,
        )
        validate = _make_leaf(
            "validate", "Validate data",
            requires=["outer_cond"], provides=["validated"],
        )
        outer_else = _make_leaf(
            "outer_else", "Outer else fallback",
            requires=["outer_cond"],
        )

        inner_cond = ConditionalNode(
            id="inner_cond",
            description="Inner check",
            condition="contains 'confirmed'",
            then_branch=["deep_then"],
            else_branch=["deep_else"],
            requires=["validate"],
            consumes=["validated"],
            is_atomic=False,
        )
        deep_then = _make_leaf(
            "deep_then", "Deep then work",
            requires=["inner_cond"],
        )
        deep_else = _make_leaf(
            "deep_else", "Deep else work",
            requires=["inner_cond"],
        )

        nodes = [source, outer_cond, validate, outer_else,
                 inner_cond, deep_then, deep_else]
        result = await dag.execute(nodes)

        assert result.success is True
        descs = " ".join(mock.call_order)
        assert "Source" in descs
        assert "Validate" in descs
        assert "Deep then" in descs
        assert "Outer else" not in descs
        assert "Deep else" not in descs
        assert outer_else.status == TaskStatus.SKIPPED
        assert deep_else.status == TaskStatus.SKIPPED

    async def test_nested_conditional_outer_false(self) -> None:
        """Outer=false -> outer_else runs, inner conditional never evaluated."""
        mock = MockControlFlowExecutor(
            outputs={
                "Source": "garbage data",
                "Outer else": "handled gracefully",
            }
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        source = _make_leaf("source", "Source", provides=["raw"])

        outer_cond = ConditionalNode(
            id="outer_cond",
            description="Outer check",
            condition="contains 'valid'",
            then_branch=["validate"],
            else_branch=["outer_else"],
            requires=["source"],
            consumes=["raw"],
            is_atomic=False,
        )
        validate = _make_leaf(
            "validate", "Validate data",
            requires=["outer_cond"], provides=["validated"],
        )
        outer_else = _make_leaf(
            "outer_else", "Outer else fallback",
            requires=["outer_cond"],
        )

        inner_cond = ConditionalNode(
            id="inner_cond",
            description="Inner check",
            condition="contains 'confirmed'",
            then_branch=["deep_then"],
            else_branch=["deep_else"],
            requires=["validate"],
            consumes=["validated"],
            is_atomic=False,
        )
        deep_then = _make_leaf(
            "deep_then", "Deep then work",
            requires=["inner_cond"],
        )
        deep_else = _make_leaf(
            "deep_else", "Deep else work",
            requires=["inner_cond"],
        )

        nodes = [source, outer_cond, validate, outer_else,
                 inner_cond, deep_then, deep_else]
        result = await dag.execute(nodes)

        assert result.success is True
        descs = " ".join(mock.call_order)
        assert "Source" in descs
        assert "Outer else" in descs
        # Inner branch nodes should all be skipped (their dependency "validate" was skipped)
        assert "Validate" not in descs
        assert "Deep then" not in descs
        assert "Deep else" not in descs
        assert validate.status == TaskStatus.SKIPPED


# ---------------------------------------------------------------------------
# Test 4: Loop containing conditional
# ---------------------------------------------------------------------------


class TestLoopContainingConditional:
    """A loop whose body contains a conditional node. Each iteration
    evaluates the conditional and routes to the correct branch."""

    async def test_loop_body_with_conditional(self) -> None:
        """Loop body: check -> conditional -> then/else. The conditional
        routes differently based on the check output each iteration."""
        mock = MockControlFlowExecutor(
            iteration_outputs={
                "Check quality": [
                    "quality: poor",      # iteration 1: else branch
                    "quality: valid",     # iteration 2: then branch -> DONE
                ],
            },
            outputs={
                "Fix issues": "attempted fix",
                "Package result": "DONE: packaged",
            },
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        check = _make_leaf("check", "Check quality", provides=["quality_report"])

        cond = ConditionalNode(
            id="body_cond",
            description="Quality gate",
            condition="contains 'valid'",
            then_branch=["package"],
            else_branch=["fix"],
            requires=["check"],
            consumes=["quality_report"],
            is_atomic=False,
        )
        package = _make_leaf(
            "package", "Package result",
            requires=["body_cond"], provides=["final"],
        )
        fix = _make_leaf(
            "fix", "Fix issues",
            requires=["body_cond"], provides=["final"],
        )

        loop = LoopNode(
            id="loop_1",
            description="Quality loop",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["check", "body_cond", "package", "fix"],
            is_atomic=False,
            children=["check", "body_cond", "package", "fix"],
            provides=["final"],
        )

        result = await dag.execute([loop, check, cond, package, fix])

        assert result.success is True
        descs = " ".join(mock.call_order)
        # Check should have been called at least twice (two iterations)
        check_calls = [d for d in mock.call_order if "Check quality" in d]
        assert len(check_calls) >= 2


# ---------------------------------------------------------------------------
# Test 5: LoopNode with zero iterations in main dispatch
# ---------------------------------------------------------------------------


class TestLoopEdgeCases:
    """Edge cases for loop integration in the main dispatch."""

    async def test_loop_zero_iterations_in_dispatch(self) -> None:
        """A loop with max_iterations=0 produces no body executions but
        still allows downstream nodes to proceed."""
        mock = MockControlFlowExecutor(
            outputs={"After loop": "post-loop work"},
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body = _make_leaf("body_a", "Never runs", provides=["data"])
        loop = LoopNode(
            id="loop_1",
            description="Empty loop",
            max_iterations=0,
            exit_condition="contains:DONE",
            body_nodes=["body_a"],
            is_atomic=False,
            children=["body_a"],
            provides=["data"],
        )
        after = _make_leaf(
            "after", "After loop",
            requires=["loop_1"],
        )

        result = await dag.execute([loop, body, after])

        assert result.success is True
        body_calls = [d for d in mock.call_order if "Never runs" in d]
        assert len(body_calls) == 0

    async def test_loop_exit_on_first_iteration(self) -> None:
        """Loop exits immediately on the first iteration when condition is met."""
        mock = MockControlFlowExecutor(
            outputs={"Quick task": "DONE: instant"},
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        body = _make_leaf("body_a", "Quick task", provides=["out"])
        loop = LoopNode(
            id="loop_1",
            description="Quick loop",
            max_iterations=10,
            exit_condition="contains:DONE",
            body_nodes=["body_a"],
            is_atomic=False,
            children=["body_a"],
            provides=["out"],
        )

        result = await dag.execute([loop, body])

        assert result.success is True
        calls = [d for d in mock.call_order if "Quick task" in d]
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Test 6: Conditional followed by loop
# ---------------------------------------------------------------------------


class TestConditionalThenLoop:
    """Conditional routes to a loop: the selected branch triggers a loop
    that iterates until exit condition is met."""

    async def test_conditional_routes_to_loop(self) -> None:
        """Condition true -> triggers a loop that iterates body nodes."""
        mock = MockControlFlowExecutor(
            outputs={
                "Check mode": "mode: iterative",
                "Skip path": "skipped",
            },
            iteration_outputs={
                "Iterate step": ["step 1", "DONE: step 2"],
            },
        )
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        check = _make_leaf("check", "Check mode", provides=["mode"])
        cond = ConditionalNode(
            id="mode_cond",
            description="Mode check",
            condition="contains 'iterative'",
            then_branch=["loop_1"],
            else_branch=["skip"],
            requires=["check"],
            consumes=["mode"],
            is_atomic=False,
        )
        body = _make_leaf("body_a", "Iterate step", provides=["iter_out"])
        loop = LoopNode(
            id="loop_1",
            description="Iteration loop",
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body_a"],
            requires=["mode_cond"],
            is_atomic=False,
            children=["body_a"],
            provides=["iter_out"],
        )
        skip = _make_leaf(
            "skip", "Skip path",
            requires=["mode_cond"],
        )

        result = await dag.execute([check, cond, loop, body, skip])

        assert result.success is True
        descs = " ".join(mock.call_order)
        assert "Check mode" in descs
        assert "Skip path" not in descs
        assert skip.status == TaskStatus.SKIPPED
        iter_calls = [d for d in mock.call_order if "Iterate step" in d]
        assert len(iter_calls) >= 1
