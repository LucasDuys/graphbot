"""Dynamic execution integration tests for DAGExecutor.

Validates expandable leaf expansion, failure re-decomposition, conditional
routing, loop retry, multi-wave event emission, and expansion depth limits
using mocked providers and decomposers.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    ConditionalNode,
    Domain,
    ExecutionResult,
    LoopNode,
    TaskNode,
    TaskStatus,
)
from core_gb.verification import VerificationConfig
from core_gb.wave_event import WaveCompleteEvent


# ---------------------------------------------------------------------------
# Mock executor
# ---------------------------------------------------------------------------


class MockSimpleExecutor:
    """Executor that returns configurable results keyed by task description substrings.

    If a task description matches a key in ``response_map``, that response is
    returned. Otherwise, a default success result is produced. Tasks whose
    descriptions match an entry in ``fail_on`` return a failed result.
    """

    def __init__(
        self,
        response_map: dict[str, str] | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self.call_order: list[str] = []
        self._response_map: dict[str, str] = response_map or {}
        self._fail_on: set[str] = fail_on or set()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def execute(
        self,
        task: str,
        complexity: int = 1,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        async with self._lock:
            self.call_order.append(task)

        for fail_key in self._fail_on:
            if fail_key in task:
                return ExecutionResult(
                    root_id=str(uuid.uuid4()),
                    output="",
                    success=False,
                    total_nodes=1,
                    total_tokens=5,
                    total_latency_ms=1.0,
                    total_cost=0.0,
                    errors=(f"Failed: {fail_key}",),
                )

        output = f"Result: {task}"
        for key, value in self._response_map.items():
            if key in task:
                output = value
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


# ---------------------------------------------------------------------------
# Mock decomposer
# ---------------------------------------------------------------------------


class MockDecomposer:
    """Decomposer that returns a pre-configured sub-DAG on each call.

    Accepts an ordered list of responses. Each call to ``decompose`` pops
    the next response from the queue. If the queue is exhausted, the last
    response is reused.
    """

    def __init__(self, responses: list[list[TaskNode]]) -> None:
        self._responses: list[list[TaskNode]] = list(responses)
        self._call_count: int = 0
        self.call_log: list[str] = []

    async def decompose(
        self,
        task: str,
        context: Any = None,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> list[TaskNode]:
        self.call_log.append(task)
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leaf(
    node_id: str,
    description: str,
    requires: list[str] | None = None,
    provides: list[str] | None = None,
    consumes: list[str] | None = None,
    expandable: bool = False,
) -> TaskNode:
    """Create an atomic leaf TaskNode with sensible defaults."""
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


def _make_dag_executor(
    executor: MockSimpleExecutor,
    decomposer: MockDecomposer | None = None,
    on_wave_complete: list[Any] | None = None,
    max_expansion_depth: int = 2,
) -> DAGExecutor:
    """Build a DAGExecutor with verification disabled for test clarity."""
    config = VerificationConfig(
        layer1_enabled=False,
        layer2_threshold=999,
        layer3_threshold=999,
    )
    dag = DAGExecutor(
        executor=executor,
        max_concurrency=10,
        verification_config=config,
        on_wave_complete=on_wave_complete,
        max_expansion_depth=max_expansion_depth,
    )
    if decomposer is not None:
        dag._decomposer = decomposer
    return dag


# ---------------------------------------------------------------------------
# 1. Expandable leaf that lazy-expands and succeeds
# ---------------------------------------------------------------------------


class TestExpandableLeafLazyExpands:
    """An expandable leaf is replaced by the decomposer's sub-DAG before
    execution. The sub-DAG leaves execute and produce a successful result."""

    async def test_expandable_leaf_expands_and_succeeds(self) -> None:
        # The expandable leaf "Summarize report" will be expanded into
        # two sub-leaves: "Extract key points" and "Draft summary".
        sub_dag: list[TaskNode] = [
            _leaf("sub_a", "Extract key points", provides=["key_points"]),
            _leaf(
                "sub_b",
                "Draft summary",
                requires=["sub_a"],
                consumes=["key_points"],
                provides=["summary"],
            ),
        ]
        decomposer = MockDecomposer([sub_dag])

        response_map: dict[str, str] = {
            "Extract key points": "Revenue up 20%, costs stable",
            "Draft summary": "The report shows strong growth with stable costs.",
        }
        executor = MockSimpleExecutor(response_map=response_map)

        nodes: list[TaskNode] = [
            _leaf("pre", "Fetch report", provides=["report_data"]),
            _leaf(
                "expand_me",
                "Summarize report",
                requires=["pre"],
                consumes=["report_data"],
                provides=["summary"],
                expandable=True,
            ),
            _leaf(
                "final",
                "Format output",
                requires=["expand_me"],
                consumes=["summary"],
            ),
        ]

        dag = _make_dag_executor(executor, decomposer=decomposer)
        result = await dag.execute(nodes)

        assert result.success is True
        # Decomposer should have been called once (for the expandable node)
        assert len(decomposer.call_log) == 1
        assert "Summarize report" in decomposer.call_log[0]
        # The sub-leaves plus the non-expandable nodes should have executed
        assert len(executor.call_order) >= 3
        # The final output should contain content from all executed nodes
        assert result.output != ""


# ---------------------------------------------------------------------------
# 2. Leaf failure triggers re-decomposition, sub-leaves succeed
# ---------------------------------------------------------------------------


class TestLeafFailureRedecomposition:
    """When a leaf node fails, the DAGExecutor attempts re-decomposition
    via expand_node. The sub-DAG produced by the decomposer succeeds,
    replacing the failed result."""

    async def test_failure_triggers_redecomposition_and_succeeds(self) -> None:
        # The "Translate document" leaf will fail on first execution.
        # The decomposer produces a sub-DAG that splits the task into
        # smaller pieces that succeed.
        sub_dag: list[TaskNode] = [
            _leaf("redecomp_a", "Translate paragraph 1", provides=["para_1"]),
            _leaf("redecomp_b", "Translate paragraph 2", provides=["para_2"]),
        ]
        decomposer = MockDecomposer([sub_dag])

        executor = MockSimpleExecutor(
            response_map={
                "Translate paragraph 1": "Translated text for paragraph 1.",
                "Translate paragraph 2": "Translated text for paragraph 2.",
            },
            fail_on={"Translate document"},
        )

        nodes: list[TaskNode] = [
            _leaf(
                "translate",
                "Translate document",
                provides=["translated_doc"],
            ),
        ]

        dag = _make_dag_executor(executor, decomposer=decomposer)
        result = await dag.execute(nodes)

        # Re-decomposition should replace the failure
        assert result.success is True
        assert result.expansion_count >= 1
        # The decomposer should have been called for re-decomposition
        assert len(decomposer.call_log) == 1
        assert "failure" in decomposer.call_log[0].lower()


# ---------------------------------------------------------------------------
# 3. ConditionalNode routes correctly based on output
# ---------------------------------------------------------------------------


class TestConditionalNodeRouting:
    """A ConditionalNode evaluates its condition against predecessor output
    and routes execution to the correct branch, skipping the other."""

    async def test_conditional_routes_to_then_branch(self) -> None:
        """When predecessor output contains the condition target, the
        then_branch executes and else_branch is skipped."""
        executor = MockSimpleExecutor(
            response_map={
                "Check validity": "The data is valid and complete",
                "Process valid": "Processed valid data successfully",
            }
        )

        check_node = _leaf(
            "check", "Check validity", provides=["validity_result"]
        )
        cond_node = ConditionalNode(
            id="cond",
            description="Route based on validity",
            is_atomic=False,
            domain=Domain.SYNTHESIS,
            status=TaskStatus.READY,
            requires=["check"],
            consumes=["validity_result"],
            condition="contains 'valid'",
            then_branch=["process_valid"],
            else_branch=["handle_invalid"],
        )
        then_node = _leaf(
            "process_valid",
            "Process valid",
            requires=["cond"],
            consumes=["validity_result"],
        )
        else_node = _leaf(
            "handle_invalid",
            "Handle invalid",
            requires=["cond"],
            consumes=["validity_result"],
        )

        nodes: list[TaskNode] = [check_node, cond_node, then_node, else_node]

        dag = _make_dag_executor(executor)
        result = await dag.execute(nodes)

        assert result.success is True
        # then_branch should have executed
        executed_descriptions = executor.call_order
        assert any("Process valid" in d for d in executed_descriptions)
        # else_branch should be skipped (not executed)
        assert not any("Handle invalid" in d for d in executed_descriptions)
        # The else_node should be marked SKIPPED
        assert else_node.status == TaskStatus.SKIPPED

    async def test_conditional_routes_to_else_branch(self) -> None:
        """When predecessor output does NOT contain the condition target, the
        else_branch executes and then_branch is skipped."""
        executor = MockSimpleExecutor(
            response_map={
                "Check validity": "The data has errors and is incomplete",
                "Handle invalid": "Handled invalid data gracefully",
            }
        )

        check_node = _leaf(
            "check", "Check validity", provides=["validity_result"]
        )
        cond_node = ConditionalNode(
            id="cond",
            description="Route based on validity",
            is_atomic=False,
            domain=Domain.SYNTHESIS,
            status=TaskStatus.READY,
            requires=["check"],
            consumes=["validity_result"],
            condition="contains 'valid and complete'",
            then_branch=["process_valid"],
            else_branch=["handle_invalid"],
        )
        then_node = _leaf(
            "process_valid",
            "Process valid",
            requires=["cond"],
            consumes=["validity_result"],
        )
        else_node = _leaf(
            "handle_invalid",
            "Handle invalid",
            requires=["cond"],
            consumes=["validity_result"],
        )

        nodes: list[TaskNode] = [check_node, cond_node, then_node, else_node]

        dag = _make_dag_executor(executor)
        result = await dag.execute(nodes)

        assert result.success is True
        executed_descriptions = executor.call_order
        # else_branch should have executed
        assert any("Handle invalid" in d for d in executed_descriptions)
        # then_branch should be skipped
        assert not any("Process valid" in d for d in executed_descriptions)
        assert then_node.status == TaskStatus.SKIPPED


# ---------------------------------------------------------------------------
# 4. LoopNode retries and succeeds
# ---------------------------------------------------------------------------


class TestLoopNodeRetryAndSucceed:
    """A LoopNode iterates its body nodes until the exit condition is met.
    On the first iteration the exit condition is not satisfied; on the
    second iteration the output contains the exit token and the loop stops.

    The loop node's output is consumed by a downstream leaf that produces
    the final aggregated result (matching real DAG usage patterns)."""

    async def test_loop_retries_then_succeeds(self) -> None:
        call_count: int = 0

        class IteratingExecutor:
            """Executor that produces different output on successive calls.

            Calls matching the body task description produce iterating
            output. Other calls (e.g. the downstream consumer) succeed
            normally with echo-style output.
            """

            def __init__(self) -> None:
                self.call_order: list[str] = []

            async def execute(
                self,
                task: str,
                complexity: int = 1,
                provides_keys: list[str] | None = None,
            ) -> ExecutionResult:
                nonlocal call_count
                self.call_order.append(task)
                # Only increment for body task calls
                if "Process data" in task:
                    call_count += 1
                    if call_count <= 1:
                        output = "Partial result, needs more work"
                    else:
                        output = "Complete result DONE"
                else:
                    # Downstream consumer -- echo back the task description
                    # which will contain the forwarded loop output
                    output = f"Consumed: {task}"
                return ExecutionResult(
                    root_id=str(uuid.uuid4()),
                    output=output,
                    success=True,
                    total_nodes=1,
                    total_tokens=10,
                    total_latency_ms=1.0,
                    total_cost=0.001,
                )

        iterating_executor = IteratingExecutor()

        loop_node = LoopNode(
            id="retry_loop",
            description="Retry until complete",
            is_atomic=False,
            domain=Domain.SYNTHESIS,
            status=TaskStatus.READY,
            max_iterations=5,
            exit_condition="contains:DONE",
            body_nodes=["body_task"],
            provides=["loop_output"],
        )
        body_node = _leaf("body_task", "Process data")
        # Downstream leaf that consumes the loop's output, ensuring it
        # appears in the aggregated result.
        consumer_node = _leaf(
            "consumer",
            "Format loop output",
            requires=["retry_loop"],
            consumes=["loop_output"],
        )

        config = VerificationConfig(
            layer1_enabled=False,
            layer2_threshold=999,
            layer3_threshold=999,
        )
        dag = DAGExecutor(
            executor=iterating_executor,
            max_concurrency=10,
            verification_config=config,
        )

        result = await dag.execute([loop_node, body_node, consumer_node])

        assert result.success is True
        # The body task should have iterated exactly 2 times
        assert call_count == 2
        # The consumer leaf should have been called and its output aggregated
        assert any("Format loop output" in t or "loop_output" in t for t in iterating_executor.call_order)
        # The overall result should include the consumer's output
        assert result.output != ""
        assert result.total_nodes >= 2


# ---------------------------------------------------------------------------
# 5. Multi-wave DAG with wave events emitted
# ---------------------------------------------------------------------------


class TestMultiWaveEventsEmitted:
    """A DAG with two topological waves (wave 0: independent nodes,
    wave 1: dependent node) emits WaveCompleteEvent after each wave."""

    async def test_wave_events_emitted_in_order(self) -> None:
        captured_events: list[WaveCompleteEvent] = []

        def on_wave(event: WaveCompleteEvent) -> None:
            captured_events.append(event)

        executor = MockSimpleExecutor(
            response_map={
                "Fetch A": "Data from A",
                "Fetch B": "Data from B",
                "Merge results": "Merged A and B",
            }
        )

        nodes: list[TaskNode] = [
            _leaf("a", "Fetch A", provides=["data_a"]),
            _leaf("b", "Fetch B", provides=["data_b"]),
            _leaf(
                "merge",
                "Merge results",
                requires=["a", "b"],
                consumes=["data_a", "data_b"],
            ),
        ]

        dag = _make_dag_executor(executor, on_wave_complete=[on_wave])
        result = await dag.execute(nodes)

        assert result.success is True

        # There should be at least 2 waves: one for {a, b}, one for {merge}
        assert len(captured_events) >= 2

        # Wave 0: the two independent nodes
        wave_0 = captured_events[0]
        assert wave_0.wave_index == 0
        assert set(wave_0.completed_nodes) == {"a", "b"}
        # After wave 0, "merge" should still be remaining
        assert "merge" in wave_0.remaining_nodes
        # Accumulated results should contain a and b
        assert "a" in wave_0.accumulated_results
        assert "b" in wave_0.accumulated_results

        # Wave 1: the dependent merge node
        wave_1 = captured_events[1]
        assert wave_1.wave_index == 1
        assert "merge" in wave_1.completed_nodes
        # No remaining nodes after wave 1
        assert wave_1.remaining_nodes == []
        # Accumulated results should contain all three
        assert len(wave_1.accumulated_results) == 3


# ---------------------------------------------------------------------------
# 6. Expansion depth limit enforced
# ---------------------------------------------------------------------------


class TestExpansionDepthLimitEnforced:
    """When a leaf fails repeatedly, re-decomposition is attempted up to
    max_expansion_depth times. Beyond that limit, the failure is final."""

    async def test_depth_limit_prevents_infinite_expansion(self) -> None:
        # The decomposer always returns a single sub-leaf that also fails.
        # This would recurse forever without the depth limit.
        def make_failing_sub_dag() -> list[TaskNode]:
            node_id = f"sub_{uuid.uuid4().hex[:8]}"
            return [
                _leaf(node_id, "Doomed sub-task", provides=["result"]),
            ]

        # Provide enough responses for max_expansion_depth attempts
        decomposer = MockDecomposer(
            [make_failing_sub_dag() for _ in range(10)]
        )

        executor = MockSimpleExecutor(fail_on={"Doomed sub-task", "Failing root"})

        nodes: list[TaskNode] = [
            _leaf("root_fail", "Failing root", provides=["result"]),
        ]

        max_depth = 2
        dag = _make_dag_executor(
            executor,
            decomposer=decomposer,
            max_expansion_depth=max_depth,
        )
        result = await dag.execute(nodes)

        # The result should be a failure because the expansion depth is exhausted
        assert result.success is False
        assert len(result.errors) > 0
        # Decomposer should have been called at most max_depth times
        # (once for the root, once for the sub-DAG node at depth 1, then
        # depth 2 is hit and expansion stops)
        assert len(decomposer.call_log) <= max_depth

    async def test_depth_zero_prevents_all_expansion(self) -> None:
        """With max_expansion_depth=0, no re-decomposition is attempted."""
        sub_dag: list[TaskNode] = [
            _leaf("sub_a", "Recovery task"),
        ]
        decomposer = MockDecomposer([sub_dag])

        executor = MockSimpleExecutor(fail_on={"Failing leaf"})

        nodes: list[TaskNode] = [
            _leaf("fail_leaf", "Failing leaf"),
        ]

        dag = _make_dag_executor(
            executor,
            decomposer=decomposer,
            max_expansion_depth=0,
        )
        result = await dag.execute(nodes)

        assert result.success is False
        # Decomposer should never have been called
        assert len(decomposer.call_log) == 0
