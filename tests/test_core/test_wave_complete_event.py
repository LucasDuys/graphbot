"""Tests for wave-complete event emission from DAGExecutor."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)
from core_gb.wave_event import WaveCompleteEvent


class MockWaveExecutor:
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


class TestWaveCompleteEventDataclass:
    """Tests for WaveCompleteEvent structure."""

    def test_wave_complete_event_fields(self) -> None:
        event = WaveCompleteEvent(
            wave_index=0,
            completed_nodes=["a", "b"],
            accumulated_results={"a": "result_a", "b": "result_b"},
            remaining_nodes=["c"],
        )
        assert event.wave_index == 0
        assert event.completed_nodes == ["a", "b"]
        assert event.accumulated_results == {"a": "result_a", "b": "result_b"}
        assert event.remaining_nodes == ["c"]

    def test_wave_complete_event_is_frozen(self) -> None:
        event = WaveCompleteEvent(
            wave_index=0,
            completed_nodes=["a"],
            accumulated_results={"a": "result"},
            remaining_nodes=[],
        )
        with pytest.raises(AttributeError):
            event.wave_index = 1  # type: ignore[misc]


class TestThreeWaveDAGEmitsThreeEvents:
    """A 3-wave DAG (A -> B -> C) emits exactly 3 wave-complete events."""

    async def test_three_wave_dag_emits_three_events(self) -> None:
        mock = MockWaveExecutor(delay=0.0)
        captured_events: list[WaveCompleteEvent] = []

        def on_wave(event: WaveCompleteEvent) -> None:
            captured_events.append(event)

        dag = DAGExecutor(
            executor=mock,
            max_concurrency=10,
            on_wave_complete=[on_wave],
        )

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", requires=["a"], consumes=["data_a"], provides=["data_b"]),
            _make_leaf("c", "Task C", requires=["b"], consumes=["data_b"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert len(captured_events) == 3

        # Wave 0: node A completes
        assert captured_events[0].wave_index == 0
        assert "a" in captured_events[0].completed_nodes
        assert "b" in captured_events[0].remaining_nodes
        assert "c" in captured_events[0].remaining_nodes

        # Wave 1: node B completes
        assert captured_events[1].wave_index == 1
        assert "b" in captured_events[1].completed_nodes
        assert "c" in captured_events[1].remaining_nodes
        assert "a" not in captured_events[1].remaining_nodes

        # Wave 2: node C completes
        assert captured_events[2].wave_index == 2
        assert "c" in captured_events[2].completed_nodes
        assert captured_events[2].remaining_nodes == []

    async def test_event_accumulated_results_grow(self) -> None:
        """Each wave event has accumulated results from all waves up to and including it."""
        mock = MockWaveExecutor(delay=0.0)
        captured_events: list[WaveCompleteEvent] = []

        dag = DAGExecutor(
            executor=mock,
            max_concurrency=10,
            on_wave_complete=[lambda e: captured_events.append(e)],
        )

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", requires=["a"], consumes=["data_a"], provides=["data_b"]),
            _make_leaf("c", "Task C", requires=["b"], consumes=["data_b"]),
        ]

        await dag.execute(nodes)

        # Wave 0: only A's result
        assert "a" in captured_events[0].accumulated_results
        assert "b" not in captured_events[0].accumulated_results

        # Wave 1: A + B results
        assert "a" in captured_events[1].accumulated_results
        assert "b" in captured_events[1].accumulated_results
        assert "c" not in captured_events[1].accumulated_results

        # Wave 2: A + B + C results
        assert "a" in captured_events[2].accumulated_results
        assert "b" in captured_events[2].accumulated_results
        assert "c" in captured_events[2].accumulated_results


class TestParallelWaveEvents:
    """Parallel nodes within the same wave are grouped into a single event."""

    async def test_parallel_wave_emits_single_event(self) -> None:
        """Two independent nodes (A, B) form wave 0; C depends on both forms wave 1."""
        mock = MockWaveExecutor(delay=0.0)
        captured_events: list[WaveCompleteEvent] = []

        dag = DAGExecutor(
            executor=mock,
            max_concurrency=10,
            on_wave_complete=[lambda e: captured_events.append(e)],
        )

        nodes = [
            _make_leaf("a", "Task A", provides=["data_a"]),
            _make_leaf("b", "Task B", provides=["data_b"]),
            _make_leaf("c", "Task C", requires=["a", "b"], consumes=["data_a", "data_b"]),
        ]

        result = await dag.execute(nodes)

        assert result.success is True
        assert len(captured_events) == 2

        # Wave 0: both A and B complete
        assert captured_events[0].wave_index == 0
        assert sorted(captured_events[0].completed_nodes) == ["a", "b"]
        assert captured_events[0].remaining_nodes == ["c"]

        # Wave 1: C completes
        assert captured_events[1].wave_index == 1
        assert captured_events[1].completed_nodes == ["c"]
        assert captured_events[1].remaining_nodes == []


class TestMultipleCallbacks:
    """Multiple callbacks all receive the same events."""

    async def test_multiple_callbacks_all_called(self) -> None:
        mock = MockWaveExecutor(delay=0.0)
        captured_1: list[WaveCompleteEvent] = []
        captured_2: list[WaveCompleteEvent] = []

        dag = DAGExecutor(
            executor=mock,
            max_concurrency=10,
            on_wave_complete=[
                lambda e: captured_1.append(e),
                lambda e: captured_2.append(e),
            ],
        )

        nodes = [
            _make_leaf("a", "Task A"),
            _make_leaf("b", "Task B"),
        ]

        await dag.execute(nodes)

        assert len(captured_1) == 1
        assert len(captured_2) == 1
        assert captured_1[0].wave_index == captured_2[0].wave_index


class TestNoCallbacksIsDefault:
    """DAGExecutor works without callbacks (backward compatible)."""

    async def test_no_callbacks_default(self) -> None:
        mock = MockWaveExecutor(delay=0.0)
        dag = DAGExecutor(executor=mock, max_concurrency=10)

        nodes = [
            _make_leaf("a", "Task A"),
            _make_leaf("b", "Task B", requires=["a"]),
        ]

        result = await dag.execute(nodes)
        assert result.success is True


class TestEmptyDAGNoEvents:
    """Empty node list emits no wave events."""

    async def test_empty_dag_no_events(self) -> None:
        mock = MockWaveExecutor(delay=0.0)
        captured_events: list[WaveCompleteEvent] = []

        dag = DAGExecutor(
            executor=mock,
            max_concurrency=10,
            on_wave_complete=[lambda e: captured_events.append(e)],
        )

        result = await dag.execute([])

        assert result.success is True
        assert len(captured_events) == 0
