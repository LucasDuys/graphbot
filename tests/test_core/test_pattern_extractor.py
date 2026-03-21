"""Tests for pattern extraction from completed task trees."""

from __future__ import annotations

import json

from core_gb.patterns import PatternExtractor
from core_gb.types import (
    Domain,
    ExecutionResult,
    FlowType,
    Pattern,
    TaskNode,
    TaskStatus,
)


def _make_weather_tree() -> list[TaskNode]:
    """Build a parallel weather tree: root + 3 city leaves + aggregation leaf."""
    root = TaskNode(
        id="root",
        description="Weather in Amsterdam, London, and Berlin",
        children=["leaf_ams", "leaf_lon", "leaf_ber", "agg"],
        domain=Domain.SYNTHESIS,
        complexity=2,
        flow_type=FlowType.PARALLEL,
    )
    leaf_ams = TaskNode(
        id="leaf_ams",
        description="Current weather in Amsterdam",
        parent_id="root",
        is_atomic=True,
        domain=Domain.WEB,
        complexity=1,
        provides=["weather_amsterdam"],
    )
    leaf_lon = TaskNode(
        id="leaf_lon",
        description="Current weather in London",
        parent_id="root",
        is_atomic=True,
        domain=Domain.WEB,
        complexity=1,
        provides=["weather_london"],
    )
    leaf_ber = TaskNode(
        id="leaf_ber",
        description="Current weather in Berlin",
        parent_id="root",
        is_atomic=True,
        domain=Domain.WEB,
        complexity=1,
        provides=["weather_berlin"],
    )
    agg = TaskNode(
        id="agg",
        description="Aggregate weather results",
        parent_id="root",
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=1,
        consumes=["weather_amsterdam", "weather_london", "weather_berlin"],
        provides=["weather_summary"],
    )
    return [root, leaf_ams, leaf_lon, leaf_ber, agg]


def _make_sequential_tree() -> list[TaskNode]:
    """Build a sequential tree: root -> read -> parse -> format."""
    root = TaskNode(
        id="root",
        description="Read README, find TODOs, list with line numbers",
        children=["read", "parse", "format"],
        flow_type=FlowType.SEQUENCE,
    )
    read = TaskNode(
        id="read",
        description="Read README.md file",
        parent_id="root",
        is_atomic=True,
        domain=Domain.FILE,
        provides=["file_content"],
    )
    parse = TaskNode(
        id="parse",
        description="Parse TODO comments from file content",
        parent_id="root",
        is_atomic=True,
        domain=Domain.CODE,
        requires=["read"],
        consumes=["file_content"],
        provides=["todo_list"],
    )
    fmt = TaskNode(
        id="format",
        description="Format TODO list with line numbers",
        parent_id="root",
        is_atomic=True,
        domain=Domain.CODE,
        requires=["parse"],
        consumes=["todo_list"],
        provides=["formatted_output"],
    )
    return [root, read, parse, fmt]


def _success_result(root_id: str = "root") -> ExecutionResult:
    return ExecutionResult(
        root_id=root_id,
        output="done",
        success=True,
        total_nodes=5,
        total_tokens=320,
        total_latency_ms=850.0,
    )


def _failed_result(root_id: str = "root") -> ExecutionResult:
    return ExecutionResult(
        root_id=root_id,
        output="",
        success=False,
        errors=("timeout",),
    )


class TestPatternExtractor:
    def test_extract_from_parallel_tree(self) -> None:
        nodes = _make_weather_tree()
        result = _success_result()
        extractor = PatternExtractor()

        pattern = extractor.extract(
            task="Weather in Amsterdam, London, and Berlin",
            nodes=nodes,
            result=result,
        )

        assert pattern is not None
        assert isinstance(pattern, Pattern)
        assert pattern.success_count == 1
        assert pattern.avg_tokens == 320.0
        assert pattern.avg_latency_ms == 850.0
        assert len(pattern.variable_slots) > 0
        assert pattern.tree_template != ""

    def test_extract_single_node_returns_none(self) -> None:
        nodes = [
            TaskNode(
                id="only",
                description="Simple math question",
                is_atomic=True,
                domain=Domain.SYSTEM,
            ),
        ]
        result = _success_result(root_id="only")
        extractor = PatternExtractor()

        pattern = extractor.extract(
            task="What is 2+2?",
            nodes=nodes,
            result=result,
        )

        assert pattern is None

    def test_extract_failed_execution_returns_none(self) -> None:
        nodes = _make_weather_tree()
        result = _failed_result()
        extractor = PatternExtractor()

        pattern = extractor.extract(
            task="Weather in Amsterdam, London, and Berlin",
            nodes=nodes,
            result=result,
        )

        assert pattern is None

    def test_generalize_replaces_entities(self) -> None:
        nodes = _make_weather_tree()
        leaves = [n for n in nodes if n.is_atomic]
        extractor = PatternExtractor()

        trigger, slots, bindings = extractor._generalize(
            task="Weather in Amsterdam, London, and Berlin",
            leaves=leaves,
        )

        # The trigger should contain at least one slot placeholder
        assert "{slot_" in trigger
        assert len(slots) > 0
        # Every slot should have a binding to a real value
        for slot in slots:
            assert slot in bindings
            assert len(bindings[slot]) >= 3

    def test_template_serialization(self) -> None:
        nodes = _make_weather_tree()
        result = _success_result()
        extractor = PatternExtractor()

        pattern = extractor.extract(
            task="Weather in Amsterdam, London, and Berlin",
            nodes=nodes,
            result=result,
        )

        assert pattern is not None
        parsed = json.loads(pattern.tree_template)
        assert isinstance(parsed, list)
        assert len(parsed) == len(nodes)
        for entry in parsed:
            assert "description" in entry
            assert "domain" in entry
            assert "is_atomic" in entry

    def test_extract_sequential_tree(self) -> None:
        nodes = _make_sequential_tree()
        result = ExecutionResult(
            root_id="root",
            output="formatted todos",
            success=True,
            total_nodes=4,
            total_tokens=200,
            total_latency_ms=600.0,
        )
        extractor = PatternExtractor()

        pattern = extractor.extract(
            task="Read README, find TODOs, list with line numbers",
            nodes=nodes,
            result=result,
        )

        assert pattern is not None
        assert isinstance(pattern, Pattern)
        assert pattern.success_count == 1
        assert pattern.avg_tokens == 200.0
        assert pattern.avg_latency_ms == 600.0
        parsed = json.loads(pattern.tree_template)
        assert len(parsed) == 4
