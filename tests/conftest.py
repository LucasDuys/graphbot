"""Pytest fixtures for GraphBot tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from core_gb.types import Domain, FlowType, TaskNode, TaskStatus


@pytest.fixture
def tmp_graph_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for Kuzu graph storage."""
    with tempfile.TemporaryDirectory(prefix="graphbot_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_task() -> TaskNode:
    """A simple atomic task node for testing."""
    return TaskNode(
        id="task_001",
        description="What's 247 * 38?",
        is_atomic=True,
        domain=Domain.SYSTEM,
        complexity=1,
        status=TaskStatus.READY,
    )


@pytest.fixture
def parallel_tree() -> dict[str, TaskNode]:
    """A parallel task tree (weather in 3 cities)."""
    root = TaskNode(
        id="root",
        description="Weather in Amsterdam, London, and Berlin",
        children=["leaf_ams", "leaf_lon", "leaf_ber"],
        domain=Domain.SYNTHESIS,
        complexity=2,
        flow_type=FlowType.PARALLEL,
    )
    leaves = [
        TaskNode(
            id="leaf_ams",
            description="Current weather in Amsterdam",
            parent_id="root",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            provides=["weather_amsterdam"],
        ),
        TaskNode(
            id="leaf_lon",
            description="Current weather in London",
            parent_id="root",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            provides=["weather_london"],
        ),
        TaskNode(
            id="leaf_ber",
            description="Current weather in Berlin",
            parent_id="root",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            provides=["weather_berlin"],
        ),
    ]
    nodes = {root.id: root}
    for leaf in leaves:
        nodes[leaf.id] = leaf
    return nodes


@pytest.fixture
def dependent_tree() -> dict[str, TaskNode]:
    """A sequential dependent task tree (read -> parse -> format)."""
    return {
        "root": TaskNode(
            id="root",
            description="Read README, find TODOs, list with line numbers",
            children=["read", "parse", "format"],
            flow_type=FlowType.SEQUENCE,
        ),
        "read": TaskNode(
            id="read",
            description="Read README.md file",
            parent_id="root",
            is_atomic=True,
            domain=Domain.FILE,
            provides=["file_content"],
        ),
        "parse": TaskNode(
            id="parse",
            description="Parse TODO comments from file content",
            parent_id="root",
            is_atomic=True,
            domain=Domain.CODE,
            requires=["read"],
            consumes=["file_content"],
            provides=["todo_list"],
        ),
        "format": TaskNode(
            id="format",
            description="Format TODO list with line numbers",
            parent_id="root",
            is_atomic=True,
            domain=Domain.CODE,
            requires=["parse"],
            consumes=["todo_list"],
            provides=["formatted_output"],
        ),
    }
