"""Tests for benchmark task suite structure and schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TASKS_FILE = Path(__file__).parent.parent.parent / "benchmarks" / "tasks.json"

REQUIRED_FIELDS: set[str] = {
    "id",
    "description",
    "category",
    "expected_behavior",
    "expected_min_nodes",
    "difficulty",
}


@pytest.fixture()
def tasks() -> list[dict]:
    """Load and return the benchmark task list."""
    assert TASKS_FILE.exists(), f"tasks.json not found at {TASKS_FILE}"
    data = json.loads(TASKS_FILE.read_text())
    assert "tasks" in data, "tasks.json must have a top-level 'tasks' key"
    return data["tasks"]


class TestBenchmarkTaskSuite:
    """Validate the benchmark task suite schema and structure."""

    def test_has_15_tasks(self, tasks: list[dict]) -> None:
        """The suite must contain exactly 15 benchmark tasks."""
        assert len(tasks) == 15, f"Expected 15 tasks, got {len(tasks)}"

    def test_each_task_has_required_fields(self, tasks: list[dict]) -> None:
        """Every task must include all required fields."""
        for task in tasks:
            missing = REQUIRED_FIELDS - set(task.keys())
            assert not missing, (
                f"Task {task.get('id', '???')} missing fields: {missing}"
            )

    def test_unique_ids(self, tasks: list[dict]) -> None:
        """All task ids must be unique."""
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), f"Duplicate task ids found: {ids}"

    def test_five_simple_tasks(self, tasks: list[dict]) -> None:
        """There must be exactly 5 simple-category tasks."""
        simple = [t for t in tasks if t["category"] == "simple"]
        assert len(simple) == 5, f"Expected 5 simple tasks, got {len(simple)}"

    def test_five_parallel_tasks(self, tasks: list[dict]) -> None:
        """There must be exactly 5 parallel-category tasks."""
        parallel = [t for t in tasks if t["category"] == "parallel"]
        assert len(parallel) == 5, f"Expected 5 parallel tasks, got {len(parallel)}"

    def test_five_sequential_tasks(self, tasks: list[dict]) -> None:
        """There must be exactly 5 sequential-category tasks."""
        sequential = [t for t in tasks if t["category"] == "sequential"]
        assert len(sequential) == 5, f"Expected 5 sequential tasks, got {len(sequential)}"

    def test_expected_behavior_values(self, tasks: list[dict]) -> None:
        """expected_behavior must be one of: direct, parallel, sequential."""
        allowed = {"direct", "parallel", "sequential"}
        for task in tasks:
            assert task["expected_behavior"] in allowed, (
                f"Task {task['id']} has invalid expected_behavior: "
                f"{task['expected_behavior']}"
            )

    def test_difficulty_in_valid_range(self, tasks: list[dict]) -> None:
        """difficulty must be an integer between 1 and 5."""
        for task in tasks:
            d = task["difficulty"]
            assert isinstance(d, int), (
                f"Task {task['id']} difficulty must be int, got {type(d).__name__}"
            )
            assert 1 <= d <= 5, (
                f"Task {task['id']} difficulty {d} out of range [1, 5]"
            )

    def test_expected_min_nodes_positive(self, tasks: list[dict]) -> None:
        """expected_min_nodes must be a positive integer."""
        for task in tasks:
            n = task["expected_min_nodes"]
            assert isinstance(n, int) and n >= 1, (
                f"Task {task['id']} expected_min_nodes must be >= 1, got {n}"
            )

    def test_expected_answer_contains_is_list(self, tasks: list[dict]) -> None:
        """expected_answer_contains, when present, must be a list of strings."""
        for task in tasks:
            if "expected_answer_contains" in task:
                val = task["expected_answer_contains"]
                assert isinstance(val, list), (
                    f"Task {task['id']} expected_answer_contains must be list"
                )
                for item in val:
                    assert isinstance(item, str), (
                        f"Task {task['id']} expected_answer_contains items must be str"
                    )

    def test_simple_tasks_expect_one_node(self, tasks: list[dict]) -> None:
        """Simple tasks should have expected_min_nodes == 1."""
        for task in tasks:
            if task["category"] == "simple":
                assert task["expected_min_nodes"] == 1, (
                    f"Simple task {task['id']} should have expected_min_nodes=1"
                )

    def test_complex_tasks_expect_multiple_nodes(self, tasks: list[dict]) -> None:
        """Parallel and sequential tasks should have expected_min_nodes >= 3."""
        for task in tasks:
            if task["category"] in ("parallel", "sequential"):
                assert task["expected_min_nodes"] >= 3, (
                    f"Complex task {task['id']} should have expected_min_nodes >= 3, "
                    f"got {task['expected_min_nodes']}"
                )
