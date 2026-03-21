"""Validate the real-world task suite definition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TASKS_FILE = Path(__file__).parent.parent.parent / "benchmarks" / "real_tasks.json"

REQUIRED_FIELDS = {"id", "description", "category", "expected_behavior", "tools_needed", "difficulty"}
REQUIRED_CATEGORIES = {"file", "web", "shell"}


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    """Load tasks from real_tasks.json."""
    assert TASKS_FILE.exists(), f"Missing {TASKS_FILE}"
    data = json.loads(TASKS_FILE.read_text())
    assert "tasks" in data, "real_tasks.json must have a 'tasks' key"
    return data["tasks"]


class TestRealTasksStructure:
    """Verify the real_tasks.json file is well-formed."""

    def test_has_10_tasks(self, tasks: list[dict]) -> None:
        assert len(tasks) == 10, f"Expected 10 tasks, got {len(tasks)}"

    def test_unique_ids(self, tasks: list[dict]) -> None:
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs found: {ids}"

    def test_required_fields(self, tasks: list[dict]) -> None:
        for task in tasks:
            missing = REQUIRED_FIELDS - set(task.keys())
            assert not missing, (
                f"Task {task.get('id', '???')} missing fields: {missing}"
            )

    def test_all_categories_represented(self, tasks: list[dict]) -> None:
        categories = {t["category"] for t in tasks}
        missing = REQUIRED_CATEGORIES - categories
        assert not missing, f"Missing categories: {missing}"

    def test_ids_follow_pattern(self, tasks: list[dict]) -> None:
        for task in tasks:
            assert task["id"].startswith("real_"), (
                f"Task ID {task['id']} does not start with 'real_'"
            )

    def test_tools_needed_is_list(self, tasks: list[dict]) -> None:
        for task in tasks:
            assert isinstance(task["tools_needed"], list), (
                f"Task {task['id']} tools_needed must be a list"
            )
            assert len(task["tools_needed"]) >= 1, (
                f"Task {task['id']} must need at least one tool"
            )

    def test_difficulty_range(self, tasks: list[dict]) -> None:
        for task in tasks:
            assert 1 <= task["difficulty"] <= 5, (
                f"Task {task['id']} difficulty {task['difficulty']} out of range [1,5]"
            )

    def test_descriptions_non_empty(self, tasks: list[dict]) -> None:
        for task in tasks:
            assert len(task["description"]) >= 10, (
                f"Task {task['id']} has too short a description"
            )

    def test_category_counts(self, tasks: list[dict]) -> None:
        from collections import Counter
        counts = Counter(t["category"] for t in tasks)
        # At least 2 tasks per category
        for cat in REQUIRED_CATEGORIES:
            assert counts[cat] >= 2, (
                f"Category '{cat}' has only {counts[cat]} tasks, need at least 2"
            )
