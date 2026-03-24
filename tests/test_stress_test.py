"""Tests for scripts/stress_test.py -- task definitions, diagnosis, and dry run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure the scripts directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from stress_test import (
    STRESS_TASKS,
    DiagnosisResult,
    FailureCategory,
    StressTask,
    StressTestEntry,
    diagnose_failure,
    dry_run_report,
)


class TestStressTaskDefinitions:
    """Verify all 10 stress tasks are properly defined."""

    def test_exactly_10_tasks(self) -> None:
        assert len(STRESS_TASKS) == 10

    def test_all_ids_unique(self) -> None:
        ids = [t.id for t in STRESS_TASKS]
        assert len(ids) == len(set(ids))

    def test_all_categories_present(self) -> None:
        categories = {t.category for t in STRESS_TASKS}
        expected = {
            "reasoning",
            "ambiguity",
            "tool_chain",
            "tool_discovery",
            "meta_query",
            "contradiction",
            "parallel",
            "decomposition",
            "temporal",
            "meta",
        }
        assert categories == expected

    def test_all_tasks_have_required_fields(self) -> None:
        for task in STRESS_TASKS:
            assert task.id, f"Task missing id"
            assert task.name, f"Task {task.id} missing name"
            assert task.description, f"Task {task.id} missing description"
            assert task.why_hard, f"Task {task.id} missing why_hard"
            assert task.expected_behavior, f"Task {task.id} missing expected_behavior"
            assert 1 <= task.difficulty <= 5, (
                f"Task {task.id} difficulty {task.difficulty} out of range"
            )

    def test_multi_hop_task(self) -> None:
        task = STRESS_TASKS[0]
        assert task.id == "stress_01_multi_hop"
        assert "capital of France" in task.description
        assert "16th letter" in task.description

    def test_ambiguous_task(self) -> None:
        task = STRESS_TASKS[1]
        assert task.id == "stress_02_ambiguous"
        assert task.description == "Make it better"
        assert task.accept_partial is True

    def test_contradictory_task(self) -> None:
        task = STRESS_TASKS[5]
        assert task.id == "stress_06_contradictory"
        assert "5-word" in task.description
        assert "500 words" in task.description


class TestDiagnosis:
    """Test failure root cause diagnosis."""

    def _make_task(self, category: str = "reasoning") -> StressTask:
        return StressTask(
            id="test",
            name="test",
            category=category,
            description="test",
            difficulty=3,
            expected_behavior="test",
            why_hard="test",
        )

    def _make_result(
        self,
        success: bool = False,
        output: str = "",
        errors: tuple[str, ...] = (),
        total_nodes: int = 1,
        tools_used: int = 0,
    ) -> MagicMock:
        result = MagicMock()
        result.success = success
        result.output = output
        result.errors = errors
        result.total_nodes = total_nodes
        result.tools_used = tools_used
        return result

    def test_timeout_by_elapsed(self) -> None:
        task = self._make_task()
        diag = diagnose_failure(task, self._make_result(), None, 70_000)
        assert diag.category == FailureCategory.TIMEOUT

    def test_timeout_by_exception(self) -> None:
        task = self._make_task()
        diag = diagnose_failure(task, None, TimeoutError("timed out"), 5_000)
        assert diag.category == FailureCategory.TIMEOUT

    def test_tool_failure_exception(self) -> None:
        task = self._make_task()
        diag = diagnose_failure(task, None, RuntimeError("tool registry error"), 1_000)
        assert diag.category == FailureCategory.TOOL_FAILURE

    def test_runtime_error(self) -> None:
        task = self._make_task()
        diag = diagnose_failure(task, None, ValueError("bad value"), 1_000)
        assert diag.category == FailureCategory.RUNTIME_ERROR

    def test_safety_blocked(self) -> None:
        task = self._make_task()
        result = self._make_result(errors=("blocked by safety filter",))
        diag = diagnose_failure(task, result, None, 1_000)
        assert diag.category == FailureCategory.SAFETY_BLOCKED

    def test_model_refusal(self) -> None:
        task = self._make_task()
        result = self._make_result(output="I cannot help with that")
        diag = diagnose_failure(task, result, None, 1_000)
        assert diag.category == FailureCategory.MODEL_REFUSAL

    def test_decomposition_error(self) -> None:
        task = self._make_task(category="decomposition")
        result = self._make_result(total_nodes=1)
        diag = diagnose_failure(task, result, None, 1_000)
        assert diag.category == FailureCategory.DECOMPOSITION_ERROR

    def test_contradiction_detection(self) -> None:
        task = self._make_task(category="contradiction")
        result = self._make_result()
        diag = diagnose_failure(task, result, None, 1_000)
        assert diag.category == FailureCategory.CONTRADICTION

    def test_context_missing(self) -> None:
        task = self._make_task(category="meta_query")
        result = self._make_result()
        diag = diagnose_failure(task, result, None, 1_000)
        assert diag.category == FailureCategory.CONTEXT_MISSING

    def test_tool_chain_no_tools(self) -> None:
        task = self._make_task(category="tool_chain")
        result = self._make_result(tools_used=0)
        diag = diagnose_failure(task, result, None, 1_000)
        assert diag.category == FailureCategory.TOOL_FAILURE

    def test_unknown_fallback(self) -> None:
        task = self._make_task(category="parallel")
        result = self._make_result()
        diag = diagnose_failure(task, result, None, 1_000)
        assert diag.category == FailureCategory.UNKNOWN


class TestDryRunReport:
    """Test dry run report generation."""

    def test_report_structure(self) -> None:
        report = dry_run_report(STRESS_TASKS)
        assert report["mode"] == "dry_run"
        assert report["total_tasks"] == 10
        assert len(report["tasks"]) == 10

    def test_all_tasks_marked_skipped(self) -> None:
        report = dry_run_report(STRESS_TASKS)
        for task_entry in report["tasks"]:
            assert task_entry["status"] == "dry_run_skipped"

    def test_avg_difficulty_calculated(self) -> None:
        report = dry_run_report(STRESS_TASKS)
        expected_avg = sum(t.difficulty for t in STRESS_TASKS) / len(STRESS_TASKS)
        assert report["avg_difficulty"] == expected_avg


class TestStressTestEntry:
    """Test result entry serialization."""

    def test_to_dict(self) -> None:
        entry = StressTestEntry(
            task_id="stress_01",
            name="Test",
            category="reasoning",
            difficulty=3,
            description="test desc",
            why_hard="test why",
            success=True,
            output_preview="some output",
            total_nodes=5,
            total_tokens=1000,
            latency_ms=1234.5,
            cost=0.001,
            model_used="test-model",
            tools_used=2,
            llm_calls=3,
        )
        d = entry.to_dict()
        assert d["task_id"] == "stress_01"
        assert d["success"] is True
        assert d["latency_ms"] == 1234  # round(1234.5) uses banker's rounding
        assert d["total_nodes"] == 5
        assert d["diagnosis"] == {}

    def test_to_dict_with_diagnosis(self) -> None:
        entry = StressTestEntry(
            task_id="stress_02",
            name="Test",
            category="ambiguity",
            difficulty=4,
            description="test",
            why_hard="test",
            success=False,
            diagnosis={
                "category": "unknown",
                "root_cause": "test root cause",
                "suggestion": "test suggestion",
            },
        )
        d = entry.to_dict()
        assert d["success"] is False
        assert d["diagnosis"]["category"] == "unknown"
