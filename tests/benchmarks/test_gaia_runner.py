"""Tests for the GAIA Level 1 benchmark runner (scripts/run_gaia.py).

All tests use a mocked Orchestrator so no real LLM calls are made.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is importable
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core_gb.types import ExecutionResult
from scripts.run_gaia import (
    FALLBACK_TASKS,
    answers_match,
    load_gaia_tasks,
    normalize_answer,
    run_gaia_benchmark,
    save_results,
)


# ---------------------------------------------------------------------------
# normalize_answer / answers_match
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    """Unit tests for answer normalization."""

    def test_strips_whitespace(self) -> None:
        assert normalize_answer("  Paris  ") == "paris"

    def test_strips_trailing_punctuation(self) -> None:
        assert normalize_answer("Paris.") == "paris"
        assert normalize_answer("Paris!") == "paris"
        assert normalize_answer("Paris,") == "paris"

    def test_lowercases(self) -> None:
        assert normalize_answer("PARIS") == "paris"

    def test_empty_string(self) -> None:
        assert normalize_answer("") == ""


class TestAnswersMatch:
    """Unit tests for answer comparison logic."""

    def test_exact_match(self) -> None:
        assert answers_match("Paris", "Paris") is True

    def test_case_insensitive_match(self) -> None:
        assert answers_match("paris", "Paris") is True

    def test_substring_match(self) -> None:
        assert answers_match("The capital of France is Paris.", "Paris") is True

    def test_no_match(self) -> None:
        assert answers_match("London", "Paris") is False

    def test_punctuation_ignored(self) -> None:
        assert answers_match("Paris.", "Paris") is True

    def test_numeric_match(self) -> None:
        assert answers_match("8", "8") is True

    def test_numeric_in_sentence(self) -> None:
        assert answers_match("There are 8 planets.", "8") is True


# ---------------------------------------------------------------------------
# Fallback tasks
# ---------------------------------------------------------------------------


class TestFallbackTasks:
    """Validate the hardcoded fallback task set."""

    def test_at_least_20_tasks(self) -> None:
        assert len(FALLBACK_TASKS) >= 20

    def test_each_task_has_required_keys(self) -> None:
        for task in FALLBACK_TASKS:
            assert "question" in task, f"Missing 'question' key: {task}"
            assert "ground_truth" in task, f"Missing 'ground_truth' key: {task}"

    def test_no_empty_fields(self) -> None:
        for task in FALLBACK_TASKS:
            assert task["question"].strip(), f"Empty question: {task}"
            assert task["ground_truth"].strip(), f"Empty ground_truth: {task}"


# ---------------------------------------------------------------------------
# load_gaia_tasks
# ---------------------------------------------------------------------------


class TestLoadGaiaTasks:
    """Test task loading with HuggingFace fallback."""

    def test_returns_fallback_when_no_datasets(self) -> None:
        tasks = load_gaia_tasks()
        assert len(tasks) >= 20

    def test_all_tasks_have_keys(self) -> None:
        tasks = load_gaia_tasks()
        for task in tasks:
            assert "question" in task
            assert "ground_truth" in task


# ---------------------------------------------------------------------------
# run_gaia_benchmark (with mocked orchestrator)
# ---------------------------------------------------------------------------


def _make_mock_result(output: str) -> ExecutionResult:
    """Create a minimal ExecutionResult with the given output."""
    return ExecutionResult(
        root_id="mock",
        output=output,
        success=True,
        total_nodes=1,
        total_tokens=50,
        total_latency_ms=100.0,
        total_cost=0.0001,
        model_used="mock-model",
    )


class TestRunGaiaBenchmark:
    """Integration tests for the benchmark runner with mocked orchestrator."""

    @pytest.fixture()
    def mock_orchestrator(self) -> MagicMock:
        """Return a mock orchestrator whose process() returns correct answers."""
        orch = MagicMock()
        # Default: return "Paris" for any question -- will match first task
        orch.process = AsyncMock(return_value=_make_mock_result("Paris"))
        return orch

    def test_runs_all_tasks(self, mock_orchestrator: MagicMock) -> None:
        """Benchmark runner calls process() for each task."""
        tasks = [
            {"question": "Capital of France?", "ground_truth": "Paris"},
            {"question": "Capital of Germany?", "ground_truth": "Berlin"},
        ]
        summary = asyncio.run(
            run_gaia_benchmark(
                tasks=tasks, orchestrator=mock_orchestrator
            )
        )
        assert mock_orchestrator.process.call_count == 2
        assert summary["total_tasks"] == 2

    def test_correct_counting(self, mock_orchestrator: MagicMock) -> None:
        """Only matching answers are counted as correct."""
        tasks = [
            {"question": "Capital of France?", "ground_truth": "Paris"},
            {"question": "Capital of Germany?", "ground_truth": "Berlin"},
        ]
        # process() returns "Paris" for all -- only first should match
        summary = asyncio.run(
            run_gaia_benchmark(
                tasks=tasks, orchestrator=mock_orchestrator
            )
        )
        assert summary["correct"] == 1
        assert summary["accuracy"] == 0.5

    def test_token_and_cost_accumulation(self, mock_orchestrator: MagicMock) -> None:
        """Tokens and cost are accumulated across tasks."""
        tasks = [
            {"question": "Q1?", "ground_truth": "A1"},
            {"question": "Q2?", "ground_truth": "A2"},
        ]
        summary = asyncio.run(
            run_gaia_benchmark(
                tasks=tasks, orchestrator=mock_orchestrator
            )
        )
        # Each mock result has 50 tokens and 0.0001 cost
        assert summary["total_tokens"] == 100
        assert summary["total_cost"] == pytest.approx(0.0002, abs=1e-8)

    def test_handles_orchestrator_exception(self) -> None:
        """Tasks that raise exceptions are recorded as failures."""
        orch = MagicMock()
        orch.process = AsyncMock(side_effect=RuntimeError("LLM down"))
        tasks = [{"question": "Q?", "ground_truth": "A"}]
        summary = asyncio.run(
            run_gaia_benchmark(tasks=tasks, orchestrator=orch)
        )
        assert summary["correct"] == 0
        assert summary["results"][0]["success"] is False
        assert "error" in summary["results"][0]

    def test_summary_has_required_fields(self, mock_orchestrator: MagicMock) -> None:
        """Summary dict must include all expected top-level keys."""
        tasks = [{"question": "Q?", "ground_truth": "A"}]
        summary = asyncio.run(
            run_gaia_benchmark(
                tasks=tasks, orchestrator=mock_orchestrator
            )
        )
        required_keys = {
            "timestamp",
            "total_tasks",
            "correct",
            "accuracy",
            "total_tokens",
            "total_cost",
            "results",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_individual_results_structure(self, mock_orchestrator: MagicMock) -> None:
        """Each result entry must contain expected fields."""
        tasks = [{"question": "Q?", "ground_truth": "A"}]
        summary = asyncio.run(
            run_gaia_benchmark(
                tasks=tasks, orchestrator=mock_orchestrator
            )
        )
        result = summary["results"][0]
        for field in ("question", "ground_truth", "predicted", "match",
                       "tokens", "cost", "latency_ms", "model", "success"):
            assert field in result, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    """Test results persistence."""

    def test_creates_json_file(self, tmp_path: Path) -> None:
        """save_results writes valid JSON to disk."""
        summary: dict[str, Any] = {
            "timestamp": "2024-01-01T00:00:00Z",
            "total_tasks": 1,
            "correct": 1,
            "accuracy": 1.0,
            "total_tokens": 50,
            "total_cost": 0.0001,
            "results": [],
        }
        out_path = tmp_path / "benchmarks" / "gaia_results.json"

        with patch(
            "scripts.run_gaia._PROJECT_ROOT", tmp_path
        ):
            result_path = save_results(summary)

        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["total_tasks"] == 1
        assert data["accuracy"] == 1.0
