"""Tests for the pattern cache warming script.

Uses mocked Orchestrator to verify:
- 30+ diverse tasks are defined across 5+ categories
- Cold run collects token stats
- Warm run shows reduced tokens (cache hits)
- Graph stats reporting works correctly
- Summary output includes all required metrics
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import ExecutionResult
from scripts.warm_cache import (
    TASK_CATEGORIES,
    RunStats,
    _all_tasks,
    _get_graph_stats,
    _print_summary,
    main,
)


# ---------------------------------------------------------------------------
# Task definition tests
# ---------------------------------------------------------------------------


class TestTaskDefinitions:
    """Verify task definitions meet the spec requirements."""

    def test_at_least_30_tasks(self) -> None:
        tasks = _all_tasks()
        assert len(tasks) >= 30, f"Expected 30+ tasks, got {len(tasks)}"

    def test_at_least_5_categories(self) -> None:
        assert len(TASK_CATEGORIES) >= 5, (
            f"Expected 5+ categories, got {len(TASK_CATEGORIES)}"
        )

    def test_at_least_6_tasks_per_category(self) -> None:
        for category, tasks in TASK_CATEGORIES.items():
            assert len(tasks) >= 6, (
                f"Category '{category}' has {len(tasks)} tasks, expected 6+"
            )

    def test_all_tasks_have_category(self) -> None:
        tasks = _all_tasks()
        for category, task in tasks:
            assert category in TASK_CATEGORIES
            assert len(task) > 10, f"Task too short: {task}"

    def test_tasks_are_diverse(self) -> None:
        """No two tasks should be identical."""
        tasks = _all_tasks()
        descriptions = [t[1] for t in tasks]
        assert len(set(descriptions)) == len(descriptions), "Duplicate tasks found"


# ---------------------------------------------------------------------------
# RunStats tests
# ---------------------------------------------------------------------------


class TestRunStats:
    """Verify RunStats tracking."""

    def test_record_success(self) -> None:
        stats = RunStats()
        result = ExecutionResult(
            root_id="r1",
            output="done",
            success=True,
            total_tokens=100,
            total_latency_ms=50.0,
            llm_calls=2,
        )
        stats.record("test task", result)

        assert stats.tasks_run == 1
        assert stats.successes == 1
        assert stats.failures == 0
        assert stats.total_tokens == 100
        assert stats.llm_calls == 2

    def test_record_failure(self) -> None:
        stats = RunStats()
        result = ExecutionResult(
            root_id="r2",
            output="error",
            success=False,
            total_tokens=50,
            total_latency_ms=20.0,
            llm_calls=1,
        )
        stats.record("failing task", result)

        assert stats.failures == 1
        assert stats.successes == 0

    def test_record_multiple(self) -> None:
        stats = RunStats()
        for i in range(5):
            result = ExecutionResult(
                root_id=f"r{i}",
                output="ok",
                success=True,
                total_tokens=100,
                total_latency_ms=10.0,
                llm_calls=2,
            )
            stats.record(f"task {i}", result)

        assert stats.tasks_run == 5
        assert stats.total_tokens == 500
        assert stats.llm_calls == 10
        assert len(stats.per_task) == 5


# ---------------------------------------------------------------------------
# Graph stats helper test
# ---------------------------------------------------------------------------


class TestGetGraphStats:
    """Verify graph stats retrieval."""

    def test_returns_counts_for_all_tables(self) -> None:
        mock_store = MagicMock()
        mock_store.query.return_value = [{"cnt": 5}]
        stats = _get_graph_stats(mock_store)

        assert "Task" in stats
        assert "PatternNode" in stats
        assert "ExecutionTree" in stats
        assert stats["Task"] == 5

    def test_handles_empty_graph(self) -> None:
        mock_store = MagicMock()
        mock_store.query.return_value = [{"cnt": 0}]
        stats = _get_graph_stats(mock_store)

        assert stats["Task"] == 0
        assert stats["PatternNode"] == 0

    def test_handles_query_exception(self) -> None:
        mock_store = MagicMock()
        mock_store.query.side_effect = Exception("no such table")
        stats = _get_graph_stats(mock_store)

        assert stats["Task"] == 0


# ---------------------------------------------------------------------------
# Print summary test (smoke test -- just verify it does not crash)
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Verify summary printing does not crash."""

    def test_print_summary_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        cold = RunStats(
            total_tokens=1000, llm_calls=20, tasks_run=10, successes=10,
        )
        warm = RunStats(
            total_tokens=600, llm_calls=12, tasks_run=10, successes=10,
        )
        pre = {"Task": 0, "PatternNode": 0, "ExecutionTree": 0}
        post_cold = {"Task": 10, "PatternNode": 8, "ExecutionTree": 10}
        post_warm = {"Task": 20, "PatternNode": 12, "ExecutionTree": 20}

        _print_summary(cold, warm, pre, post_cold, post_warm)
        captured = capsys.readouterr()

        assert "CACHE WARMING SUMMARY" in captured.out
        assert "COLD RUN" in captured.out
        assert "WARM RUN" in captured.out
        assert "TOKEN REDUCTION" in captured.out
        assert "40.0%" in captured.out
        assert "PASS" in captured.out

    def test_print_summary_zero_cold_tokens(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Handle edge case where cold tokens are zero."""
        cold = RunStats(total_tokens=0, llm_calls=0, tasks_run=0, successes=0)
        warm = RunStats(total_tokens=0, llm_calls=0, tasks_run=0, successes=0)
        pre = {"Task": 0, "PatternNode": 0, "ExecutionTree": 0}

        _print_summary(cold, warm, pre, pre, pre)
        captured = capsys.readouterr()
        assert "cannot compute reduction" in captured.out


# ---------------------------------------------------------------------------
# Integration test with mocked orchestrator
# ---------------------------------------------------------------------------


class TestMainWithMockedOrchestrator:
    """Test the main() function with a fully mocked orchestrator."""

    async def test_main_cold_and_warm_runs(self) -> None:
        """Verify main() runs cold and warm passes and returns summary."""
        call_count = 0

        async def mock_process(message: str) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            # Cold run: higher tokens; warm run: lower tokens
            tasks = _all_tasks()
            total_tasks = len(tasks)
            is_warm = call_count > total_tasks
            tokens = 50 if is_warm else 100
            llm_calls_val = 1 if is_warm else 3
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="result",
                success=True,
                total_tokens=tokens,
                total_latency_ms=10.0,
                llm_calls=llm_calls_val,
            )

        mock_store = MagicMock()
        mock_store.initialize = MagicMock()
        mock_store.close = MagicMock()
        # Return increasing pattern counts across calls
        pattern_call_count = 0

        def mock_query(cypher: str, params: dict | None = None) -> list[dict]:
            nonlocal pattern_call_count
            pattern_call_count += 1
            if "PatternNode" in cypher:
                # After cold run: some patterns; after warm: more
                return [{"cnt": min(pattern_call_count * 4, 15)}]
            elif "Task" in cypher:
                return [{"cnt": pattern_call_count * 5}]
            elif "ExecutionTree" in cypher:
                return [{"cnt": pattern_call_count * 5}]
            return [{"cnt": 0}]

        mock_store.query = mock_query

        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(side_effect=mock_process)

        with patch("scripts.warm_cache.GraphStore", return_value=mock_store), \
             patch("scripts.warm_cache.Orchestrator", return_value=mock_orchestrator), \
             patch("scripts.warm_cache._build_router", return_value=MagicMock()):
            summary = await main(db_path=None)

        total_tasks = len(_all_tasks())

        assert summary["total_tasks"] == total_tasks
        assert summary["cold_tokens"] == total_tasks * 100
        assert summary["warm_tokens"] == total_tasks * 50
        assert summary["reduction_pct"] == pytest.approx(50.0)
        assert summary["cold_llm_calls"] == total_tasks * 3
        assert summary["warm_llm_calls"] == total_tasks * 1

        # Verify orchestrator.process was called 2x for each task (cold + warm)
        assert mock_orchestrator.process.call_count == total_tasks * 2
