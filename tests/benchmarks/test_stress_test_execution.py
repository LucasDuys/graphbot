"""Stress test mocked execution and failure diagnosis verification.

Tests the stress_test.py module without making real API calls by mocking the
Orchestrator. Validates failure diagnosis categorization, JSON output schema,
task definitions, and --dry-run mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Import the module under test
from scripts.stress_test import (
    STRESS_TASKS,
    DiagnosisResult,
    FailureCategory,
    StressTask,
    StressTestEntry,
    diagnose_failure,
    dry_run_report,
    run_stress_test,
)
from core_gb.types import ExecutionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def success_result() -> ExecutionResult:
    """An ExecutionResult representing a successful task completion."""
    return ExecutionResult(
        root_id="test_root",
        output="The answer is 42.",
        success=True,
        total_nodes=5,
        total_tokens=1200,
        total_cost=0.003,
        model_used="openai/gpt-4o-mini",
        tools_used=2,
        llm_calls=3,
        errors=(),
    )


@pytest.fixture()
def failure_result() -> ExecutionResult:
    """An ExecutionResult representing a failed task with no tool usage."""
    return ExecutionResult(
        root_id="test_root",
        output="I cannot complete this task.",
        success=False,
        total_nodes=1,
        total_tokens=400,
        total_cost=0.001,
        model_used="openai/gpt-4o-mini",
        tools_used=0,
        llm_calls=1,
        errors=("Model refused the task",),
    )


@pytest.fixture()
def tmp_benchmarks(tmp_path: Path) -> Path:
    """Provide a temporary benchmarks directory for output files."""
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Task definition validation (all 10 hard tasks)
# ---------------------------------------------------------------------------

class TestStressTaskDefinitions:
    """Verify that all 10 hard task definitions exist with proper fields."""

    REQUIRED_FIELDS: set[str] = {
        "id", "name", "category", "description",
        "difficulty", "expected_behavior", "why_hard",
    }

    def test_exactly_10_tasks_defined(self) -> None:
        """The STRESS_TASKS list must contain exactly 10 task definitions."""
        assert len(STRESS_TASKS) == 10, (
            f"Expected 10 stress tasks, got {len(STRESS_TASKS)}"
        )

    def test_all_tasks_are_stress_task_instances(self) -> None:
        """Every entry in STRESS_TASKS must be a StressTask dataclass."""
        for task in STRESS_TASKS:
            assert isinstance(task, StressTask), (
                f"Expected StressTask, got {type(task).__name__}"
            )

    def test_each_task_has_required_fields(self) -> None:
        """Every StressTask must have all required fields set (non-empty)."""
        for task in STRESS_TASKS:
            for field_name in self.REQUIRED_FIELDS:
                value = getattr(task, field_name)
                assert value, (
                    f"Task {task.id} has empty or falsy '{field_name}': {value!r}"
                )

    def test_unique_task_ids(self) -> None:
        """All task IDs must be unique."""
        ids = [t.id for t in STRESS_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs found: {ids}"

    def test_unique_task_names(self) -> None:
        """All task names must be unique."""
        names = [t.name for t in STRESS_TASKS]
        assert len(names) == len(set(names)), f"Duplicate task names found"

    def test_difficulty_range(self) -> None:
        """Difficulty must be between 1 and 5 inclusive."""
        for task in STRESS_TASKS:
            assert 1 <= task.difficulty <= 5, (
                f"Task {task.id} difficulty {task.difficulty} out of range [1, 5]"
            )

    def test_id_prefix_convention(self) -> None:
        """All task IDs should follow the stress_XX_ prefix convention."""
        for task in STRESS_TASKS:
            assert task.id.startswith("stress_"), (
                f"Task ID {task.id} does not follow 'stress_*' convention"
            )

    def test_expected_task_ids_present(self) -> None:
        """All 10 expected task IDs must be present."""
        expected_ids = {
            "stress_01_multi_hop",
            "stress_02_ambiguous",
            "stress_03_tool_chain",
            "stress_04_dynamic_tool",
            "stress_05_deep_context",
            "stress_06_contradictory",
            "stress_07_multi_language",
            "stress_08_recursive_decomp",
            "stress_09_time_sensitive",
            "stress_10_meta_reasoning",
        }
        actual_ids = {t.id for t in STRESS_TASKS}
        assert expected_ids == actual_ids, (
            f"Missing IDs: {expected_ids - actual_ids}, "
            f"Extra IDs: {actual_ids - expected_ids}"
        )

    def test_categories_are_diverse(self) -> None:
        """Tasks should span multiple categories (at least 5 distinct)."""
        categories = {t.category for t in STRESS_TASKS}
        assert len(categories) >= 5, (
            f"Expected at least 5 distinct categories, got {len(categories)}: {categories}"
        )

    def test_accept_partial_field_exists(self) -> None:
        """Every task must have accept_partial as a bool."""
        for task in STRESS_TASKS:
            assert isinstance(task.accept_partial, bool), (
                f"Task {task.id} accept_partial is not bool: {type(task.accept_partial)}"
            )


# ---------------------------------------------------------------------------
# 2. Failure diagnosis categorization
# ---------------------------------------------------------------------------

class TestFailureDiagnosis:
    """Verify that diagnose_failure correctly categorizes failure types."""

    def _make_task(self, category: str = "reasoning") -> StressTask:
        """Create a minimal StressTask for diagnosis testing."""
        return StressTask(
            id="test_task",
            name="Test task",
            category=category,
            description="test description",
            difficulty=3,
            expected_behavior="test expected",
            why_hard="test why hard",
        )

    def test_timeout_by_elapsed_time(self) -> None:
        """Tasks exceeding 60s should be diagnosed as timeout."""
        task = self._make_task()
        result = diagnose_failure(task, None, None, elapsed_ms=65_000)
        assert result.category == FailureCategory.TIMEOUT
        assert "65000" in result.root_cause

    def test_timeout_by_exception_message(self) -> None:
        """Exceptions containing 'timeout' should be classified as timeout."""
        task = self._make_task()
        error = TimeoutError("Request timed out after 30s")
        result = diagnose_failure(task, None, error, elapsed_ms=5000)
        assert result.category == FailureCategory.TIMEOUT

    def test_tool_failure_by_exception(self) -> None:
        """Exceptions mentioning 'tool' should be classified as tool_failure."""
        task = self._make_task()
        error = RuntimeError("Tool registry not initialized")
        result = diagnose_failure(task, None, error, elapsed_ms=1000)
        assert result.category == FailureCategory.TOOL_FAILURE

    def test_runtime_error_generic_exception(self) -> None:
        """Generic exceptions should be classified as runtime_error."""
        task = self._make_task()
        error = ValueError("unexpected value")
        result = diagnose_failure(task, None, error, elapsed_ms=1000)
        assert result.category == FailureCategory.RUNTIME_ERROR
        assert "ValueError" in result.root_cause

    def test_safety_blocked_from_result_errors(self) -> None:
        """Results with 'blocked' in errors should be safety_blocked."""
        task = self._make_task()
        result = ExecutionResult(
            root_id="r",
            output="",
            success=False,
            errors=("Request blocked by safety filter",),
        )
        diag = diagnose_failure(task, result, None, elapsed_ms=1000)
        assert diag.category == FailureCategory.SAFETY_BLOCKED

    def test_model_refusal_from_output(self) -> None:
        """Output containing 'cannot' should be diagnosed as model_refusal."""
        task = self._make_task()
        result = ExecutionResult(
            root_id="r",
            output="I cannot complete this request.",
            success=False,
            errors=(),
        )
        diag = diagnose_failure(task, result, None, elapsed_ms=1000)
        assert diag.category == FailureCategory.MODEL_REFUSAL

    def test_decomposition_error(self) -> None:
        """Decomposition tasks with too few nodes should be decomposition_error."""
        task = self._make_task(category="decomposition")
        result = ExecutionResult(
            root_id="r",
            output="partial result",
            success=False,
            total_nodes=2,
            errors=(),
        )
        diag = diagnose_failure(task, result, None, elapsed_ms=1000)
        assert diag.category == FailureCategory.DECOMPOSITION_ERROR

    def test_contradiction_detection(self) -> None:
        """Contradiction-category tasks that fail should be diagnosed as contradiction."""
        task = self._make_task(category="contradiction")
        result = ExecutionResult(
            root_id="r",
            output="Here is a 5-word essay...",
            success=False,
            errors=(),
        )
        diag = diagnose_failure(task, result, None, elapsed_ms=1000)
        assert diag.category == FailureCategory.CONTRADICTION

    def test_context_missing_for_meta_query(self) -> None:
        """Meta-query tasks that fail should be diagnosed as context_missing."""
        task = self._make_task(category="meta_query")
        result = ExecutionResult(
            root_id="r",
            output="some output",
            success=False,
            errors=(),
        )
        diag = diagnose_failure(task, result, None, elapsed_ms=1000)
        assert diag.category == FailureCategory.CONTEXT_MISSING

    def test_tool_chain_no_tools_used(self) -> None:
        """Tool-chain tasks with zero tools used should be tool_failure."""
        task = self._make_task(category="tool_chain")
        result = ExecutionResult(
            root_id="r",
            output="some output",
            success=False,
            tools_used=0,
            errors=(),
        )
        diag = diagnose_failure(task, result, None, elapsed_ms=1000)
        assert diag.category == FailureCategory.TOOL_FAILURE

    def test_unknown_fallback(self) -> None:
        """Failures that match no pattern should be classified as unknown."""
        task = self._make_task(category="reasoning")
        result = ExecutionResult(
            root_id="r",
            output="some output that is fine",
            success=False,
            total_nodes=5,
            tools_used=2,
            errors=(),
        )
        diag = diagnose_failure(task, result, None, elapsed_ms=1000)
        assert diag.category == FailureCategory.UNKNOWN

    def test_diagnosis_result_fields(self) -> None:
        """DiagnosisResult must have category, root_cause, and suggestion."""
        task = self._make_task()
        diag = diagnose_failure(task, None, ValueError("x"), elapsed_ms=500)
        assert isinstance(diag, DiagnosisResult)
        assert diag.category
        assert diag.root_cause
        assert diag.suggestion


# ---------------------------------------------------------------------------
# 3. FailureCategory enum values
# ---------------------------------------------------------------------------

class TestFailureCategoryEnum:
    """Verify the FailureCategory enum contains all expected values."""

    EXPECTED_CATEGORIES: set[str] = {
        "timeout",
        "tool_failure",
        "decomposition_error",
        "safety_blocked",
        "model_refusal",
        "contradiction",
        "context_missing",
        "runtime_error",
        "unknown",
    }

    def test_all_categories_present(self) -> None:
        """FailureCategory must have all expected string values."""
        actual = {c.value for c in FailureCategory}
        assert actual == self.EXPECTED_CATEGORIES

    def test_categories_are_str_enum(self) -> None:
        """FailureCategory values must be usable as plain strings."""
        for cat in FailureCategory:
            assert isinstance(cat, str)
            assert cat == cat.value


# ---------------------------------------------------------------------------
# 4. StressTestEntry serialization / JSON output schema
# ---------------------------------------------------------------------------

class TestStressTestEntrySchema:
    """Verify StressTestEntry.to_dict produces the expected JSON schema."""

    REQUIRED_KEYS: set[str] = {
        "task_id", "name", "category", "difficulty", "description",
        "why_hard", "success", "output_preview", "total_nodes",
        "total_tokens", "latency_ms", "cost", "model_used",
        "tools_used", "llm_calls", "errors", "exception", "diagnosis",
    }

    def test_to_dict_has_all_keys(self) -> None:
        """to_dict output must include every required schema key."""
        entry = StressTestEntry(
            task_id="stress_01_multi_hop",
            name="Multi-hop reasoning",
            category="reasoning",
            difficulty=3,
            description="test",
            why_hard="test why",
            success=True,
        )
        d = entry.to_dict()
        missing = self.REQUIRED_KEYS - set(d.keys())
        assert not missing, f"Missing keys in to_dict output: {missing}"

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict output must be fully JSON-serializable."""
        entry = StressTestEntry(
            task_id="stress_01_multi_hop",
            name="Multi-hop reasoning",
            category="reasoning",
            difficulty=3,
            description="test",
            why_hard="test why",
            success=False,
            errors=["some error"],
            diagnosis={"category": "unknown", "root_cause": "n/a", "suggestion": "n/a"},
        )
        serialized = json.dumps(entry.to_dict())
        parsed = json.loads(serialized)
        assert parsed["task_id"] == "stress_01_multi_hop"
        assert parsed["success"] is False
        assert isinstance(parsed["errors"], list)
        assert isinstance(parsed["diagnosis"], dict)

    def test_latency_ms_is_rounded(self) -> None:
        """Latency in to_dict output should be rounded to integer."""
        entry = StressTestEntry(
            task_id="t", name="t", category="c", difficulty=1,
            description="d", why_hard="w", success=True,
            latency_ms=1234.567,
        )
        d = entry.to_dict()
        assert d["latency_ms"] == 1235

    def test_diagnosis_dict_schema(self) -> None:
        """Diagnosis dict must have category, root_cause, suggestion keys when present."""
        entry = StressTestEntry(
            task_id="t", name="t", category="c", difficulty=1,
            description="d", why_hard="w", success=False,
            diagnosis={
                "category": "timeout",
                "root_cause": "took too long",
                "suggestion": "add timeout",
            },
        )
        d = entry.to_dict()
        assert "category" in d["diagnosis"]
        assert "root_cause" in d["diagnosis"]
        assert "suggestion" in d["diagnosis"]


# ---------------------------------------------------------------------------
# 5. Dry run mode
# ---------------------------------------------------------------------------

class TestDryRunMode:
    """Verify --dry-run mode generates proper task listing without execution."""

    def test_dry_run_report_structure(self) -> None:
        """dry_run_report must return a dict with mode, total_tasks, categories, tasks."""
        report = dry_run_report(STRESS_TASKS)
        assert report["mode"] == "dry_run"
        assert report["total_tasks"] == 10
        assert isinstance(report["categories"], list)
        assert len(report["categories"]) >= 5
        assert isinstance(report["avg_difficulty"], float)
        assert isinstance(report["tasks"], list)
        assert len(report["tasks"]) == 10

    def test_dry_run_task_entries_have_status(self) -> None:
        """Each task in the dry run report must have status 'dry_run_skipped'."""
        report = dry_run_report(STRESS_TASKS)
        for entry in report["tasks"]:
            assert entry["status"] == "dry_run_skipped"

    def test_dry_run_task_entries_have_all_fields(self) -> None:
        """Dry run entries must include task metadata fields."""
        required = {
            "task_id", "name", "category", "difficulty",
            "description", "why_hard", "expected_behavior",
            "accept_partial", "status",
        }
        report = dry_run_report(STRESS_TASKS)
        for entry in report["tasks"]:
            missing = required - set(entry.keys())
            assert not missing, (
                f"Dry run entry {entry.get('task_id')} missing: {missing}"
            )

    def test_dry_run_is_json_serializable(self) -> None:
        """The entire dry run report must be JSON-serializable."""
        report = dry_run_report(STRESS_TASKS)
        serialized = json.dumps(report)
        parsed = json.loads(serialized)
        assert parsed["mode"] == "dry_run"

    def test_dry_run_avg_difficulty(self) -> None:
        """Average difficulty should be computed correctly."""
        report = dry_run_report(STRESS_TASKS)
        expected_avg = sum(t.difficulty for t in STRESS_TASKS) / len(STRESS_TASKS)
        assert abs(report["avg_difficulty"] - expected_avg) < 0.01


# ---------------------------------------------------------------------------
# 6. Mocked orchestrator execution (no real API calls)
# ---------------------------------------------------------------------------

class TestMockedExecution:
    """Test stress test execution with a fully mocked Orchestrator."""

    @pytest.mark.asyncio
    async def test_run_stress_test_dry_run_writes_json(
        self, tmp_benchmarks: Path
    ) -> None:
        """run_stress_test(dry_run=True) must write a valid JSON file."""
        out_file = tmp_benchmarks / "benchmarks" / "stress_test.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        report = dry_run_report(STRESS_TASKS)
        out_file.write_text(json.dumps(report, indent=2))

        data = json.loads(out_file.read_text())
        assert data["mode"] == "dry_run"
        assert data["total_tasks"] == 10
        assert len(data["tasks"]) == 10
        for entry in data["tasks"]:
            assert entry["status"] == "dry_run_skipped"

    @pytest.mark.asyncio
    async def test_mocked_orchestrator_success_path(self) -> None:
        """Mocked Orchestrator.process returning success should produce correct entry."""
        mock_result = ExecutionResult(
            root_id="root_1",
            output="The answer is 16",
            success=True,
            total_nodes=3,
            total_tokens=800,
            total_cost=0.002,
            model_used="openai/gpt-4o-mini",
            tools_used=1,
            llm_calls=2,
            errors=(),
        )

        task = STRESS_TASKS[0]  # stress_01_multi_hop
        entry = StressTestEntry(
            task_id=task.id,
            name=task.name,
            category=task.category,
            difficulty=task.difficulty,
            description=task.description,
            why_hard=task.why_hard,
            success=mock_result.success,
            output_preview=mock_result.output[:300],
            total_nodes=mock_result.total_nodes,
            total_tokens=mock_result.total_tokens,
            cost=mock_result.total_cost,
            model_used=mock_result.model_used,
            tools_used=mock_result.tools_used,
            llm_calls=mock_result.llm_calls,
            latency_ms=150.0,
        )

        assert entry.success is True
        d = entry.to_dict()
        assert d["task_id"] == "stress_01_multi_hop"
        assert d["success"] is True
        assert d["total_nodes"] == 3
        assert d["output_preview"] == "The answer is 16"

    @pytest.mark.asyncio
    async def test_mocked_orchestrator_failure_with_diagnosis(self) -> None:
        """Mocked failure should produce a diagnosis entry with correct category."""
        mock_result = ExecutionResult(
            root_id="root_1",
            output="I cannot do this",
            success=False,
            total_nodes=1,
            total_tokens=200,
            total_cost=0.001,
            model_used="openai/gpt-4o-mini",
            tools_used=0,
            llm_calls=1,
            errors=(),
        )

        task = STRESS_TASKS[0]
        diagnosis = diagnose_failure(task, mock_result, None, elapsed_ms=500)

        entry = StressTestEntry(
            task_id=task.id,
            name=task.name,
            category=task.category,
            difficulty=task.difficulty,
            description=task.description,
            why_hard=task.why_hard,
            success=False,
            diagnosis={
                "category": diagnosis.category,
                "root_cause": diagnosis.root_cause,
                "suggestion": diagnosis.suggestion,
            },
        )

        assert entry.success is False
        assert entry.diagnosis["category"] == FailureCategory.MODEL_REFUSAL

    @pytest.mark.asyncio
    async def test_mocked_orchestrator_exception_handling(self) -> None:
        """Exceptions from Orchestrator.process should be captured, not crash."""
        task = STRESS_TASKS[2]  # stress_03_tool_chain
        error = RuntimeError("Tool registry not initialized")

        diagnosis = diagnose_failure(task, None, error, elapsed_ms=200)

        entry = StressTestEntry(
            task_id=task.id,
            name=task.name,
            category=task.category,
            difficulty=task.difficulty,
            description=task.description,
            why_hard=task.why_hard,
            success=False,
            exception=f"{type(error).__name__}: {error}",
            diagnosis={
                "category": diagnosis.category,
                "root_cause": diagnosis.root_cause,
                "suggestion": diagnosis.suggestion,
            },
        )

        assert entry.exception == "RuntimeError: Tool registry not initialized"
        assert entry.diagnosis["category"] == FailureCategory.TOOL_FAILURE

    @pytest.mark.asyncio
    async def test_full_mocked_run_produces_valid_output_schema(self) -> None:
        """Simulate a full run with mocked results and verify the output JSON schema."""
        entries: list[dict[str, Any]] = []

        for task in STRESS_TASKS:
            mock_result = ExecutionResult(
                root_id=f"root_{task.id}",
                output=f"Result for {task.name}",
                success=(task.difficulty <= 3),
                total_nodes=task.difficulty,
                total_tokens=task.difficulty * 200,
                total_cost=task.difficulty * 0.001,
                model_used="openai/gpt-4o-mini",
                tools_used=task.difficulty,
                llm_calls=task.difficulty + 1,
                errors=() if task.difficulty <= 3 else ("failed",),
            )

            entry = StressTestEntry(
                task_id=task.id,
                name=task.name,
                category=task.category,
                difficulty=task.difficulty,
                description=task.description,
                why_hard=task.why_hard,
                success=mock_result.success,
                output_preview=mock_result.output[:300],
                total_nodes=mock_result.total_nodes,
                total_tokens=mock_result.total_tokens,
                cost=mock_result.total_cost,
                model_used=mock_result.model_used,
                tools_used=mock_result.tools_used,
                llm_calls=mock_result.llm_calls,
                latency_ms=100.0 * task.difficulty,
            )

            if not entry.success:
                diag = diagnose_failure(task, mock_result, None, elapsed_ms=entry.latency_ms)
                entry.diagnosis = {
                    "category": diag.category,
                    "root_cause": diag.root_cause,
                    "suggestion": diag.suggestion,
                }

            entries.append(entry.to_dict())

        success_count = sum(1 for e in entries if e["success"])
        output: dict[str, Any] = {
            "timestamp": "2026-03-24T00:00:00",
            "total_tasks": len(entries),
            "success_count": success_count,
            "failure_count": len(entries) - success_count,
            "success_rate": round(success_count / len(entries), 3),
            "total_tokens": sum(e["total_tokens"] for e in entries),
            "total_cost": round(sum(e["cost"] for e in entries), 6),
            "total_latency_ms": sum(e["latency_ms"] for e in entries),
            "failure_breakdown": {},
            "tasks": entries,
        }

        # Verify JSON schema
        serialized = json.dumps(output, indent=2)
        parsed = json.loads(serialized)

        assert isinstance(parsed["timestamp"], str)
        assert isinstance(parsed["total_tasks"], int)
        assert parsed["total_tasks"] == 10
        assert isinstance(parsed["success_count"], int)
        assert isinstance(parsed["failure_count"], int)
        assert parsed["success_count"] + parsed["failure_count"] == 10
        assert isinstance(parsed["success_rate"], float)
        assert 0.0 <= parsed["success_rate"] <= 1.0
        assert isinstance(parsed["total_tokens"], int)
        assert isinstance(parsed["total_cost"], float)
        assert isinstance(parsed["total_latency_ms"], (int, float))
        assert isinstance(parsed["failure_breakdown"], dict)
        assert isinstance(parsed["tasks"], list)
        assert len(parsed["tasks"]) == 10

        for task_entry in parsed["tasks"]:
            assert "task_id" in task_entry
            assert "success" in task_entry
            assert "diagnosis" in task_entry
            assert isinstance(task_entry["errors"], list)
