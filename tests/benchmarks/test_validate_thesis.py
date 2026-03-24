"""Tests for thesis validation benchmark: 4-configuration comparison.

All provider calls are mocked to avoid real API usage. Verifies:
  - 4-tier comparison logic with mocked providers
  - Quality scoring heuristic produces reasonable 1-5 scores
  - Cost/token tracking per configuration
  - Summary table computation (averages, reduction %)
  - JSON output format matches expected schema
  - --dry-run mode produces valid output
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.types import CompletionResult, ExecutionResult
from models.base import ModelProvider

from scripts.validate_thesis import (
    CONFIGURATIONS,
    GPT_4O,
    LLAMA_8B,
    LLAMA_70B,
    TASKS,
    ConfigSummary,
    MockProvider,
    RunResult,
    compute_config_summaries,
    format_summary_table,
    results_to_json,
    run_all_configurations,
    run_direct_call,
    run_single_task,
    score_quality,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubProvider(ModelProvider):
    """Configurable mock provider for thesis validation tests."""

    def __init__(
        self,
        default_result: CompletionResult | None = None,
        model_results: dict[str, CompletionResult] | None = None,
    ) -> None:
        self._default: CompletionResult = default_result or CompletionResult(
            content="Paris is the capital of France.",
            model="stub-model",
            tokens_in=20,
            tokens_out=10,
            latency_ms=100.0,
            cost=0.001,
        )
        self._model_results: dict[str, CompletionResult] = model_results or {}
        self.call_count: int = 0
        self.captured_models: list[str] = []

    @property
    def name(self) -> str:
        return "stub"

    async def complete(
        self, messages: list[dict[str, str]], model: str, **kwargs: object
    ) -> CompletionResult:
        self.call_count += 1
        self.captured_models.append(model)
        return self._model_results.get(model, self._default)


def _make_run_result(
    config_id: str = "llama8b_direct",
    config_label: str = "Llama 8B Direct",
    model: str = LLAMA_8B,
    mode: str = "direct",
    task_id: str = "qa_01",
    task_category: str = "Simple Q&A",
    task_description: str = "What is the capital of France?",
    output: str = "Paris is the capital of France.",
    quality: int = 5,
    tokens: int = 160,
    cost: float = 0.0,
    latency_ms: float = 50.0,
    success: bool = True,
    keyword_hits: int = 1,
    keyword_total: int = 1,
) -> RunResult:
    """Build a RunResult with sensible defaults."""
    return RunResult(
        config_id=config_id,
        config_label=config_label,
        model=model,
        mode=mode,
        task_id=task_id,
        task_category=task_category,
        task_description=task_description,
        output=output,
        quality=quality,
        tokens=tokens,
        cost=cost,
        latency_ms=latency_ms,
        success=success,
        keyword_hits=keyword_hits,
        keyword_total=keyword_total,
    )


def _make_mock_orchestrator(
    output: str = "Pipeline output with Paris",
    success: bool = True,
    total_tokens: int = 30,
    total_cost: float = 0.002,
) -> MagicMock:
    """Create a mocked Orchestrator that returns a fixed ExecutionResult."""
    mock_orch: MagicMock = MagicMock()
    mock_orch.process = AsyncMock(return_value=ExecutionResult(
        root_id="test-root",
        output=output,
        success=success,
        total_tokens=total_tokens,
        total_cost=total_cost,
    ))
    return mock_orch


# ---------------------------------------------------------------------------
# Tests: score_quality heuristic
# ---------------------------------------------------------------------------


class TestScoreQuality:
    """Verify quality scoring heuristic produces reasonable 1-5 scores."""

    def test_empty_output_returns_1(self) -> None:
        assert score_quality("", ["keyword"]) == 1

    def test_whitespace_only_returns_1(self) -> None:
        assert score_quality("   \n  ", ["keyword"]) == 1

    def test_no_keywords_long_output_returns_5(self) -> None:
        assert score_quality("x" * 150, []) == 5

    def test_no_keywords_medium_output_returns_4(self) -> None:
        assert score_quality("x" * 50, []) == 4

    def test_no_keywords_short_output_returns_3(self) -> None:
        assert score_quality("x" * 10, []) == 3

    def test_all_keywords_matched_returns_5(self) -> None:
        assert score_quality("Paris France", ["Paris", "France"]) == 5

    def test_over_half_keywords_returns_4(self) -> None:
        output = "Paris France"
        keywords = ["Paris", "France", "Europe", "capital"]
        assert score_quality(output, keywords) == 4

    def test_under_half_keywords_returns_3(self) -> None:
        output = "Paris is nice"
        keywords = ["Paris", "France", "Europe", "capital"]
        assert score_quality(output, keywords) == 3

    def test_no_keywords_matched_returns_2(self) -> None:
        assert score_quality("unrelated text", ["Paris", "France"]) == 2

    def test_case_insensitive(self) -> None:
        assert score_quality("paris france", ["Paris", "France"]) == 5

    def test_scores_always_in_1_to_5(self) -> None:
        """Exhaustively test that all scores are within the 1-5 range."""
        test_cases: list[tuple[str, list[str]]] = [
            ("", ["x"]),
            ("  ", []),
            ("short", []),
            ("a" * 50, []),
            ("a" * 200, []),
            ("match", ["match"]),
            ("no match", ["xyz"]),
            ("partial match here", ["partial", "missing"]),
        ]
        for output, keywords in test_cases:
            score = score_quality(output, keywords)
            assert 1 <= score <= 5, (
                f"score_quality({output!r}, {keywords!r}) = {score}, "
                f"expected 1-5"
            )

    def test_single_keyword_match_returns_5(self) -> None:
        assert score_quality("The answer is 1945", ["1945"]) == 5

    def test_boundary_length_30_returns_4(self) -> None:
        output = "x" * 30
        assert score_quality(output, []) == 4

    def test_boundary_length_100_returns_5(self) -> None:
        output = "x" * 100
        assert score_quality(output, []) == 5


# ---------------------------------------------------------------------------
# Tests: run_direct_call
# ---------------------------------------------------------------------------


class TestRunDirectCall:
    """Test the direct single-call execution path."""

    @pytest.mark.asyncio
    async def test_returns_completion_result(self) -> None:
        provider = StubProvider()
        result = await run_direct_call(provider, "test-model", "Hello")
        assert isinstance(result, CompletionResult)
        assert result.content == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_passes_correct_model(self) -> None:
        provider = StubProvider()
        await run_direct_call(provider, "my-model", "prompt")
        assert provider.captured_models == ["my-model"]


# ---------------------------------------------------------------------------
# Tests: run_single_task (direct + pipeline modes)
# ---------------------------------------------------------------------------


class TestRunSingleTask:
    """Test per-task execution with mocked providers."""

    @pytest.mark.asyncio
    async def test_direct_mode_captures_metrics(self) -> None:
        result_cr = CompletionResult(
            content="The answer is 1945",
            model=LLAMA_8B,
            tokens_in=15,
            tokens_out=8,
            latency_ms=42.0,
            cost=0.0,
        )
        provider = StubProvider(default_result=result_cr)
        orchestrator = _make_mock_orchestrator()

        task: dict[str, Any] = {
            "id": "qa_05",
            "category": "Simple Q&A",
            "description": "What year did WW2 end?",
            "expected_keywords": ["1945"],
        }
        config: dict[str, str] = {
            "id": "llama8b_direct",
            "label": "Llama 8B Direct",
            "model": LLAMA_8B,
            "mode": "direct",
        }

        result = await run_single_task(task, config, provider, orchestrator)

        assert result.config_id == "llama8b_direct"
        assert result.mode == "direct"
        assert result.tokens == 23  # 15 + 8
        assert result.cost == 0.0
        assert result.success is True
        assert result.quality == 5  # "1945" found
        assert result.keyword_hits == 1
        assert result.keyword_total == 1

    @pytest.mark.asyncio
    async def test_pipeline_mode_captures_metrics(self) -> None:
        provider = StubProvider()
        orchestrator = _make_mock_orchestrator(
            output="The answer is 1945",
            total_tokens=45,
            total_cost=0.003,
        )

        task: dict[str, Any] = {
            "id": "qa_05",
            "category": "Simple Q&A",
            "description": "What year did WW2 end?",
            "expected_keywords": ["1945"],
        }
        config: dict[str, str] = {
            "id": "llama8b_pipeline",
            "label": "Llama 8B Pipeline",
            "model": LLAMA_8B,
            "mode": "pipeline",
        }

        result = await run_single_task(task, config, provider, orchestrator)

        assert result.config_id == "llama8b_pipeline"
        assert result.mode == "pipeline"
        assert result.tokens == 45
        assert result.cost == 0.003
        assert result.success is True
        assert result.quality == 5

    @pytest.mark.asyncio
    async def test_direct_mode_handles_provider_error(self) -> None:
        provider = StubProvider()
        provider.complete = AsyncMock(side_effect=Exception("API error"))  # type: ignore[assignment]
        orchestrator = _make_mock_orchestrator()

        task: dict[str, Any] = {
            "id": "fail_01",
            "category": "Test",
            "description": "This will fail",
            "expected_keywords": [],
        }
        config: dict[str, str] = {
            "id": "llama8b_direct",
            "label": "Llama 8B Direct",
            "model": LLAMA_8B,
            "mode": "direct",
        }

        result = await run_single_task(task, config, provider, orchestrator)
        assert result.success is False
        assert result.quality == 1
        assert result.tokens == 0
        assert result.cost == 0.0

    @pytest.mark.asyncio
    async def test_pipeline_mode_handles_orchestrator_error(self) -> None:
        provider = StubProvider()
        orchestrator = MagicMock()
        orchestrator.process = AsyncMock(side_effect=Exception("Pipeline crashed"))

        task: dict[str, Any] = {
            "id": "fail_02",
            "category": "Test",
            "description": "Pipeline fails",
            "expected_keywords": [],
        }
        config: dict[str, str] = {
            "id": "llama8b_pipeline",
            "label": "Llama 8B Pipeline",
            "model": LLAMA_8B,
            "mode": "pipeline",
        }

        result = await run_single_task(task, config, provider, orchestrator)
        assert result.success is False
        assert result.quality == 1
        assert result.tokens == 0


# ---------------------------------------------------------------------------
# Tests: 4-configuration comparison logic
# ---------------------------------------------------------------------------


class TestFourConfigComparison:
    """Test the 4-tier comparison by running all configurations with mocks."""

    @pytest.mark.asyncio
    async def test_runs_all_4_configurations(self) -> None:
        """Running 2 tasks across 4 configs should produce 8 results."""
        provider = StubProvider()
        orchestrator = _make_mock_orchestrator()
        tasks: list[dict[str, Any]] = TASKS[:2]

        results = await run_all_configurations(
            tasks, CONFIGURATIONS, provider, orchestrator,
        )

        assert len(results) == 8  # 4 configs * 2 tasks
        config_ids = {r.config_id for r in results}
        assert config_ids == {
            "llama8b_direct",
            "llama8b_pipeline",
            "llama70b_direct",
            "gpt4o_direct",
        }

    @pytest.mark.asyncio
    async def test_each_config_gets_correct_model(self) -> None:
        """Each configuration should use its designated model."""
        provider = StubProvider()
        orchestrator = _make_mock_orchestrator()
        tasks: list[dict[str, Any]] = TASKS[:1]

        results = await run_all_configurations(
            tasks, CONFIGURATIONS, provider, orchestrator,
        )

        model_by_config: dict[str, str] = {r.config_id: r.model for r in results}
        assert model_by_config["llama8b_direct"] == LLAMA_8B
        assert model_by_config["llama8b_pipeline"] == LLAMA_8B
        assert model_by_config["llama70b_direct"] == LLAMA_70B
        assert model_by_config["gpt4o_direct"] == GPT_4O

    @pytest.mark.asyncio
    async def test_modes_assigned_correctly(self) -> None:
        """Direct configs use 'direct' mode, pipeline uses 'pipeline'."""
        provider = StubProvider()
        orchestrator = _make_mock_orchestrator()
        tasks: list[dict[str, Any]] = TASKS[:1]

        results = await run_all_configurations(
            tasks, CONFIGURATIONS, provider, orchestrator,
        )

        mode_by_config: dict[str, str] = {r.config_id: r.mode for r in results}
        assert mode_by_config["llama8b_direct"] == "direct"
        assert mode_by_config["llama8b_pipeline"] == "pipeline"
        assert mode_by_config["llama70b_direct"] == "direct"
        assert mode_by_config["gpt4o_direct"] == "direct"


# ---------------------------------------------------------------------------
# Tests: cost/token tracking per configuration
# ---------------------------------------------------------------------------


class TestCostTokenTracking:
    """Verify cost and token tracking per configuration."""

    @pytest.mark.asyncio
    async def test_direct_mode_sums_tokens_in_and_out(self) -> None:
        cr = CompletionResult(
            content="output",
            model=LLAMA_8B,
            tokens_in=25,
            tokens_out=15,
            latency_ms=10.0,
            cost=0.0005,
        )
        provider = StubProvider(default_result=cr)
        orchestrator = _make_mock_orchestrator()

        task: dict[str, Any] = {
            "id": "qa_01",
            "category": "Q&A",
            "description": "test",
            "expected_keywords": [],
        }
        config: dict[str, str] = {
            "id": "llama8b_direct",
            "label": "Llama 8B Direct",
            "model": LLAMA_8B,
            "mode": "direct",
        }

        result = await run_single_task(task, config, provider, orchestrator)
        assert result.tokens == 40  # 25 + 15
        assert result.cost == 0.0005

    @pytest.mark.asyncio
    async def test_pipeline_mode_uses_execution_result_tokens(self) -> None:
        provider = StubProvider()
        orchestrator = _make_mock_orchestrator(
            total_tokens=88,
            total_cost=0.0042,
        )

        task: dict[str, Any] = {
            "id": "qa_01",
            "category": "Q&A",
            "description": "test",
            "expected_keywords": [],
        }
        config: dict[str, str] = {
            "id": "llama8b_pipeline",
            "label": "Llama 8B Pipeline",
            "model": LLAMA_8B,
            "mode": "pipeline",
        }

        result = await run_single_task(task, config, provider, orchestrator)
        assert result.tokens == 88
        assert result.cost == 0.0042

    def test_per_config_cost_differs_in_mock_provider(self) -> None:
        """MockProvider assigns different costs per model tier."""
        mock = MockProvider()
        # Verify the cost map is correctly defined
        import asyncio

        async def _check() -> None:
            r_8b = await mock.complete(
                [{"role": "user", "content": "test"}], LLAMA_8B,
            )
            r_70b = await mock.complete(
                [{"role": "user", "content": "test"}], LLAMA_70B,
            )
            r_gpt = await mock.complete(
                [{"role": "user", "content": "test"}], GPT_4O,
            )
            assert r_8b.cost == 0.0
            assert r_70b.cost == 0.000035
            assert r_gpt.cost == 0.000250

        asyncio.run(_check())

    def test_per_config_tokens_differ_in_mock_provider(self) -> None:
        """MockProvider assigns different token counts per model tier."""
        mock = MockProvider()
        import asyncio

        async def _check() -> None:
            r_8b = await mock.complete(
                [{"role": "user", "content": "test"}], LLAMA_8B,
            )
            r_70b = await mock.complete(
                [{"role": "user", "content": "test"}], LLAMA_70B,
            )
            r_gpt = await mock.complete(
                [{"role": "user", "content": "test"}], GPT_4O,
            )
            assert r_8b.tokens_out == 150
            assert r_70b.tokens_out == 120
            assert r_gpt.tokens_out == 100

        asyncio.run(_check())


# ---------------------------------------------------------------------------
# Tests: compute_config_summaries (averages, reduction %)
# ---------------------------------------------------------------------------


class TestComputeConfigSummaries:
    """Verify summary table computation: averages and reduction percentages."""

    def test_single_config_averages(self) -> None:
        results = [
            _make_run_result(config_id="llama8b_direct", tokens=100, cost=0.001, quality=3),
            _make_run_result(config_id="llama8b_direct", tokens=200, cost=0.003, quality=5),
        ]
        summaries = compute_config_summaries(results)

        s = summaries["llama8b_direct"]
        assert s.avg_tokens == 150.0
        assert s.avg_cost == 0.002
        assert s.avg_quality == 4.0
        assert s.total_tasks == 2

    def test_all_four_configs_present(self) -> None:
        results = [
            _make_run_result(config_id="llama8b_direct", config_label="Llama 8B Direct"),
            _make_run_result(config_id="llama8b_pipeline", config_label="Llama 8B Pipeline", mode="pipeline"),
            _make_run_result(config_id="llama70b_direct", config_label="Llama 70B Direct", model=LLAMA_70B),
            _make_run_result(config_id="gpt4o_direct", config_label="GPT-4o Direct", model=GPT_4O),
        ]
        summaries = compute_config_summaries(results)

        assert len(summaries) == 4
        assert "llama8b_direct" in summaries
        assert "llama8b_pipeline" in summaries
        assert "llama70b_direct" in summaries
        assert "gpt4o_direct" in summaries

    def test_success_rate_computed_correctly(self) -> None:
        results = [
            _make_run_result(config_id="llama8b_direct", success=True),
            _make_run_result(config_id="llama8b_direct", success=True),
            _make_run_result(config_id="llama8b_direct", success=False),
        ]
        summaries = compute_config_summaries(results)
        assert abs(summaries["llama8b_direct"].success_rate - 2.0 / 3.0) < 0.01

    def test_token_reduction_vs_gpt4o(self) -> None:
        """Llama 8B direct with 50 tokens vs GPT-4o with 200 tokens = 75% reduction."""
        results = [
            _make_run_result(config_id="llama8b_direct", tokens=50),
            _make_run_result(config_id="gpt4o_direct", config_label="GPT-4o Direct", tokens=200),
        ]
        summaries = compute_config_summaries(results)

        assert abs(summaries["llama8b_direct"].token_reduction_vs_gpt4o_pct - 75.0) < 0.1

    def test_gpt4o_token_reduction_is_zero(self) -> None:
        results = [
            _make_run_result(config_id="gpt4o_direct", config_label="GPT-4o Direct", tokens=200),
        ]
        summaries = compute_config_summaries(results)
        assert summaries["gpt4o_direct"].token_reduction_vs_gpt4o_pct == 0.0

    def test_negative_reduction_when_config_uses_more_tokens(self) -> None:
        """If a config uses more tokens than GPT-4o, reduction is negative."""
        results = [
            _make_run_result(config_id="llama8b_direct", tokens=300),
            _make_run_result(config_id="gpt4o_direct", config_label="GPT-4o Direct", tokens=200),
        ]
        summaries = compute_config_summaries(results)
        # (200 - 300) / 200 * 100 = -50%
        assert abs(summaries["llama8b_direct"].token_reduction_vs_gpt4o_pct - (-50.0)) < 0.1

    def test_avg_latency_computed(self) -> None:
        results = [
            _make_run_result(config_id="llama8b_direct", latency_ms=100.0),
            _make_run_result(config_id="llama8b_direct", latency_ms=200.0),
        ]
        summaries = compute_config_summaries(results)
        assert summaries["llama8b_direct"].avg_latency_ms == 150.0


# ---------------------------------------------------------------------------
# Tests: format_summary_table
# ---------------------------------------------------------------------------


class TestFormatSummaryTable:
    """Verify summary table formatting."""

    def _make_summaries(self) -> dict[str, ConfigSummary]:
        return {
            "llama8b_direct": ConfigSummary(
                config_id="llama8b_direct", config_label="Llama 8B Direct",
                avg_quality=3.5, avg_tokens=160, avg_cost=0.0,
                avg_latency_ms=50, total_tasks=30, success_rate=1.0,
                token_reduction_vs_gpt4o_pct=37.5,
            ),
            "llama8b_pipeline": ConfigSummary(
                config_id="llama8b_pipeline", config_label="Llama 8B Pipeline",
                avg_quality=4.2, avg_tokens=140, avg_cost=0.0,
                avg_latency_ms=120, total_tasks=30, success_rate=0.97,
                token_reduction_vs_gpt4o_pct=45.0,
            ),
            "llama70b_direct": ConfigSummary(
                config_id="llama70b_direct", config_label="Llama 70B Direct",
                avg_quality=4.0, avg_tokens=130, avg_cost=0.000035,
                avg_latency_ms=80, total_tasks=30, success_rate=1.0,
                token_reduction_vs_gpt4o_pct=22.5,
            ),
            "gpt4o_direct": ConfigSummary(
                config_id="gpt4o_direct", config_label="GPT-4o Direct",
                avg_quality=4.8, avg_tokens=110, avg_cost=0.00025,
                avg_latency_ms=200, total_tasks=30, success_rate=1.0,
                token_reduction_vs_gpt4o_pct=0.0,
            ),
        }

    def test_contains_all_config_labels(self) -> None:
        summaries = self._make_summaries()
        table = format_summary_table(summaries)
        assert "Llama 8B Direct" in table
        assert "Llama 8B Pipeline" in table
        assert "Llama 70B Direct" in table
        assert "GPT-4o Direct" in table

    def test_contains_thesis_metrics_section(self) -> None:
        summaries = self._make_summaries()
        table = format_summary_table(summaries)
        assert "KEY THESIS METRICS" in table

    def test_contains_quality_lift(self) -> None:
        summaries = self._make_summaries()
        table = format_summary_table(summaries)
        assert "Quality lift" in table

    def test_contains_quality_gap(self) -> None:
        summaries = self._make_summaries()
        table = format_summary_table(summaries)
        assert "Quality gap" in table

    def test_contains_cost_ratio(self) -> None:
        summaries = self._make_summaries()
        table = format_summary_table(summaries)
        assert "Cost ratio" in table

    def test_empty_summaries_does_not_crash(self) -> None:
        table = format_summary_table({})
        assert "Thesis Validation Benchmark" in table


# ---------------------------------------------------------------------------
# Tests: JSON output format
# ---------------------------------------------------------------------------


class TestResultsToJson:
    """Verify JSON output format matches expected schema."""

    def _build_json_data(self) -> dict[str, Any]:
        results = [
            _make_run_result(config_id="llama8b_direct"),
            _make_run_result(config_id="gpt4o_direct", config_label="GPT-4o Direct"),
        ]
        summaries = compute_config_summaries(results)
        return results_to_json(results, summaries)

    def test_top_level_keys(self) -> None:
        data = self._build_json_data()
        required_keys = {
            "timestamp",
            "task_count",
            "configuration_count",
            "configurations",
            "results",
            "summaries",
        }
        assert required_keys.issubset(set(data.keys()))

    def test_task_count_matches(self) -> None:
        data = self._build_json_data()
        assert data["task_count"] == len(TASKS)

    def test_configuration_count_matches(self) -> None:
        data = self._build_json_data()
        assert data["configuration_count"] == len(CONFIGURATIONS)

    def test_results_are_list(self) -> None:
        data = self._build_json_data()
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2

    def test_result_entry_fields(self) -> None:
        data = self._build_json_data()
        entry = data["results"][0]
        required_fields = {
            "config_id",
            "config_label",
            "model",
            "mode",
            "task_id",
            "task_category",
            "quality",
            "tokens",
            "cost",
            "latency_ms",
            "success",
            "keyword_hits",
            "keyword_total",
        }
        assert required_fields.issubset(set(entry.keys()))

    def test_summaries_are_dict(self) -> None:
        data = self._build_json_data()
        assert isinstance(data["summaries"], dict)

    def test_summary_entry_fields(self) -> None:
        data = self._build_json_data()
        for config_id, summary in data["summaries"].items():
            required_fields = {
                "config_label",
                "avg_quality",
                "avg_tokens",
                "avg_cost",
                "avg_latency_ms",
                "total_tasks",
                "success_rate",
                "token_reduction_vs_gpt4o_pct",
            }
            missing = required_fields - set(summary.keys())
            assert not missing, (
                f"Summary for {config_id} missing fields: {missing}"
            )

    def test_json_serializable(self) -> None:
        data = self._build_json_data()
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert parsed["task_count"] == data["task_count"]

    def test_cost_is_rounded(self) -> None:
        data = self._build_json_data()
        for entry in data["results"]:
            # cost should be a float (rounded in results_to_json)
            assert isinstance(entry["cost"], (int, float))

    def test_quality_is_int_in_range(self) -> None:
        data = self._build_json_data()
        for entry in data["results"]:
            assert isinstance(entry["quality"], int)
            assert 1 <= entry["quality"] <= 5


# ---------------------------------------------------------------------------
# Tests: MockProvider (--dry-run mode)
# ---------------------------------------------------------------------------


class TestMockProvider:
    """Verify the built-in MockProvider for --dry-run mode produces valid output."""

    @pytest.mark.asyncio
    async def test_returns_completion_result(self) -> None:
        mock = MockProvider()
        result = await mock.complete(
            [{"role": "user", "content": "Hello"}], LLAMA_8B,
        )
        assert isinstance(result, CompletionResult)

    @pytest.mark.asyncio
    async def test_name_is_mock(self) -> None:
        mock = MockProvider()
        assert mock.name == "mock"

    @pytest.mark.asyncio
    async def test_output_is_nonempty(self) -> None:
        mock = MockProvider()
        result = await mock.complete(
            [{"role": "user", "content": "test"}], LLAMA_8B,
        )
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_tokens_vary_by_model(self) -> None:
        mock = MockProvider()
        r_8b = await mock.complete(
            [{"role": "user", "content": "test"}], LLAMA_8B,
        )
        r_70b = await mock.complete(
            [{"role": "user", "content": "test"}], LLAMA_70B,
        )
        r_gpt = await mock.complete(
            [{"role": "user", "content": "test"}], GPT_4O,
        )
        # Each model tier should have different token counts
        assert r_8b.tokens_out != r_70b.tokens_out
        assert r_70b.tokens_out != r_gpt.tokens_out

    @pytest.mark.asyncio
    async def test_cost_varies_by_model(self) -> None:
        mock = MockProvider()
        r_8b = await mock.complete(
            [{"role": "user", "content": "test"}], LLAMA_8B,
        )
        r_gpt = await mock.complete(
            [{"role": "user", "content": "test"}], GPT_4O,
        )
        assert r_8b.cost < r_gpt.cost

    @pytest.mark.asyncio
    async def test_mock_contains_common_keywords(self) -> None:
        """Mock output should contain keywords that many tasks check for."""
        mock = MockProvider()
        result = await mock.complete(
            [{"role": "user", "content": "What is the capital of France?"}],
            LLAMA_8B,
        )
        output_lower = result.content.lower()
        assert "paris" in output_lower
        assert "leonardo" in output_lower or "vinci" in output_lower

    @pytest.mark.asyncio
    async def test_latency_is_fixed(self) -> None:
        mock = MockProvider()
        result = await mock.complete(
            [{"role": "user", "content": "test"}], LLAMA_8B,
        )
        assert result.latency_ms == 50.0


# ---------------------------------------------------------------------------
# Tests: dry-run integration (MockProvider through full pipeline)
# ---------------------------------------------------------------------------


class TestDryRunIntegration:
    """Test that --dry-run mode produces valid, complete output."""

    @pytest.mark.asyncio
    async def test_dry_run_produces_results_for_all_configs(self) -> None:
        """Dry run with 2 tasks should produce 8 results (4 configs * 2 tasks)."""
        provider = MockProvider()
        orchestrator = _make_mock_orchestrator()
        tasks: list[dict[str, Any]] = TASKS[:2]

        results = await run_all_configurations(
            tasks, CONFIGURATIONS, provider, orchestrator,
        )

        assert len(results) == 8

    @pytest.mark.asyncio
    async def test_dry_run_all_results_succeed(self) -> None:
        """With mock provider, all results should succeed."""
        provider = MockProvider()
        orchestrator = _make_mock_orchestrator()
        tasks: list[dict[str, Any]] = TASKS[:3]

        results = await run_all_configurations(
            tasks, CONFIGURATIONS, provider, orchestrator,
        )

        for r in results:
            assert r.success is True, (
                f"Result for {r.config_id}/{r.task_id} failed unexpectedly"
            )

    @pytest.mark.asyncio
    async def test_dry_run_json_output_valid(self) -> None:
        """Full dry-run flow should produce valid JSON output."""
        provider = MockProvider()
        orchestrator = _make_mock_orchestrator()
        tasks: list[dict[str, Any]] = TASKS[:2]

        results = await run_all_configurations(
            tasks, CONFIGURATIONS, provider, orchestrator,
        )
        summaries = compute_config_summaries(results)
        json_data = results_to_json(results, summaries)

        # Must be JSON-serializable
        serialized = json.dumps(json_data)
        parsed = json.loads(serialized)

        assert "results" in parsed
        assert "summaries" in parsed
        assert len(parsed["results"]) == 8

    @pytest.mark.asyncio
    async def test_dry_run_quality_scores_reasonable(self) -> None:
        """Mock provider output should produce quality scores >= 1, with most >= 3."""
        provider = MockProvider()
        orchestrator = _make_mock_orchestrator(
            output=(
                "Paris is the capital of France. Leonardo da Vinci. "
                "1945. Python Rust TCP UDP QUIC. Functional programming."
            ),
        )
        tasks: list[dict[str, Any]] = TASKS[:5]  # Simple Q&A tasks

        results = await run_all_configurations(
            tasks, CONFIGURATIONS, provider, orchestrator,
        )

        direct_results = [r for r in results if r.mode == "direct"]
        # All scores must be in valid range
        for r in direct_results:
            assert 1 <= r.quality <= 5, (
                f"Direct result for {r.config_id}/{r.task_id} has quality "
                f"({r.quality}) outside valid 1-5 range"
            )
        # Majority of scores should be >= 3 (mock output covers most keywords)
        high_quality_count = sum(1 for r in direct_results if r.quality >= 3)
        assert high_quality_count >= len(direct_results) * 0.6, (
            f"Expected >= 60% of direct results to have quality >= 3, "
            f"got {high_quality_count}/{len(direct_results)}"
        )


# ---------------------------------------------------------------------------
# Tests: TASKS and CONFIGURATIONS structure
# ---------------------------------------------------------------------------


class TestTasksStructure:
    """Validate the embedded task and configuration definitions."""

    def test_has_30_tasks(self) -> None:
        assert len(TASKS) == 30

    def test_unique_task_ids(self) -> None:
        ids = [t["id"] for t in TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs found"

    def test_each_task_has_required_fields(self) -> None:
        required = {"id", "description", "category", "expected_keywords"}
        for task in TASKS:
            missing = required - set(task.keys())
            assert not missing, (
                f"Task {task.get('id', '???')} missing fields: {missing}"
            )

    def test_diverse_categories(self) -> None:
        categories = {t["category"] for t in TASKS}
        assert len(categories) >= 10, (
            f"Expected 10+ categories, got {len(categories)}: {categories}"
        )

    def test_has_4_configurations(self) -> None:
        assert len(CONFIGURATIONS) == 4

    def test_configuration_ids(self) -> None:
        ids = {c["id"] for c in CONFIGURATIONS}
        assert ids == {
            "llama8b_direct",
            "llama8b_pipeline",
            "llama70b_direct",
            "gpt4o_direct",
        }

    def test_configurations_have_required_keys(self) -> None:
        required = {"id", "label", "model", "mode"}
        for config in CONFIGURATIONS:
            missing = required - set(config.keys())
            assert not missing, (
                f"Config {config.get('id', '???')} missing keys: {missing}"
            )
