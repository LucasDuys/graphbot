"""Tests for the model tier comparison benchmark script.

All provider calls are mocked to avoid real API usage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import CompletionResult, ExecutionResult
from models.base import ModelProvider
from models.router import ModelRouter
from graph.store import GraphStore

# Import the benchmark module under test
from scripts.compare_models import (
    BENCHMARK_TASKS,
    MODEL_TIERS,
    ModelRunResult,
    TierSummary,
    compute_pipeline_token_reduction,
    compute_tier_summaries,
    format_summary_table,
    results_to_json,
    run_direct_call,
    run_task_benchmark,
    run_full_benchmark,
    score_quality,
)


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockBenchmarkProvider(ModelProvider):
    """Mock provider returning configurable CompletionResult per model."""

    def __init__(
        self,
        default_result: CompletionResult | None = None,
        model_results: dict[str, CompletionResult] | None = None,
    ) -> None:
        self._default = default_result or CompletionResult(
            content="Paris is the capital of France.",
            model="mock-model",
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
        return "mock-benchmark"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        self.call_count += 1
        self.captured_models.append(model)
        return self._model_results.get(model, self._default)


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _make_mock_result(
    content: str = "Test output",
    tokens_in: int = 15,
    tokens_out: int = 10,
    cost: float = 0.002,
    latency_ms: float = 80.0,
) -> CompletionResult:
    """Build a CompletionResult with sensible defaults."""
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        cost=cost,
    )


def _make_run_result(
    model: str = "mock/model",
    tier: str = "free",
    mode: str = "direct",
    task_id: str = "test_01",
    tokens: int = 100,
    cost: float = 0.001,
    quality: int = 4,
    latency_ms: float = 50.0,
    success: bool = True,
) -> ModelRunResult:
    """Build a ModelRunResult with sensible defaults."""
    return ModelRunResult(
        model=model,
        tier=tier,
        mode=mode,
        task_id=task_id,
        output="Test output",
        quality=quality,
        tokens=tokens,
        cost=cost,
        latency_ms=latency_ms,
        success=success,
        keyword_hits=1,
        keyword_total=2,
    )


# ---------------------------------------------------------------------------
# Tests: score_quality
# ---------------------------------------------------------------------------


class TestScoreQuality:
    """Test the placeholder quality scoring function."""

    def test_empty_output_scores_1(self) -> None:
        assert score_quality("", ["keyword"]) == 1

    def test_whitespace_only_scores_1(self) -> None:
        assert score_quality("   ", ["keyword"]) == 1

    def test_no_keywords_long_output_scores_5(self) -> None:
        output = "A" * 150
        assert score_quality(output, []) == 5

    def test_no_keywords_short_output_scores_4(self) -> None:
        output = "A" * 50
        assert score_quality(output, []) == 4

    def test_no_keywords_very_short_output_scores_3(self) -> None:
        output = "A" * 10
        assert score_quality(output, []) == 3

    def test_all_keywords_matched_scores_5(self) -> None:
        output = "Paris is the capital of France"
        assert score_quality(output, ["Paris", "France"]) == 5

    def test_some_keywords_matched_scores_4(self) -> None:
        output = "Paris is beautiful"
        assert score_quality(output, ["Paris", "France", "Europe", "city"]) == 3

    def test_half_keywords_matched_scores_4(self) -> None:
        output = "Paris and France are related"
        assert score_quality(output, ["Paris", "France", "Europe", "city"]) == 4

    def test_no_keywords_matched_scores_2(self) -> None:
        output = "Something completely unrelated"
        assert score_quality(output, ["Paris", "France"]) == 2

    def test_case_insensitive_matching(self) -> None:
        output = "paris is in france"
        assert score_quality(output, ["Paris", "France"]) == 5


# ---------------------------------------------------------------------------
# Tests: run_direct_call
# ---------------------------------------------------------------------------


class TestRunDirectCall:
    """Test the direct single-call function."""

    async def test_direct_call_returns_completion(self) -> None:
        provider = MockBenchmarkProvider()
        result = await run_direct_call(
            provider, "mock-model", "What is the capital of France?"
        )
        assert isinstance(result, CompletionResult)
        assert result.content == "Paris is the capital of France."
        assert provider.call_count == 1

    async def test_direct_call_passes_correct_model(self) -> None:
        provider = MockBenchmarkProvider()
        await run_direct_call(provider, "test-model-123", "Hello")
        assert provider.captured_models == ["test-model-123"]


# ---------------------------------------------------------------------------
# Tests: run_task_benchmark
# ---------------------------------------------------------------------------


class TestRunTaskBenchmark:
    """Test the per-task benchmark runner with mocked orchestrator."""

    async def test_returns_two_results(self) -> None:
        """Each task should produce exactly 2 results: direct + pipeline."""
        provider = MockBenchmarkProvider()
        router = ModelRouter(provider=provider)
        store = _make_store()

        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value=ExecutionResult(
            root_id="test-root",
            output="Pipeline output with Paris",
            success=True,
            total_tokens=30,
            total_cost=0.002,
        ))

        task = {
            "id": "test_01",
            "description": "What is the capital of France?",
            "expected_keywords": ["Paris"],
        }

        results = await run_task_benchmark(
            task, provider, mock_orchestrator, "free", "mock-model",
        )

        assert len(results) == 2
        assert results[0].mode == "direct"
        assert results[1].mode == "pipeline"
        store.close()

    async def test_direct_mode_captures_metrics(self) -> None:
        """Direct mode should capture tokens, cost, latency from provider."""
        result = _make_mock_result(
            content="9386",
            tokens_in=20,
            tokens_out=5,
            cost=0.0001,
        )
        provider = MockBenchmarkProvider(default_result=result)

        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value=ExecutionResult(
            root_id="r", output="", success=True,
        ))

        task = {
            "id": "math_01",
            "description": "What is 247 * 38?",
            "expected_keywords": ["9386"],
        }

        results = await run_task_benchmark(
            task, provider, mock_orchestrator, "free", "test-model",
        )

        direct = results[0]
        assert direct.tokens == 25  # 20 + 5
        assert direct.cost == 0.0001
        assert direct.success is True
        assert direct.quality == 5  # "9386" keyword found

    async def test_pipeline_mode_captures_metrics(self) -> None:
        """Pipeline mode should capture metrics from ExecutionResult."""
        provider = MockBenchmarkProvider()
        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value=ExecutionResult(
            root_id="r",
            output="The answer is 9386",
            success=True,
            total_tokens=45,
            total_cost=0.003,
        ))

        task = {
            "id": "math_01",
            "description": "What is 247 * 38?",
            "expected_keywords": ["9386"],
        }

        results = await run_task_benchmark(
            task, provider, mock_orchestrator, "mid", "test-model",
        )

        pipeline = results[1]
        assert pipeline.tokens == 45
        assert pipeline.cost == 0.003
        assert pipeline.success is True
        assert pipeline.mode == "pipeline"
        assert pipeline.tier == "mid"

    async def test_handles_provider_error_gracefully(self) -> None:
        """If the provider raises, result should show success=False."""
        provider = MockBenchmarkProvider()
        # Override complete to raise
        provider.complete = AsyncMock(side_effect=Exception("API down"))

        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(
            side_effect=Exception("Pipeline down")
        )

        task = {
            "id": "fail_01",
            "description": "This should fail",
            "expected_keywords": [],
        }

        results = await run_task_benchmark(
            task, provider, mock_orchestrator, "free", "fail-model",
        )

        assert len(results) == 2
        assert results[0].success is False
        assert results[1].success is False
        assert results[0].quality == 1
        assert results[1].quality == 1


# ---------------------------------------------------------------------------
# Tests: compute_tier_summaries
# ---------------------------------------------------------------------------


class TestComputeTierSummaries:
    """Test aggregation of per-tier summaries."""

    def test_single_tier_averages(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", tokens=100, cost=0.001, quality=3),
            _make_run_result(tier="free", mode="direct", tokens=200, cost=0.003, quality=5),
        ]
        summaries = compute_tier_summaries(results, "direct")

        assert "free" in summaries
        s = summaries["free"]
        assert s.avg_tokens == 150.0
        assert s.avg_cost == 0.002
        assert s.avg_quality == 4.0
        assert s.total_tasks == 2

    def test_multiple_tiers(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", tokens=100),
            _make_run_result(tier="mid", mode="direct", tokens=200),
            _make_run_result(tier="frontier", mode="direct", tokens=300),
        ]
        summaries = compute_tier_summaries(results, "direct")

        assert len(summaries) == 3
        assert "free" in summaries
        assert "mid" in summaries
        assert "frontier" in summaries

    def test_filters_by_mode(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", tokens=100),
            _make_run_result(tier="free", mode="pipeline", tokens=50),
        ]

        direct = compute_tier_summaries(results, "direct")
        pipeline = compute_tier_summaries(results, "pipeline")

        assert direct["free"].avg_tokens == 100.0
        assert pipeline["free"].avg_tokens == 50.0

    def test_success_rate(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", success=True),
            _make_run_result(tier="free", mode="direct", success=True),
            _make_run_result(tier="free", mode="direct", success=False),
        ]
        summaries = compute_tier_summaries(results, "direct")
        assert abs(summaries["free"].success_rate - 2.0 / 3.0) < 0.01

    def test_token_reduction_vs_frontier(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", tokens=50),
            _make_run_result(tier="frontier", mode="direct", tokens=200),
        ]
        summaries = compute_tier_summaries(results, "direct")

        # free uses 50 tokens vs frontier 200 => (200-50)/200 * 100 = 75%
        assert abs(summaries["free"].token_reduction_vs_frontier_pct - 75.0) < 0.1

    def test_frontier_token_reduction_is_zero(self) -> None:
        results = [
            _make_run_result(tier="frontier", mode="direct", tokens=200),
        ]
        summaries = compute_tier_summaries(results, "direct")
        assert summaries["frontier"].token_reduction_vs_frontier_pct == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_pipeline_token_reduction
# ---------------------------------------------------------------------------


class TestComputePipelineTokenReduction:
    """Test pipeline vs direct token reduction calculation."""

    def test_reduction_computed_correctly(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", tokens=200),
            _make_run_result(tier="free", mode="pipeline", tokens=100),
        ]
        reductions = compute_pipeline_token_reduction(results)
        # (200-100)/200 * 100 = 50%
        assert abs(reductions["free"] - 50.0) < 0.1

    def test_negative_reduction_when_pipeline_uses_more(self) -> None:
        results = [
            _make_run_result(tier="mid", mode="direct", tokens=100),
            _make_run_result(tier="mid", mode="pipeline", tokens=150),
        ]
        reductions = compute_pipeline_token_reduction(results)
        # Pipeline uses more tokens: (100-150)/100 * 100 = -50%
        assert abs(reductions["mid"] - (-50.0)) < 0.1

    def test_zero_direct_tokens(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", tokens=0),
            _make_run_result(tier="free", mode="pipeline", tokens=50),
        ]
        reductions = compute_pipeline_token_reduction(results)
        assert reductions["free"] == 0.0

    def test_multiple_tiers(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct", tokens=200),
            _make_run_result(tier="free", mode="pipeline", tokens=100),
            _make_run_result(tier="mid", mode="direct", tokens=300),
            _make_run_result(tier="mid", mode="pipeline", tokens=200),
        ]
        reductions = compute_pipeline_token_reduction(results)
        assert abs(reductions["free"] - 50.0) < 0.1
        assert abs(reductions["mid"] - 33.3) < 0.1


# ---------------------------------------------------------------------------
# Tests: format_summary_table
# ---------------------------------------------------------------------------


class TestFormatSummaryTable:
    """Test the summary table output formatting."""

    def test_contains_mode_headers(self) -> None:
        direct = {
            "free": TierSummary(
                tier="free", avg_quality=3.5, avg_tokens=100, avg_cost=0.001,
                avg_latency_ms=50, total_tasks=5, success_rate=1.0,
                token_reduction_vs_frontier_pct=50.0,
            ),
        }
        pipeline: dict[str, TierSummary] = {}
        reductions: dict[str, float] = {}

        table = format_summary_table(direct, pipeline, reductions)
        assert "MODE A: Direct Single-Call" in table
        assert "MODE B: GraphBot Pipeline" in table

    def test_contains_tier_data(self) -> None:
        direct = {
            "free": TierSummary(
                tier="free", avg_quality=3.5, avg_tokens=100, avg_cost=0.001,
                avg_latency_ms=50, total_tasks=5, success_rate=1.0,
                token_reduction_vs_frontier_pct=50.0,
            ),
            "frontier": TierSummary(
                tier="frontier", avg_quality=4.5, avg_tokens=200, avg_cost=0.01,
                avg_latency_ms=200, total_tasks=5, success_rate=1.0,
                token_reduction_vs_frontier_pct=0.0,
            ),
        }
        pipeline: dict[str, TierSummary] = {}
        reductions: dict[str, float] = {}

        table = format_summary_table(direct, pipeline, reductions)
        assert "free" in table
        assert "frontier" in table

    def test_contains_pipeline_reduction_section(self) -> None:
        direct: dict[str, TierSummary] = {}
        pipeline: dict[str, TierSummary] = {}
        reductions = {"free": 45.2, "mid": 30.1}

        table = format_summary_table(direct, pipeline, reductions)
        assert "Pipeline vs Direct Token Reduction" in table
        assert "45.2%" in table


# ---------------------------------------------------------------------------
# Tests: results_to_json
# ---------------------------------------------------------------------------


class TestResultsToJson:
    """Test JSON serialization of results."""

    def test_json_structure(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct"),
        ]
        direct = {"free": TierSummary(
            tier="free", avg_quality=4.0, avg_tokens=100, avg_cost=0.001,
            avg_latency_ms=50, total_tasks=1, success_rate=1.0,
            token_reduction_vs_frontier_pct=0.0,
        )}
        pipeline: dict[str, TierSummary] = {}
        reductions: dict[str, float] = {}

        data = results_to_json(results, direct, pipeline, reductions)

        assert "timestamp" in data
        assert "task_count" in data
        assert "results" in data
        assert "tier_summaries" in data
        assert "pipeline_vs_direct_token_reduction" in data

    def test_results_are_serializable(self) -> None:
        results = [
            _make_run_result(tier="free", mode="direct"),
            _make_run_result(tier="mid", mode="pipeline"),
        ]
        direct: dict[str, TierSummary] = {}
        pipeline: dict[str, TierSummary] = {}
        reductions: dict[str, float] = {}

        data = results_to_json(results, direct, pipeline, reductions)
        # Must be valid JSON
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert len(parsed["results"]) == 2

    def test_result_fields_present(self) -> None:
        results = [_make_run_result(tier="free", mode="direct")]
        data = results_to_json(results, {}, {}, {})

        entry = data["results"][0]
        assert "model" in entry
        assert "tier" in entry
        assert "mode" in entry
        assert "task_id" in entry
        assert "quality" in entry
        assert "tokens" in entry
        assert "cost" in entry
        assert "latency_ms" in entry
        assert "success" in entry
        assert "keyword_hits" in entry
        assert "keyword_total" in entry


# ---------------------------------------------------------------------------
# Tests: BENCHMARK_TASKS structure
# ---------------------------------------------------------------------------


class TestBenchmarkTasks:
    """Validate the embedded benchmark task definitions."""

    def test_at_least_15_tasks(self) -> None:
        assert len(BENCHMARK_TASKS) >= 15

    def test_unique_ids(self) -> None:
        ids = [t["id"] for t in BENCHMARK_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs: {ids}"

    def test_each_task_has_required_fields(self) -> None:
        required = {"id", "description", "category", "expected_keywords"}
        for task in BENCHMARK_TASKS:
            missing = required - set(task.keys())
            assert not missing, (
                f"Task {task.get('id', '???')} missing fields: {missing}"
            )

    def test_diverse_categories(self) -> None:
        categories = {t["category"] for t in BENCHMARK_TASKS}
        assert len(categories) >= 5, (
            f"Expected 5+ diverse categories, got {len(categories)}: {categories}"
        )


# ---------------------------------------------------------------------------
# Tests: MODEL_TIERS structure
# ---------------------------------------------------------------------------


class TestModelTiers:
    """Validate model tier definitions."""

    def test_has_free_tier(self) -> None:
        assert "free" in MODEL_TIERS
        assert len(MODEL_TIERS["free"]) >= 1

    def test_has_mid_tier(self) -> None:
        assert "mid" in MODEL_TIERS
        assert len(MODEL_TIERS["mid"]) >= 1

    def test_has_frontier_tier(self) -> None:
        assert "frontier" in MODEL_TIERS
        assert len(MODEL_TIERS["frontier"]) >= 1

    def test_free_tier_contains_8b(self) -> None:
        assert "meta-llama/llama-3.1-8b-instruct" in MODEL_TIERS["free"]

    def test_mid_tier_contains_70b(self) -> None:
        assert "meta-llama/llama-3.3-70b-instruct" in MODEL_TIERS["mid"]

    def test_mid_tier_contains_gpt4o_mini(self) -> None:
        assert "openai/gpt-4o-mini" in MODEL_TIERS["mid"]

    def test_frontier_tier_contains_claude_sonnet(self) -> None:
        assert "anthropic/claude-sonnet-4-6" in MODEL_TIERS["frontier"]

    def test_frontier_tier_contains_claude_opus(self) -> None:
        assert "anthropic/claude-opus-4-6" in MODEL_TIERS["frontier"]

    def test_frontier_tier_contains_gpt4o(self) -> None:
        assert "openai/gpt-4o" in MODEL_TIERS["frontier"]


# ---------------------------------------------------------------------------
# Tests: run_full_benchmark (integration with mocks)
# ---------------------------------------------------------------------------


class TestRunFullBenchmark:
    """Integration test for the full benchmark runner with mocked providers."""

    async def test_runs_all_tiers_and_tasks(self) -> None:
        """Full benchmark should produce 2 results per (model, task) pair."""
        provider = MockBenchmarkProvider()
        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value=ExecutionResult(
            root_id="r", output="Pipeline result", success=True,
            total_tokens=20, total_cost=0.001,
        ))

        # Use only 2 tasks and 2 tiers for speed
        tasks = BENCHMARK_TASKS[:2]
        tiers = {
            "free": ["meta-llama/llama-3.1-8b-instruct"],
            "mid": ["openai/gpt-4o-mini"],
        }

        results = await run_full_benchmark(
            tasks, tiers, provider, mock_orchestrator,
        )

        # 2 tiers * 1 model each * 2 tasks * 2 modes = 8 results
        assert len(results) == 8

        # Verify both modes present
        modes = {r.mode for r in results}
        assert modes == {"direct", "pipeline"}

        # Verify both tiers present
        tiers_seen = {r.tier for r in results}
        assert tiers_seen == {"free", "mid"}

    async def test_results_have_correct_tier_labels(self) -> None:
        provider = MockBenchmarkProvider()
        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value=ExecutionResult(
            root_id="r", output="Output", success=True,
        ))

        tasks = BENCHMARK_TASKS[:1]
        tiers = {"frontier": ["openai/gpt-4o"]}

        results = await run_full_benchmark(
            tasks, tiers, provider, mock_orchestrator,
        )

        for r in results:
            assert r.tier == "frontier"
            assert r.model == "openai/gpt-4o"
