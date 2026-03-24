"""End-to-end tests for the research report demo pipeline.

Validates dry-run mode, output file creation, pipeline stats display,
and correct Orchestrator interaction with mocked components.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on the path so demo script imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core_gb.types import ExecutionResult

# Import demo module components.
from scripts.demo_research_report import (
    DEMO_TASK,
    DEMOS_DIR,
    OUTPUT_PATH,
    PipelineStats,
    _build_mock_completion,
    _extract_stats,
    _mock_router_factory,
    _MOCK_SYNTHESIS,
    run_dry,
    save_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_execution_result() -> ExecutionResult:
    """Build a representative ExecutionResult for stats extraction tests."""
    return ExecutionResult(
        root_id="root",
        output=_MOCK_SYNTHESIS,
        success=True,
        total_nodes=8,
        total_tokens=3500,
        total_latency_ms=320.0,
        total_cost=0.000450,
        model_used="mock/dry-run",
        tools_used=5,
        llm_calls=7,
        nodes=("root", "research", "search1", "search2", "search3", "search4", "search5", "synthesize"),
        errors=(),
    )


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Provide a temporary demos directory for file-write tests."""
    demos = tmp_path / "demos"
    demos.mkdir()
    return demos


# ---------------------------------------------------------------------------
# Dry-run mode tests
# ---------------------------------------------------------------------------


class TestDryRunMode:
    """Tests for the dry-run pipeline producing valid output."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_successful_result(self) -> None:
        """Dry-run mode should return a successful ExecutionResult with content."""
        result, elapsed = await run_dry()

        assert result.success is True
        assert elapsed > 0.0
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_dry_run_output_contains_framework_names(self) -> None:
        """The dry-run synthesis should mention all 5 frameworks."""
        result, _ = await run_dry()

        for framework in ["LangGraph", "CrewAI", "AutoGen", "OpenAI Agents SDK", "Claude Agent SDK"]:
            assert framework in result.output, (
                f"Expected framework '{framework}' in dry-run output"
            )

    @pytest.mark.asyncio
    async def test_dry_run_output_contains_comparison_table(self) -> None:
        """The mock synthesis includes a markdown comparison table."""
        result, _ = await run_dry()

        assert "|" in result.output, "Expected markdown table in output"
        assert "Framework" in result.output or "Best For" in result.output

    @pytest.mark.asyncio
    async def test_dry_run_no_api_calls(self) -> None:
        """Dry-run mode should never call a real OpenRouterProvider."""
        with patch("models.openrouter.OpenRouterProvider") as mock_provider_cls:
            result, _ = await run_dry()
            # The provider class should NOT have been instantiated by run_dry.
            mock_provider_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------


class TestOutputSave:
    """Tests for saving the report to demos/research_report.md."""

    def test_save_report_creates_file(self, tmp_path: Path) -> None:
        """save_report should write the output to the expected path."""
        result = ExecutionResult(
            root_id="root",
            output="# Test Report\n\nThis is a test.",
            success=True,
        )

        # Temporarily redirect OUTPUT_PATH to tmp_path.
        target = tmp_path / "demos" / "research_report.md"
        with patch("scripts.demo_research_report.DEMOS_DIR", tmp_path / "demos"), \
             patch("scripts.demo_research_report.OUTPUT_PATH", target):
            saved = save_report(result)

        assert saved.exists()
        content = saved.read_text(encoding="utf-8")
        assert "# Test Report" in content
        assert "This is a test." in content

    def test_save_report_creates_parent_directory(self, tmp_path: Path) -> None:
        """save_report should create the demos directory if it does not exist."""
        demos = tmp_path / "new_demos"
        target = demos / "research_report.md"

        result = ExecutionResult(
            root_id="root",
            output="Content here",
            success=True,
        )

        with patch("scripts.demo_research_report.DEMOS_DIR", demos), \
             patch("scripts.demo_research_report.OUTPUT_PATH", target):
            saved = save_report(result)

        assert demos.is_dir()
        assert saved.exists()

    def test_save_report_overwrites_existing(self, tmp_path: Path) -> None:
        """save_report should overwrite any existing file at the path."""
        demos = tmp_path / "demos"
        demos.mkdir()
        target = demos / "research_report.md"
        target.write_text("old content", encoding="utf-8")

        result = ExecutionResult(
            root_id="root",
            output="new content",
            success=True,
        )

        with patch("scripts.demo_research_report.DEMOS_DIR", demos), \
             patch("scripts.demo_research_report.OUTPUT_PATH", target):
            save_report(result)

        assert target.read_text(encoding="utf-8") == "new content"


# ---------------------------------------------------------------------------
# Pipeline stats tests
# ---------------------------------------------------------------------------


class TestPipelineStats:
    """Tests for pipeline stats extraction and display."""

    def test_extract_stats_fields(self, mock_execution_result: ExecutionResult) -> None:
        """_extract_stats should populate all PipelineStats fields correctly."""
        stats = _extract_stats(DEMO_TASK, mock_execution_result, elapsed_s=1.5)

        assert stats.task == DEMO_TASK
        assert stats.success is True
        assert stats.nodes_executed == 8
        assert stats.latency_s == 1.5
        assert stats.total_cost == pytest.approx(0.000450)
        assert stats.total_tokens == 3500
        assert stats.tools_used == 5
        assert stats.llm_calls == 7
        assert stats.model_used == "mock/dry-run"
        assert stats.errors == ()

    def test_display_contains_all_stat_labels(
        self, mock_execution_result: ExecutionResult
    ) -> None:
        """The display string should contain labels for nodes, latency, cost, and tools."""
        stats = _extract_stats(DEMO_TASK, mock_execution_result, elapsed_s=2.0)
        display = stats.display()

        assert "Nodes executed" in display
        assert "Latency" in display
        assert "Total cost" in display
        assert "Tools used" in display
        assert "LLM calls" in display
        assert "Model" in display

    def test_display_contains_separator_lines(
        self, mock_execution_result: ExecutionResult
    ) -> None:
        """The display should be framed by separator lines."""
        stats = _extract_stats(DEMO_TASK, mock_execution_result, elapsed_s=0.5)
        display = stats.display()

        assert "=" * 60 in display
        assert "Pipeline Stats" in display

    def test_display_shows_errors_when_present(self) -> None:
        """When errors are present, they should appear in the display output."""
        result = ExecutionResult(
            root_id="root",
            output="partial",
            success=False,
            total_nodes=3,
            errors=("timeout on search3", "model overloaded"),
        )
        stats = _extract_stats(DEMO_TASK, result, elapsed_s=10.0)
        display = stats.display()

        assert "Errors" in display
        assert "timeout on search3" in display
        assert "model overloaded" in display


# ---------------------------------------------------------------------------
# Mocked Orchestrator tests
# ---------------------------------------------------------------------------


class TestMockedOrchestrator:
    """Tests using a fully mocked Orchestrator to verify pipeline wiring."""

    @pytest.mark.asyncio
    async def test_mock_router_handles_decomposition(self) -> None:
        """The mock router should return valid JSON for decomposition calls."""
        mock_router = _mock_router_factory()

        # Simulate a decomposition call.
        from core_gb.types import TaskNode

        node = TaskNode(id="root", description="decompose")
        messages = [{"role": "system", "content": "You are a Task Decomposer agent."}]
        result = await mock_router.route(node, messages)

        assert result.content is not None
        import json
        parsed = json.loads(result.content)
        assert "nodes" in parsed
        assert len(parsed["nodes"]) > 0

    @pytest.mark.asyncio
    async def test_mock_router_handles_search_calls(self) -> None:
        """The mock router should return framework-specific text for search nodes."""
        mock_router = _mock_router_factory()
        from core_gb.types import TaskNode

        node = TaskNode(id="search1", description="Search for LangGraph framework 2026")
        messages = [{"role": "user", "content": "search"}]
        result = await mock_router.route(node, messages)

        assert "LangGraph" in result.content

    @pytest.mark.asyncio
    async def test_mock_router_handles_synthesis(self) -> None:
        """The mock router should return the full synthesis for non-decomposition, non-search calls."""
        mock_router = _mock_router_factory()
        from core_gb.types import TaskNode

        node = TaskNode(id="synthesize", description="Combine all research into report")
        messages = [{"role": "user", "content": "synthesize"}]
        result = await mock_router.route(node, messages)

        assert "Comparison" in result.content or "comparison" in result.content.lower()

    def test_build_mock_completion_fields(self) -> None:
        """_build_mock_completion should return a well-formed CompletionResult."""
        cr = _build_mock_completion("test content", model="test/model")

        assert cr.content == "test content"
        assert cr.model == "test/model"
        assert cr.tokens_in == 200
        assert cr.tokens_out == 300
        assert cr.latency_ms == 50.0
        assert cr.cost == 0.0

    @pytest.mark.asyncio
    async def test_orchestrator_called_with_demo_task(self) -> None:
        """run_dry should pass the DEMO_TASK string to orchestrator.process."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.process = AsyncMock(
            return_value=ExecutionResult(
                root_id="root",
                output="mock output",
                success=True,
                total_nodes=1,
            )
        )

        with patch("scripts.demo_research_report.Orchestrator", return_value=mock_orchestrator), \
             patch("scripts.demo_research_report.GraphStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store
            result, elapsed = await run_dry()

        mock_orchestrator.process.assert_called_once_with(DEMO_TASK)
