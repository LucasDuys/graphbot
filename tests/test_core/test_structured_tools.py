"""Tests for structured tool parameters (T088) and smart decomposition (T089)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.intake import IntakeParser, TaskType
from core_gb.types import Domain, ExecutionResult, TaskNode, TaskStatus


class TestTaskNodeToolFields:
    """T088: Verify TaskNode has tool_method and tool_params fields."""

    def test_task_node_has_tool_fields(self) -> None:
        node = TaskNode(id="t1", description="test")
        assert node.tool_method is None
        assert node.tool_params == {}

    def test_task_node_accepts_tool_method(self) -> None:
        node = TaskNode(
            id="t1",
            description="Read file",
            tool_method="file_read",
            tool_params={"path": "README.md"},
        )
        assert node.tool_method == "file_read"
        assert node.tool_params == {"path": "README.md"}


class TestRegistryUsesToolMethod:
    """T088: Verify ToolRegistry routes via tool_method when present."""

    @pytest.fixture
    def registry(self) -> MagicMock:
        from tools_gb.registry import ToolRegistry
        reg = ToolRegistry(workspace=".")
        return reg

    @pytest.mark.asyncio
    async def test_registry_uses_tool_method_file_read(self, registry: MagicMock) -> None:
        registry._file = MagicMock()
        registry._file.read.return_value = {
            "success": True,
            "content": "hello world",
        }

        node = TaskNode(
            id="t1",
            description="Read test.py",
            domain=Domain.FILE,
            is_atomic=True,
            tool_method="file_read",
            tool_params={"path": "test.py"},
        )
        result = await registry.execute(node)
        registry._file.read.assert_called_once_with("test.py")
        assert result.success is True
        assert result.output == "hello world"

    @pytest.mark.asyncio
    async def test_registry_uses_tool_params_web(self, registry: MagicMock) -> None:
        registry._web = MagicMock()
        registry._web.search = AsyncMock(return_value={
            "success": True,
            "content": "search results",
        })

        node = TaskNode(
            id="t2",
            description="Search for python",
            domain=Domain.WEB,
            is_atomic=True,
            tool_method="web_search",
            tool_params={"query": "python"},
        )
        result = await registry.execute(node)
        registry._web.search.assert_called_once_with("python")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_registry_fallback_no_method(self, registry: MagicMock) -> None:
        registry._web = MagicMock()
        registry._web.search_and_summarize = AsyncMock(return_value={
            "success": True,
            "content": "weather is nice",
        })

        node = TaskNode(
            id="t3",
            description="Get weather forecast",
            domain=Domain.WEB,
            is_atomic=True,
            # No tool_method set -- should fall back to domain routing
        )
        result = await registry.execute(node)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_registry_llm_reason_returns_none(self, registry: MagicMock) -> None:
        node = TaskNode(
            id="t4",
            description="Think about this",
            domain=Domain.SYNTHESIS,
            is_atomic=True,
            tool_method="llm_reason",
        )
        # llm_reason should return None from _execute_by_method,
        # then fall through to domain routing (_no_tool_result for SYNTHESIS)
        result = await registry.execute(node)
        assert result.success is False
        assert "No tool for domain" in result.errors[0]


class TestTaskTypeClassification:
    """T089: Verify IntakeParser classifies task types correctly."""

    def setup_method(self) -> None:
        self.parser = IntakeParser()

    def test_task_type_integrated(self) -> None:
        result = self.parser.parse("Compare Python vs JavaScript")
        assert result.task_type == TaskType.INTEGRATED

    def test_task_type_integrated_pros_cons(self) -> None:
        result = self.parser.parse("What are the pros and cons of microservices")
        assert result.task_type == TaskType.INTEGRATED

    def test_task_type_integrated_differences(self) -> None:
        result = self.parser.parse("What are the differences between REST and GraphQL")
        assert result.task_type == TaskType.INTEGRATED

    def test_task_type_integrated_with_tools_becomes_decomposable(self) -> None:
        # "Compare" + tool need -> should NOT be INTEGRATED (needs data first)
        result = self.parser.parse("Compare the weather in Amsterdam, London, and Berlin")
        assert result.task_type != TaskType.INTEGRATED

    def test_task_type_data_parallel(self) -> None:
        result = self.parser.parse(
            "Search the web for weather in Amsterdam, London, and Berlin"
        )
        assert result.task_type == TaskType.DATA_PARALLEL

    def test_task_type_sequential(self) -> None:
        result = self.parser.parse(
            "First read the file, then parse it, finally save the output"
        )
        assert result.task_type == TaskType.SEQUENTIAL

    def test_task_type_atomic(self) -> None:
        result = self.parser.parse("What is 2+2?")
        assert result.task_type == TaskType.ATOMIC

    def test_task_type_atomic_simple_question(self) -> None:
        result = self.parser.parse("Explain how recursion works")
        assert result.task_type == TaskType.ATOMIC


class TestIntegratedSkipsDecomposition:
    """T089: INTEGRATED tasks should go directly to executor, not decomposer."""

    @pytest.mark.asyncio
    async def test_integrated_skips_decomposition(self) -> None:
        from core_gb.orchestrator import Orchestrator

        mock_store = MagicMock()
        mock_store.get_context.return_value = None
        mock_router = MagicMock()

        orch = Orchestrator(mock_store, mock_router)

        fake_result = ExecutionResult(
            root_id="test",
            output="Comparison result",
            success=True,
            total_nodes=1,
            total_tokens=100,
        )
        orch._executor = MagicMock()
        orch._executor.execute = AsyncMock(return_value=fake_result)
        orch._decomposer = MagicMock()
        orch._decomposer.decompose = AsyncMock()
        orch._graph_updater = MagicMock()
        orch._pattern_store = MagicMock()
        orch._pattern_store.load_all.return_value = []

        result = await orch.process("Compare Python versus JavaScript")

        orch._executor.execute.assert_called_once()
        orch._decomposer.decompose.assert_not_called()
        assert result.success is True
