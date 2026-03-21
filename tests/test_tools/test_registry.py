"""Tests for tools_gb.registry.ToolRegistry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import Domain, TaskNode, TaskStatus
from tools_gb.registry import ToolRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(workspace=str(tmp_path))


def _make_node(
    domain: Domain,
    description: str,
    node_id: str = "test-node-001",
) -> TaskNode:
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=domain,
        complexity=1,
        status=TaskStatus.READY,
    )


class TestHasTool:
    def test_has_tool_web(self, registry: ToolRegistry) -> None:
        assert registry.has_tool(Domain.WEB) is True

    def test_has_tool_file(self, registry: ToolRegistry) -> None:
        assert registry.has_tool(Domain.FILE) is True

    def test_has_tool_code(self, registry: ToolRegistry) -> None:
        assert registry.has_tool(Domain.CODE) is True

    def test_no_tool_system(self, registry: ToolRegistry) -> None:
        assert registry.has_tool(Domain.SYSTEM) is False

    def test_no_tool_synthesis(self, registry: ToolRegistry) -> None:
        assert registry.has_tool(Domain.SYNTHESIS) is False

    def test_no_tool_comms(self, registry: ToolRegistry) -> None:
        assert registry.has_tool(Domain.COMMS) is False


class TestExecuteWeb:
    @pytest.mark.asyncio
    async def test_execute_web_search(self, registry: ToolRegistry) -> None:
        node = _make_node(Domain.WEB, "search python tutorials")
        mock_result = {
            "success": True,
            "results": [{"title": "Python Tutorial", "url": "https://example.com", "snippet": "Learn Python"}],
        }
        registry._web.search = AsyncMock(return_value=mock_result)

        result = await registry.execute(node)

        registry._web.search.assert_awaited_once_with(node.description)
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.success is True
        assert result.model_used == "tool:web"

    @pytest.mark.asyncio
    async def test_execute_web_fetch_url(self, registry: ToolRegistry) -> None:
        node = _make_node(Domain.WEB, "fetch content from https://example.com/page")
        mock_result = {"success": True, "content": "Page content here", "title": "Example"}
        registry._web.fetch = AsyncMock(return_value=mock_result)

        result = await registry.execute(node)

        registry._web.fetch.assert_awaited_once_with("https://example.com/page")
        assert result.success is True
        assert "Page content here" in result.output


class TestExecuteFile:
    @pytest.mark.asyncio
    async def test_execute_file_list(self, registry: ToolRegistry, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "b.py").write_text("world", encoding="utf-8")

        node = _make_node(Domain.FILE, "list files in current directory")

        result = await registry.execute(node)

        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.model_used == "tool:file"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_file_read(self, registry: ToolRegistry, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("file content here", encoding="utf-8")
        node = _make_node(Domain.FILE, "read the file 'readme.txt'")

        result = await registry.execute(node)

        assert result.success is True
        assert "file content here" in result.output


class TestExecuteShell:
    @pytest.mark.asyncio
    async def test_execute_shell(self, registry: ToolRegistry) -> None:
        node = _make_node(Domain.CODE, "run `echo hello`")
        mock_result = {"success": True, "stdout": "hello", "stderr": "", "exit_code": 0}
        registry._shell.run = AsyncMock(return_value=mock_result)

        result = await registry.execute(node)

        registry._shell.run.assert_awaited_once_with("echo hello")
        assert result.success is True
        assert "hello" in result.output
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.model_used == "tool:code"


class TestResultMetrics:
    @pytest.mark.asyncio
    async def test_result_has_zero_tokens(self, registry: ToolRegistry, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("data", encoding="utf-8")
        node = _make_node(Domain.FILE, "list files in .")

        result = await registry.execute(node)

        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_result_has_zero_cost(self, registry: ToolRegistry, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("data", encoding="utf-8")
        node = _make_node(Domain.FILE, "list files in .")

        result = await registry.execute(node)

        assert result.total_cost == 0.0


class TestNoToolDomain:
    @pytest.mark.asyncio
    async def test_no_tool_returns_error(self, registry: ToolRegistry) -> None:
        node = _make_node(Domain.SYSTEM, "do something system-level")

        result = await registry.execute(node)

        assert result.success is False
        assert "No tool for domain: system" in result.errors[0]
        assert result.total_tokens == 0
        assert result.total_cost == 0.0


class TestExceptionHandling:
    @pytest.mark.asyncio
    async def test_exception_returns_failure(self, registry: ToolRegistry) -> None:
        node = _make_node(Domain.WEB, "search something")
        registry._web.search = AsyncMock(side_effect=RuntimeError("network down"))

        result = await registry.execute(node)

        assert result.success is False
        assert "network down" in result.errors[0]
        assert result.total_tokens == 0
