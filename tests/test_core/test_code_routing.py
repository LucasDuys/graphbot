"""Tests for code tool routing -- R002 acceptance criteria.

Verifies that:
- Code generation/question prompts go to LLM, not shell tool
- Code edit requests go to CodeEditAgent
- code_generate tool_method returns None (skip to LLM)
- code_edit tool_method routes to CodeEditAgent
- CODE domain without tool_method skips tool attempt for non-edit tasks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.intake import IntakeParser
from core_gb.types import Domain, ExecutionResult, TaskNode, TaskStatus
from tools_gb.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Intake classification tests
# ---------------------------------------------------------------------------

class TestCodeDomainNarrowing:
    """CODE domain should only match explicit code-edit or shell requests."""

    def setup_method(self) -> None:
        self.parser = IntakeParser()

    def test_code_generation_not_code_domain(self) -> None:
        """'Write a Python function' is generation, not a code-edit task."""
        result = self.parser.parse("Write a Python function to sort a list")
        assert result.domain != Domain.CODE, (
            "Code generation prompts must not route to CODE domain"
        )

    def test_code_question_not_code_domain(self) -> None:
        """'Explain how classes work in Python' is a question, not code-edit."""
        result = self.parser.parse("Explain how classes work in Python")
        assert result.domain != Domain.CODE

    def test_code_edit_stays_code_domain(self) -> None:
        """'Edit file main.py to fix the bug' is a real code-edit task."""
        result = self.parser.parse("Edit file main.py to fix the bug")
        assert result.domain == Domain.CODE

    def test_debug_stays_code_domain(self) -> None:
        """'Debug this code and fix the bug' is a code-edit task."""
        result = self.parser.parse("Debug this code and fix the bug")
        assert result.domain == Domain.CODE

    def test_refactor_stays_code_domain(self) -> None:
        """'Refactor the class to use composition' is a code-edit task."""
        result = self.parser.parse("Refactor the class to use composition")
        assert result.domain == Domain.CODE

    def test_run_command_stays_code_domain(self) -> None:
        """'Run pytest tests/' is a shell execution task."""
        result = self.parser.parse("Run pytest tests/")
        assert result.domain == Domain.CODE

    def test_shell_command_stays_code_domain(self) -> None:
        """'Execute git log --oneline' is a shell task."""
        result = self.parser.parse("Execute git log --oneline")
        assert result.domain == Domain.CODE

    def test_implement_feature_in_file_routes_to_file(self) -> None:
        """'Implement the login function in auth.py' references a file path.

        FILE domain is correct here -- the FILE domain's CodeEditAgent
        handles file-edit requests through the existing FILE->edit path.
        """
        result = self.parser.parse("Implement the login function in auth.py")
        # FILE domain because '.py' file extension triggers FILE keywords.
        # This is correct: FILE domain routes edit requests to CodeEditAgent.
        assert result.domain == Domain.FILE

    def test_write_haiku_not_code(self) -> None:
        """'Write a haiku about spring' should not be CODE domain."""
        result = self.parser.parse("Write a haiku about spring")
        assert result.domain != Domain.CODE

    def test_generate_code_snippet_not_code(self) -> None:
        """'Generate a JavaScript snippet for form validation' is generation."""
        result = self.parser.parse(
            "Generate a JavaScript snippet for form validation"
        )
        assert result.domain != Domain.CODE


# ---------------------------------------------------------------------------
# Registry routing tests
# ---------------------------------------------------------------------------

def _make_node(
    domain: Domain,
    description: str,
    tool_method: str | None = None,
    tool_params: dict[str, str] | None = None,
    node_id: str = "test-node-001",
) -> TaskNode:
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=domain,
        complexity=1,
        status=TaskStatus.READY,
        tool_method=tool_method,
        tool_params=tool_params or {},
    )


class TestCodeGenerateToolMethod:
    """code_generate tool_method should return None -- skip to LLM."""

    @pytest.mark.asyncio
    async def test_code_generate_returns_none(self, tmp_path: Path) -> None:
        registry = ToolRegistry(workspace=str(tmp_path))
        node = _make_node(
            Domain.CODE,
            "Write a Python function to sort a list",
            tool_method="code_generate",
        )
        result = await registry._execute_by_method(node)
        assert result is None, "code_generate must return None to skip to LLM"


class TestCodeEditToolMethod:
    """code_edit tool_method should route to CodeEditAgent."""

    @pytest.mark.asyncio
    async def test_code_edit_routes_to_code_agent(self, tmp_path: Path) -> None:
        router = AsyncMock()
        registry = ToolRegistry(workspace=str(tmp_path), router=router)

        # Mock the code agent's edit method
        mock_result = ExecutionResult(
            root_id="test",
            output="Edit applied to main.py",
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=50.0,
            total_cost=0.001,
            model_used="code_agent",
        )
        registry._code_agent = MagicMock()
        registry._code_agent.edit = AsyncMock(return_value=mock_result)

        node = _make_node(
            Domain.CODE,
            "Fix the typo in main.py",
            tool_method="code_edit",
            tool_params={"path": "main.py"},
        )
        result = await registry._execute_by_method(node)

        assert result is not None
        assert result.success is True
        assert result.model_used == "code_agent"
        registry._code_agent.edit.assert_awaited_once()


class TestCodeDomainNoToolMethod:
    """CODE domain without tool_method should skip tool for non-edit tasks."""

    @pytest.mark.asyncio
    async def test_code_generation_skips_shell(self, tmp_path: Path) -> None:
        """Non-edit CODE domain task should skip shell tool entirely."""
        registry = ToolRegistry(workspace=str(tmp_path))
        node = _make_node(
            Domain.CODE,
            "Write a Python function to calculate fibonacci",
        )

        result = await registry.execute(node)

        # Should NOT have attempted shell execution -- should signal
        # that it needs LLM fallback (success=False with skip signal)
        # OR return a no-tool result so the executor falls back to LLM.
        assert result.success is False
        assert "No tool" in result.errors[0] or "skip" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_shell_command_still_works(self, tmp_path: Path) -> None:
        """Explicit shell commands in CODE domain still execute via ShellTool."""
        registry = ToolRegistry(workspace=str(tmp_path))
        node = _make_node(Domain.CODE, "run `echo hello`")
        mock_result = {
            "success": True,
            "stdout": "hello",
            "stderr": "",
            "exit_code": 0,
        }
        registry._shell.run = AsyncMock(return_value=mock_result)

        result = await registry.execute(node)

        assert result.success is True
        registry._shell.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_request_routes_to_code_agent(self, tmp_path: Path) -> None:
        """CODE domain edit request routes to CodeEditAgent when available."""
        router = AsyncMock()
        registry = ToolRegistry(workspace=str(tmp_path), router=router)

        mock_result = ExecutionResult(
            root_id="test",
            output="Edit applied",
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=50.0,
            total_cost=0.001,
            model_used="code_agent",
        )
        registry._code_agent.edit = AsyncMock(return_value=mock_result)

        node = _make_node(
            Domain.CODE,
            "Edit main.py to fix the import error",
        )
        result = await registry.execute(node)

        assert result.success is True
        registry._code_agent.edit.assert_awaited_once()
