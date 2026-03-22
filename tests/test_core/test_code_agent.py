"""Tests for the CodeEditAgent read-analyze-edit-test loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core_gb.code_agent import CodeEditAgent, MAX_RETRIES
from core_gb.types import CompletionResult, ExecutionResult


class _FakeFileTool:
    """FileTool backed by a real temp directory."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    def read(self, path: str) -> dict[str, Any]:
        p = self._workspace / path
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        content = p.read_text(encoding="utf-8")
        return {"success": True, "content": content}

    def edit(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        p = self._workspace / path
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        content = p.read_text(encoding="utf-8")
        if old_text not in content:
            return {"success": False, "error": "Text not found in file."}
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return {"success": True, "path": str(p), "replacements": 1}


class _FakeShellTool:
    """ShellTool that returns configurable results per call."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []
        self._call_idx = 0

    async def run(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        if self._call_idx < len(self._results):
            result = self._results[self._call_idx]
            self._call_idx += 1
            return result
        return {"success": True, "stdout": "All tests passed", "stderr": "", "exit_code": 0}


def _make_completion(old_text: str, new_text: str) -> CompletionResult:
    content = json.dumps({"old_text": old_text, "new_text": new_text})
    return CompletionResult(
        content=content,
        model="test-model",
        tokens_in=10,
        tokens_out=20,
        latency_ms=5.0,
        cost=0.001,
    )


class TestCodeEditAgent:
    async def test_edit_simple_replacement(self, tmp_path: Path) -> None:
        """Edit replaces 'teh' with 'the' in a temp file."""
        test_file = tmp_path / "hello.py"
        test_file.write_text("print('teh world')\n", encoding="utf-8")

        file_tool = _FakeFileTool(tmp_path)
        shell_tool = _FakeShellTool()
        router = AsyncMock()
        router.route.return_value = _make_completion("teh", "the")

        agent = CodeEditAgent(file_tool, shell_tool, router)
        result = await agent.edit("Fix typo: teh -> the", "hello.py")

        assert result.success is True
        assert "Edit applied" in result.output
        assert test_file.read_text(encoding="utf-8") == "print('the world')\n"

    async def test_edit_with_test_verification(self, tmp_path: Path) -> None:
        """Edit succeeds and test command also succeeds."""
        test_file = tmp_path / "app.py"
        test_file.write_text("x = 1 + 2\n", encoding="utf-8")

        file_tool = _FakeFileTool(tmp_path)
        shell_tool = _FakeShellTool([
            {"success": True, "stdout": "OK", "stderr": "", "exit_code": 0},
        ])
        router = AsyncMock()
        router.route.return_value = _make_completion("1 + 2", "1 + 3")

        agent = CodeEditAgent(file_tool, shell_tool, router)
        result = await agent.edit(
            "Change 1+2 to 1+3", "app.py", test_command="python -m pytest",
        )

        assert result.success is True
        assert test_file.read_text(encoding="utf-8") == "x = 1 + 3\n"

    async def test_edit_retry_on_test_failure(self, tmp_path: Path) -> None:
        """First edit breaks tests, second edit fixes them."""
        test_file = tmp_path / "calc.py"
        test_file.write_text("result = 10 / 2\n", encoding="utf-8")

        file_tool = _FakeFileTool(tmp_path)
        # First test run fails, second succeeds
        shell_tool = _FakeShellTool([
            {"success": False, "stdout": "AssertionError", "stderr": "", "exit_code": 1},
            {"success": True, "stdout": "OK", "stderr": "", "exit_code": 0},
        ])
        router = AsyncMock()
        # First call: bad edit; second call: good edit
        router.route.side_effect = [
            _make_completion("10 / 2", "10 / 0"),  # bad: div by zero
            _make_completion("10 / 0", "10 / 5"),  # fix
        ]

        agent = CodeEditAgent(file_tool, shell_tool, router)
        result = await agent.edit(
            "Change divisor", "calc.py", test_command="python -m pytest",
        )

        assert result.success is True
        assert test_file.read_text(encoding="utf-8") == "result = 10 / 5\n"
        assert router.route.call_count == 2

    async def test_edit_max_retries_exceeded(self, tmp_path: Path) -> None:
        """All attempts fail -- returns failure with error message."""
        test_file = tmp_path / "broken.py"
        test_file.write_text("x = 1\n", encoding="utf-8")

        file_tool = _FakeFileTool(tmp_path)
        shell_tool = _FakeShellTool()
        router = AsyncMock()
        # Router always raises
        router.route.side_effect = Exception("LLM unavailable")

        agent = CodeEditAgent(file_tool, shell_tool, router)
        result = await agent.edit("Fix something", "broken.py")

        assert result.success is False
        assert len(result.errors) == 1
        assert f"after {MAX_RETRIES} attempts" in result.errors[0]

    async def test_edit_file_not_found(self, tmp_path: Path) -> None:
        """Returns failure when file does not exist."""
        file_tool = _FakeFileTool(tmp_path)
        shell_tool = _FakeShellTool()
        router = AsyncMock()

        agent = CodeEditAgent(file_tool, shell_tool, router)
        result = await agent.edit("Fix it", "nonexistent.py")

        assert result.success is False
        assert "Cannot read file" in result.errors[0]


class TestParseEdit:
    def test_parse_edit_valid_json(self) -> None:
        content = json.dumps({"old_text": "foo", "new_text": "bar"})
        result = CodeEditAgent._parse_edit(content)
        assert result is not None
        assert result["old_text"] == "foo"
        assert result["new_text"] == "bar"

    def test_parse_edit_invalid(self) -> None:
        result = CodeEditAgent._parse_edit("not json at all {{{")
        assert result is None

    def test_parse_edit_missing_keys(self) -> None:
        content = json.dumps({"old_text": "foo"})
        result = CodeEditAgent._parse_edit(content)
        assert result is None

    def test_parse_edit_extra_keys_ok(self) -> None:
        content = json.dumps({"old_text": "a", "new_text": "b", "explanation": "typo"})
        result = CodeEditAgent._parse_edit(content)
        assert result is not None
        assert result["old_text"] == "a"
        assert result["new_text"] == "b"
