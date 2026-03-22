"""Tests for shell command extraction and live shell execution."""

import os
import re

import pytest

from tools_gb.registry import ToolRegistry
from tools_gb.shell import ShellTool


# ---------------------------------------------------------------------------
# Unit tests for _extract_command
# ---------------------------------------------------------------------------


class TestExtractCommand:
    """Tests for ToolRegistry._extract_command static method."""

    def test_extract_backtick_command(self) -> None:
        result = ToolRegistry._extract_command("`git log --oneline -5`")
        assert result == "git log --oneline -5"

    def test_extract_single_quoted(self) -> None:
        result = ToolRegistry._extract_command("'python --version'")
        assert result == "python --version"

    def test_extract_run_prefix(self) -> None:
        result = ToolRegistry._extract_command("Run git log --oneline -10")
        assert result == "git log --oneline -10"

    def test_extract_run_the_command(self) -> None:
        result = ToolRegistry._extract_command(
            "Run the command 'python -m pytest tests/ --co -q'"
        )
        assert result == "python -m pytest tests/ --co -q"

    def test_extract_complex(self) -> None:
        result = ToolRegistry._extract_command(
            "Execute python -m pytest tests/ --co -q and count tests"
        )
        assert result == "python -m pytest tests/ --co -q"

    def test_extract_double_quoted(self) -> None:
        result = ToolRegistry._extract_command('Run "git status"')
        assert result == "git status"

    def test_extract_plain_passthrough(self) -> None:
        result = ToolRegistry._extract_command("git status")
        assert result == "git status"

    def test_extract_run_command_prefix(self) -> None:
        result = ToolRegistry._extract_command("Run command echo hello")
        assert result == "echo hello"

    def test_extract_lowercase_run(self) -> None:
        result = ToolRegistry._extract_command("run git diff --stat")
        assert result == "git diff --stat"

    def test_extract_execute_lowercase(self) -> None:
        result = ToolRegistry._extract_command("execute pip list")
        assert result == "pip list"

    def test_extract_suffix_then(self) -> None:
        result = ToolRegistry._extract_command(
            "Run git status then commit the changes"
        )
        assert result == "git status"

    def test_extract_suffix_in_order(self) -> None:
        result = ToolRegistry._extract_command(
            "Run pip install requests in order to add the dependency"
        )
        assert result == "pip install requests"


# ---------------------------------------------------------------------------
# Live shell execution tests
# ---------------------------------------------------------------------------


@pytest.fixture
def project_shell() -> ShellTool:
    """ShellTool with workspace set to the graphbot project root."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return ShellTool(workspace=project_root, timeout=15)


@pytest.mark.asyncio
async def test_shell_git_log_live(project_shell: ShellTool) -> None:
    result = await project_shell.run("git log --oneline -5")
    assert result["success"] is True
    stdout = result["stdout"]
    # Each line should start with a short commit hash (hex chars)
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    assert len(lines) > 0, "Expected at least one git log line"
    for line in lines:
        assert re.match(r"^[0-9a-f]{7,}", line.strip()), (
            f"Expected commit hash at start of line: {line!r}"
        )


@pytest.mark.asyncio
async def test_shell_python_version_live(project_shell: ShellTool) -> None:
    result = await project_shell.run("python --version")
    assert result["success"] is True
    assert "Python" in result["stdout"] or "Python" in result["stderr"]
