"""Tests for tools_gb.shell -- ShellTool."""

import os
import sys

import pytest

from tools_gb.shell import BLOCKED_PATTERNS, ShellTool


@pytest.fixture
def shell(tmp_path):
    return ShellTool(workspace=str(tmp_path), timeout=10, max_output=5000)


@pytest.mark.asyncio
async def test_run_simple_command(shell):
    result = await shell.run("echo hello")
    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_run_exit_code(shell):
    cmd = "python -c \"import sys; sys.exit(1)\""
    result = await shell.run(cmd)
    assert result["success"] is False
    assert result["exit_code"] != 0


@pytest.mark.asyncio
async def test_blocked_command(shell):
    result = await shell.run("rm -rf /")
    assert result["success"] is False
    assert result["exit_code"] == -1
    assert "Blocked" in result["error"]


@pytest.mark.asyncio
async def test_blocked_format(shell):
    result = await shell.run("format C:")
    assert result["success"] is False
    assert result["exit_code"] == -1
    assert "Blocked" in result["error"]


@pytest.mark.asyncio
async def test_timeout(shell):
    result = await shell.run("python -c \"import time; time.sleep(10)\"", timeout=1)
    assert result["success"] is False
    assert "Timeout" in result["error"]


@pytest.mark.asyncio
async def test_output_truncation():
    tool = ShellTool(max_output=100)
    # Generate output longer than 100 chars
    cmd = "python -c \"print('A' * 500)\""
    result = await tool.run(cmd)
    assert result["success"] is True
    assert "truncated" in result["stdout"]
    assert len(result["stdout"]) < 500


@pytest.mark.asyncio
async def test_env_filtered(shell):
    # Set a secret-like env var and verify it is not passed through
    os.environ["TEST_API_KEY_GRAPHBOT"] = "supersecret"
    try:
        cmd = "python -c \"import os; print(os.environ.get('TEST_API_KEY_GRAPHBOT', 'MISSING'))\""
        result = await shell.run(cmd)
        assert result["success"] is True
        assert "MISSING" in result["stdout"]
        assert "supersecret" not in result["stdout"]
    finally:
        del os.environ["TEST_API_KEY_GRAPHBOT"]


@pytest.mark.asyncio
async def test_ansi_stripped(shell):
    # Echo ANSI escape codes and verify they are stripped
    cmd = "python -c \"print('\\x1b[31mred\\x1b[0m normal')\""
    result = await shell.run(cmd)
    assert result["success"] is True
    assert "\x1b" not in result["stdout"]
    assert "red" in result["stdout"]
    assert "normal" in result["stdout"]


@pytest.mark.asyncio
async def test_workspace_cwd(tmp_path):
    tool = ShellTool(workspace=str(tmp_path))
    cmd = "python -c \"import os; print(os.getcwd())\""
    result = await tool.run(cmd)
    assert result["success"] is True
    # Normalize paths for comparison (resolve symlinks, case differences on Windows)
    reported = os.path.normcase(os.path.realpath(result["stdout"].strip()))
    expected = os.path.normcase(os.path.realpath(str(tmp_path)))
    assert reported == expected
