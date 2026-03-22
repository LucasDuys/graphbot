"""Tests for tools_gb.shell -- ShellTool."""

import os
import sys
from unittest.mock import patch

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


@pytest.mark.asyncio
async def test_run_returns_answer_field(shell):
    """Shell run results include an interpreted 'answer' field."""
    result = await shell.run("echo hello world")
    assert "answer" in result
    assert "hello world" in result["answer"]


class TestInterpretOutput:
    """Unit tests for ShellTool.interpret_output."""

    def test_failed_command(self):
        answer = ShellTool.interpret_output("bad_cmd", "", "not found", 127)
        assert "failed" in answer.lower()
        assert "127" in answer

    def test_empty_stdout(self):
        answer = ShellTool.interpret_output("true", "", "", 0)
        assert "no output" in answer.lower()

    def test_short_output_returned_verbatim(self):
        stdout = "line1\nline2\nline3"
        answer = ShellTool.interpret_output("echo hi", stdout, "", 0)
        assert "line1" in answer
        assert "line2" in answer
        assert "line3" in answer

    def test_long_output_truncated(self):
        lines = [f"line{i}" for i in range(50)]
        stdout = "\n".join(lines)
        answer = ShellTool.interpret_output("some_cmd", stdout, "", 0)
        assert "50 total lines" in answer
        assert "line0" in answer   # head
        assert "line49" in answer  # tail

    def test_pytest_collect_output(self):
        stdout = "tests/test_a.py::test_1\ntests/test_a.py::test_2\n3 tests collected"
        answer = ShellTool.interpret_output(
            "python -m pytest tests/ --co -q", stdout, "", 0,
        )
        assert "3 tests collected" in answer

    def test_pytest_collect_fallback_counts_lines(self):
        stdout = "tests/test_a.py::test_1\ntests/test_b.py::test_2"
        answer = ShellTool.interpret_output(
            "python -m pytest --co -q tests/", stdout, "", 0,
        )
        assert "2 tests collected" in answer

    def test_git_log_output(self):
        stdout = "abc1234 first commit\ndef5678 second commit"
        answer = ShellTool.interpret_output(
            "git log --oneline -2", stdout, "", 0,
        )
        assert "2 commits" in answer
        assert "first commit" in answer
        assert "second commit" in answer


# ---------------------------------------------------------------------------
# Shell allowlist / blocklist from environment
# ---------------------------------------------------------------------------


class TestShellAllowlist:
    """SHELL_ALLOWLIST env var restricts execution to only listed commands."""

    @pytest.mark.asyncio
    async def test_allowlist_permits_listed_command(self, tmp_path) -> None:
        """Commands on the allowlist are permitted."""
        with patch.dict(os.environ, {"SHELL_ALLOWLIST": "echo,ls,python"}, clear=False):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("echo hello")
            assert result["success"] is True
            assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_allowlist_blocks_unlisted_command(self, tmp_path) -> None:
        """Commands not on the allowlist are blocked."""
        with patch.dict(os.environ, {"SHELL_ALLOWLIST": "echo,ls"}, clear=False):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("python -c \"print('hi')\"")
            assert result["success"] is False
            assert result["exit_code"] == -1
            assert "allowlist" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_allowlist_empty_means_unrestricted(self, tmp_path) -> None:
        """Empty SHELL_ALLOWLIST means no allowlist restriction."""
        with patch.dict(os.environ, {"SHELL_ALLOWLIST": ""}, clear=False):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("echo allowed")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_allowlist_not_set_means_unrestricted(self, tmp_path) -> None:
        """Absent SHELL_ALLOWLIST means no allowlist restriction."""
        env = os.environ.copy()
        env.pop("SHELL_ALLOWLIST", None)
        env.pop("SHELL_BLOCKLIST", None)
        with patch.dict(os.environ, env, clear=True):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("echo allowed")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_allowlist_whitespace_handling(self, tmp_path) -> None:
        """Allowlist entries are trimmed of whitespace."""
        with patch.dict(
            os.environ, {"SHELL_ALLOWLIST": " echo , ls , python "}, clear=False
        ):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("echo trimmed")
            assert result["success"] is True


class TestShellBlocklist:
    """SHELL_BLOCKLIST env var blocks specific commands on top of built-in patterns."""

    @pytest.mark.asyncio
    async def test_blocklist_blocks_listed_command(self, tmp_path) -> None:
        """Commands on the blocklist are blocked."""
        with patch.dict(os.environ, {"SHELL_BLOCKLIST": "curl,wget,nc"}, clear=False):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("curl http://example.com")
            assert result["success"] is False
            assert result["exit_code"] == -1
            assert "blocklist" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_blocklist_permits_unlisted_command(self, tmp_path) -> None:
        """Commands not on the blocklist are permitted."""
        with patch.dict(os.environ, {"SHELL_BLOCKLIST": "curl,wget"}, clear=False):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("echo hello")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_blocklist_empty_means_no_extra_blocks(self, tmp_path) -> None:
        """Empty SHELL_BLOCKLIST adds no extra blocks (built-in patterns still apply)."""
        with patch.dict(os.environ, {"SHELL_BLOCKLIST": ""}, clear=False):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("echo allowed")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_blocklist_plus_builtin_patterns(self, tmp_path) -> None:
        """Built-in BLOCKED_PATTERNS still apply even with custom blocklist."""
        with patch.dict(os.environ, {"SHELL_BLOCKLIST": "nc"}, clear=False):
            tool = ShellTool(workspace=str(tmp_path))
            # Built-in pattern: rm -rf /
            result = await tool.run("rm -rf /")
            assert result["success"] is False
            assert result["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_blocklist_whitespace_handling(self, tmp_path) -> None:
        """Blocklist entries are trimmed of whitespace."""
        with patch.dict(
            os.environ, {"SHELL_BLOCKLIST": " curl , wget , nc "}, clear=False
        ):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("wget http://example.com/file")
            assert result["success"] is False
            assert "blocklist" in result["error"].lower()


class TestShellAllowlistBlocklistCombined:
    """When both SHELL_ALLOWLIST and SHELL_BLOCKLIST are set."""

    @pytest.mark.asyncio
    async def test_allowlist_and_blocklist_together(self, tmp_path) -> None:
        """Blocklist takes precedence: even if on allowlist, blocklist blocks."""
        with patch.dict(
            os.environ,
            {"SHELL_ALLOWLIST": "echo,curl", "SHELL_BLOCKLIST": "curl"},
            clear=False,
        ):
            tool = ShellTool(workspace=str(tmp_path))
            # curl is on both lists -- blocklist wins
            result = await tool.run("curl http://example.com")
            assert result["success"] is False
            assert "blocklist" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_allowlist_and_blocklist_permitted(self, tmp_path) -> None:
        """Command on allowlist and not on blocklist is permitted."""
        with patch.dict(
            os.environ,
            {"SHELL_ALLOWLIST": "echo,ls", "SHELL_BLOCKLIST": "curl"},
            clear=False,
        ):
            tool = ShellTool(workspace=str(tmp_path))
            result = await tool.run("echo safe")
            assert result["success"] is True
