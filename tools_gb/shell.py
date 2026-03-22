"""Shell execution tools for GraphBot DAG leaf execution."""

import asyncio
import logging
import os
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Commands that are always blocked
BLOCKED_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\s+/",     # rm -rf /
    r"\bdel\s+/[fq]\s+[A-Z]:\\",  # del /f C:\
    r"\bformat\b",
    r"\b(shutdown|reboot|poweroff|halt)\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{",                  # fork bomb
]

# Env vars to filter from child processes
SECRET_PATTERNS = [
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
]


def _parse_list_env(env_var: str) -> list[str]:
    """Parse a comma-separated environment variable into a list of trimmed strings.

    Returns an empty list if the variable is not set or empty.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _extract_base_command(command: str) -> str:
    """Extract the base command name from a shell command string.

    Handles leading whitespace, environment variable assignments, and
    paths (e.g., /usr/bin/python -> python).

    Returns:
        The base command name in lowercase.
    """
    # Strip leading env assignments like VAR=value cmd
    parts = command.strip().split()
    for part in parts:
        if "=" in part and not part.startswith("-"):
            continue
        # Strip path: /usr/bin/python -> python
        return part.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    return ""


class ShellTool:
    """Sandboxed shell execution for DAG leaf tasks."""

    def __init__(
        self,
        workspace: str | None = None,
        timeout: int = 30,
        max_output: int = 5000,
    ) -> None:
        self._workspace = workspace or str(os.getcwd())
        self._timeout = timeout
        self._max_output = max_output

    async def run(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        """Run a shell command and return the output.

        Returns: {success, stdout, stderr, exit_code, error}
        """
        effective_timeout = timeout or self._timeout

        # Security check
        blocked = self._check_blocked(command)
        if blocked:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "error": f"Blocked: {blocked}",
            }

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workspace,
                env=self._safe_env(),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1,
                    "error": f"Timeout after {effective_timeout}s",
                }

            stdout = self._clean_output(stdout_bytes.decode("utf-8", errors="replace"))
            stderr = self._clean_output(stderr_bytes.decode("utf-8", errors="replace"))
            exit_code = process.returncode or 0

            return {
                "success": exit_code == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "answer": self.interpret_output(command, stdout, stderr, exit_code),
            }

        except Exception as exc:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "error": str(exc),
            }

    def _check_blocked(self, command: str) -> str | None:
        """Check if command is blocked by patterns, blocklist, or allowlist.

        Evaluation order:
        1. Built-in BLOCKED_PATTERNS (always checked first)
        2. SHELL_BLOCKLIST from environment (custom blocklist)
        3. SHELL_ALLOWLIST from environment (if set, only listed commands run)

        Returns reason string if blocked, None if permitted.
        """
        # 1. Built-in dangerous pattern check
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Matches blocked pattern: {pattern}"

        base_cmd = _extract_base_command(command)

        # 2. Environment blocklist check
        blocklist = _parse_list_env("SHELL_BLOCKLIST")
        if blocklist:
            for blocked_cmd in blocklist:
                if blocked_cmd.lower() == base_cmd:
                    return (
                        f"Blocked by SHELL_BLOCKLIST: "
                        f"'{blocked_cmd}' is not permitted"
                    )

        # 3. Environment allowlist check
        allowlist = _parse_list_env("SHELL_ALLOWLIST")
        if allowlist:
            allowed_lower = [a.lower() for a in allowlist]
            if base_cmd not in allowed_lower:
                return (
                    f"Not on SHELL_ALLOWLIST: '{base_cmd}' "
                    f"is not in allowed commands "
                    f"({', '.join(allowlist)})"
                )

        return None

    def _safe_env(self) -> dict[str, str]:
        env = os.environ.copy()
        to_remove = []
        for key in env:
            for secret_pattern in SECRET_PATTERNS:
                if secret_pattern in key.upper():
                    to_remove.append(key)
                    break
        for key in to_remove:
            del env[key]
        return env

    @staticmethod
    def interpret_output(
        command: str, stdout: str, stderr: str, exit_code: int,
    ) -> str:
        """Parse shell stdout into a meaningful answer string.

        Extracts summary lines, counts, and key information from raw output
        so downstream nodes and the final aggregation receive a concise answer
        rather than raw terminal text.
        """
        if exit_code != 0:
            error_text = stderr.strip() or stdout.strip() or "unknown error"
            return f"Command failed (exit {exit_code}): {error_text}"

        lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return "Command completed successfully with no output."

        # pytest --co -q: last line is "N tests collected" or "N items"
        if "pytest" in command and ("--co" in command or "--collect-only" in command):
            for line in reversed(lines):
                if re.search(r"\d+\s+test", line, re.IGNORECASE):
                    return line.strip()
            # Fallback: count non-empty lines (each is a test id)
            test_lines = [ln for ln in lines if "::" in ln or "test_" in ln.lower()]
            if test_lines:
                return f"{len(test_lines)} tests collected"

        # git log: return all lines as-is (each is a commit summary)
        if "git log" in command:
            return f"{len(lines)} commits:\n" + "\n".join(lines)

        # Generic: if output is short (<=10 lines), return it all
        if len(lines) <= 10:
            return "\n".join(lines)

        # Longer output: return first 5 lines, count, and last 3 lines
        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-3:])
        return (
            f"{head}\n"
            f"... ({len(lines)} total lines) ...\n"
            f"{tail}"
        )

    def _clean_output(self, text: str) -> str:
        text = ANSI_ESCAPE.sub("", text)
        if len(text) > self._max_output:
            half = self._max_output // 2
            text = (
                text[:half]
                + f"\n\n... ({len(text) - self._max_output} chars truncated) ...\n\n"
                + text[-half:]
            )
        return text.strip()
