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
        """Check if command matches any blocked patterns. Returns reason or None."""
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Matches blocked pattern: {pattern}"
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
