"""Tool registry -- maps Domain enum to tool instances."""

from __future__ import annotations

import copy
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from core_gb.types import Domain, ExecutionResult, TaskNode
from tools_gb.browser import BrowserTool
from tools_gb.browser_planner import BrowserPlanner
from tools_gb.file import FileTool
from tools_gb.shell import ShellTool
from tools_gb.web import WebTool

logger = logging.getLogger(__name__)

# Minimum success rate before a tool is flagged as degraded.
_DEGRADED_THRESHOLD: float = 0.5


@dataclass
class ToolStats:
    """Per-tool quality statistics tracked across executions."""

    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0

    @property
    def total_count(self) -> int:
        """Total number of executions (success + failure)."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Fraction of executions that succeeded (0.0 to 1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    @property
    def is_degraded(self) -> bool:
        """True when success rate drops below the degraded threshold.

        Tools with zero executions are not considered degraded.
        """
        if self.total_count == 0:
            return False
        return self.success_rate < _DEGRADED_THRESHOLD


class ToolRegistry:
    """Maps domains to tool instances for DAG leaf execution."""

    def __init__(self, workspace: str | None = None, router: Any | None = None) -> None:
        self._web = WebTool()
        self._file = FileTool(workspace=workspace)
        self._shell = ShellTool(workspace=workspace)
        self._browser = BrowserTool()
        self._domain_map: dict[Domain, Any] = {
            Domain.WEB: self._web,
            Domain.FILE: self._file,
            Domain.CODE: self._shell,
            Domain.BROWSER: self._browser,
        }
        self._browser_planner = BrowserPlanner()
        self._stats: dict[str, ToolStats] = {}
        self._code_agent: Any | None = None
        self._tool_factory: Any | None = None
        if router is not None:
            from core_gb.code_agent import CodeEditAgent
            self._code_agent = CodeEditAgent(self._file, self._shell, router)

    def set_tool_factory(self, factory: Any) -> None:
        """Attach a ToolFactory so generated tools are accessible via the registry."""
        self._tool_factory = factory

    def has_tool(self, domain: Domain) -> bool:
        """Check if a domain has a registered tool."""
        return domain in self._domain_map

    def has_generated_tool(self, name: str) -> bool:
        """Check if a dynamically generated tool exists by name.

        Looks up the attached ToolFactory's in-memory registry.
        Returns False if no ToolFactory is attached.
        """
        if self._tool_factory is None:
            return False
        return self._tool_factory.get_tool(name) is not None

    # -- Quality tracking ----------------------------------------------------

    def _record_stat(self, tool_key: str, success: bool, latency_ms: float) -> None:
        """Record the outcome of a single tool execution.

        Updates success/failure counts and recalculates the running average
        latency.  If the tool's success rate drops below the degraded
        threshold, a warning is logged.
        """
        if tool_key not in self._stats:
            self._stats[tool_key] = ToolStats()

        stats = self._stats[tool_key]
        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

        # Running average: avg = ((avg * (n-1)) + new) / n
        n = stats.total_count
        stats.avg_latency_ms = (
            (stats.avg_latency_ms * (n - 1) + latency_ms) / n
        )

        if stats.is_degraded:
            logger.warning(
                "Tool '%s' has a low success rate: %.0f%% (%d/%d executions)",
                tool_key,
                stats.success_rate * 100,
                stats.success_count,
                stats.total_count,
            )

    def get_stats(self) -> dict[str, ToolStats]:
        """Return a copy of all tool quality statistics.

        The returned dict is a shallow copy -- callers can mutate it without
        affecting the registry's internal state.  The ToolStats values are
        copied via ``copy.copy`` to ensure isolation.
        """
        return {k: copy.copy(v) for k, v in self._stats.items()}

    def check_tool_stats(self, report: Any) -> None:
        """Append tool quality results to a HealthReport.

        Meant to be called from the healthcheck script.  Each tracked tool
        produces one result entry.  Degraded tools are flagged as failures
        (non-critical).
        """
        from scripts.healthcheck import CheckResult

        stats = self.get_stats()
        if not stats:
            report.add(CheckResult(
                name="Tool quality stats",
                passed=True,
                message="No tool executions recorded yet",
                critical=False,
            ))
            return

        for tool_key, ts in stats.items():
            passed = not ts.is_degraded
            report.add(CheckResult(
                name=f"Tool quality: {tool_key}",
                passed=passed,
                message=(
                    f"{tool_key} success rate {ts.success_rate:.0%} "
                    f"({ts.success_count}/{ts.total_count} executions, "
                    f"avg latency {ts.avg_latency_ms:.1f}ms)"
                ),
                critical=False,
            ))

    # -- Execution -----------------------------------------------------------

    async def execute(self, node: TaskNode) -> ExecutionResult:
        """Execute a leaf node using the appropriate tool.

        If node.tool_method is set, routes directly via structured params.
        Otherwise falls back to domain-based description parsing.
        """
        if node.tool_method:
            result = await self._execute_by_method(node)
            if result is not None:
                return result

        start = time.perf_counter()
        domain = node.domain

        try:
            if domain == Domain.WEB:
                result_data = await self._execute_web(node)
            elif domain == Domain.FILE:
                result_data = self._execute_file(node)
                # Handle code agent sentinel from _execute_file
                if result_data.get("_code_agent"):
                    agent_result = await self._code_agent.edit(
                        result_data["instruction"], result_data["path"],
                    )
                    return agent_result
            elif domain == Domain.CODE:
                result_data = await self._execute_shell(node)
            elif domain == Domain.BROWSER:
                result_data = await self._execute_browser(node)
            else:
                result = self._no_tool_result(node, start)
                elapsed = result.total_latency_ms
                self._record_stat(domain.value, False, elapsed)
                return result

            elapsed = (time.perf_counter() - start) * 1000
            output = (
                result_data.get("content", "")
                or result_data.get("answer", "")
                or result_data.get("stdout", "")
                or json.dumps(result_data)
            )
            success = result_data.get("success", False)

            errors: tuple[str, ...] = ()
            if result_data.get("error"):
                errors = (result_data["error"],)

            self._record_stat(domain.value, success, elapsed)

            return ExecutionResult(
                root_id=node.id,
                output=output,
                success=success,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=elapsed,
                total_cost=0.0,
                model_used="tool:" + domain.value,
                errors=errors,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Tool execution failed for node %s: %s", node.id, exc)
            self._record_stat(domain.value, False, elapsed)
            return ExecutionResult(
                root_id=node.id,
                output="",
                success=False,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=elapsed,
                total_cost=0.0,
                model_used="tool:" + domain.value,
                errors=(str(exc),),
            )

    async def _execute_by_method(self, node: TaskNode) -> ExecutionResult | None:
        """Execute using explicit tool_method and tool_params."""
        method = node.tool_method
        params = node.tool_params
        start = time.perf_counter()

        try:
            result_data: dict[str, Any] | None = None

            if method == "file_read":
                result_data = self._file.read(params.get("path", ""))
            elif method == "file_list":
                result_data = self._file.list_dir(
                    params.get("directory", "."), params.get("pattern", "*")
                )
            elif method == "file_search":
                result_data = self._file.search(
                    params.get("directory", "."), params.get("query", "")
                )
            elif method == "web_search":
                result_data = await self._web.search(params.get("query", ""))
            elif method == "web_fetch":
                result_data = await self._web.fetch(params.get("url", ""))
            elif method == "shell_run":
                command = params.get("command", "")
                if not command:
                    command = self._extract_command(node.description)
                result_data = await self._shell.run(command)
            elif method == "browser_navigate":
                result_data = await self._browser.navigate(params.get("url", ""))
            elif method == "browser_extract_text":
                result_data = await self._browser.extract_text(
                    params.get("selector", "body")
                )
            elif method == "browser_screenshot":
                result_data = await self._browser.screenshot(
                    params.get("path")
                )
            elif method == "browser_click":
                result_data = await self._browser.click(params.get("selector", ""))
            elif method == "browser_fill":
                result_data = await self._browser.fill(
                    params.get("selector", ""), params.get("value", "")
                )
            elif method == "llm_reason":
                return None
            else:
                return None

            elapsed = (time.perf_counter() - start) * 1000
            output = (
                result_data.get("content", "")
                or result_data.get("answer", "")
                or result_data.get("stdout", "")
                or json.dumps(result_data)
            )
            success = result_data.get("success", False)
            errors: tuple[str, ...] = ()
            if result_data.get("error"):
                errors = (result_data["error"],)

            self._record_stat(method, success, elapsed)

            return ExecutionResult(
                root_id=node.id,
                output=output,
                success=success,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=elapsed,
                total_cost=0.0,
                model_used=f"tool:{method}",
                errors=errors,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Tool method %s failed for node %s: %s", method, node.id, exc)
            self._record_stat(method, False, elapsed)
            return ExecutionResult(
                root_id=node.id,
                output="",
                success=False,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=elapsed,
                total_cost=0.0,
                model_used=f"tool:{method}",
                errors=(str(exc),),
            )

    async def _execute_web(self, node: TaskNode) -> dict[str, Any]:
        """Route web tasks to appropriate WebTool method."""
        desc = node.description.lower()
        url_match = re.search(r"https?://[^\s]+", node.description)

        if url_match:
            return await self._web.fetch(url_match.group())
        elif any(kw in desc for kw in ["search", "find", "look up", "query"]):
            return await self._web.search(node.description)
        else:
            return await self._web.search_and_summarize(node.description)

    def _execute_file(self, node: TaskNode) -> dict[str, Any]:
        """Route file tasks to appropriate FileTool method."""
        desc = node.description.lower()
        path_match = re.search(
            r"['\"`]([^'\"`]+\.\w+)['\"`]|(\S+\.\w{1,5})", node.description
        )
        path = ""
        if path_match:
            path = path_match.group(1) or path_match.group(2) or ""

        if any(kw in desc for kw in ["read", "open", "show", "cat", "display"]):
            if path:
                return self._file.read(path)
            return {"success": False, "error": "No file path found"}
        elif any(kw in desc for kw in ["write", "create", "save"]):
            return {
                "success": False,
                "error": "Write requires content -- not supported as atomic leaf",
            }
        elif any(kw in desc for kw in ["edit", "replace", "change", "modify", "fix", "refactor", "update"]):
            if self._code_agent and path:
                # Defer to async code agent -- return sentinel for async handling
                return {"_code_agent": True, "path": path, "instruction": node.description}
            return {
                "success": False,
                "error": "Edit requires old/new text -- not supported as atomic leaf",
            }
        elif any(kw in desc for kw in ["list", "ls", "dir", "find files"]):
            target = path or "."
            return self._file.list_dir(target, "**/*")
        elif any(kw in desc for kw in ["search", "grep", "find"]):
            return self._file.search(".", node.description)
        else:
            return self._file.list_dir(".", "**/*")

    async def _execute_shell(self, node: TaskNode) -> dict[str, Any]:
        """Route code/shell tasks to ShellTool."""
        # If tool_params has a command, use it directly
        if node.tool_params and node.tool_params.get("command"):
            return await self._shell.run(node.tool_params["command"])

        # Extract command from description
        command = self._extract_command(node.description)
        return await self._shell.run(command)

    async def _execute_browser(self, node: TaskNode) -> dict[str, Any]:
        """Route browser tasks through the planner-grounder pattern.

        The BrowserPlanner generates a structured navigation plan from the
        task description, then the grounder executes each step sequentially
        using the BrowserTool. This replaces ad-hoc description parsing with
        a principled plan-then-execute approach.

        Returns the grounding result dict with success, output, and error keys.
        """
        result = await self._browser_planner.plan_and_ground(
            node.description, self._browser,
        )
        # Flatten the result to match the expected dict[str, Any] interface
        # that the execute() method consumes (it reads "success" and "error").
        output: dict[str, Any] = {
            "success": result.get("success", False),
            "content": result.get("output", ""),
        }
        if result.get("error"):
            output["error"] = result["error"]
        return output

    @staticmethod
    def _extract_command(description: str) -> str:
        """Extract shell command from a task description."""
        # Try backtick-quoted command first: `command here`
        backtick = re.search(r'`([^`]+)`', description)
        if backtick:
            return backtick.group(1)

        # Try single-quoted: 'command here'
        single = re.search(r"'([^']+)'", description)
        if single:
            return single.group(1)

        # Try double-quoted: "command here"
        double = re.search(r'"([^"]+)"', description)
        if double and not double.group(1).startswith("Run"):
            return double.group(1)

        # Strip common prefixes
        text = description.strip()
        prefixes = [
            "Run the command ", "Run command ", "Run ", "Execute ",
            "run the command ", "run command ", "run ", "execute ",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                # Remove trailing "and..." or "then..."
                for suffix in [" and ", " then ", " to ", " in order"]:
                    idx = text.find(suffix)
                    if idx > 0:
                        text = text[:idx]
                return text.strip().strip("'\"")

        return text

    def _no_tool_result(self, node: TaskNode, start: float) -> ExecutionResult:
        """Return error result for domains without tools."""
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            root_id=node.id,
            output="",
            success=False,
            total_nodes=1,
            total_tokens=0,
            total_latency_ms=elapsed,
            total_cost=0.0,
            errors=(f"No tool for domain: {node.domain.value}",),
        )
