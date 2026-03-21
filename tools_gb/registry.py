"""Tool registry -- maps Domain enum to tool instances."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from core_gb.types import Domain, ExecutionResult, TaskNode
from tools_gb.file import FileTool
from tools_gb.shell import ShellTool
from tools_gb.web import WebTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Maps domains to tool instances for DAG leaf execution."""

    def __init__(self, workspace: str | None = None) -> None:
        self._web = WebTool()
        self._file = FileTool(workspace=workspace)
        self._shell = ShellTool(workspace=workspace)
        self._domain_map: dict[Domain, Any] = {
            Domain.WEB: self._web,
            Domain.FILE: self._file,
            Domain.CODE: self._shell,
        }

    def has_tool(self, domain: Domain) -> bool:
        """Check if a domain has a registered tool."""
        return domain in self._domain_map

    async def execute(self, node: TaskNode) -> ExecutionResult:
        """Execute a leaf node using the appropriate tool.

        Parses the node description to determine which tool method to call.
        Falls back to returning an error for domains without tools.
        """
        start = time.perf_counter()
        domain = node.domain

        try:
            if domain == Domain.WEB:
                result_data = await self._execute_web(node)
            elif domain == Domain.FILE:
                result_data = self._execute_file(node)
            elif domain == Domain.CODE:
                result_data = await self._execute_shell(node)
            else:
                return self._no_tool_result(node, start)

            elapsed = (time.perf_counter() - start) * 1000
            output = (
                result_data.get("content", "")
                or result_data.get("stdout", "")
                or json.dumps(result_data)
            )
            success = result_data.get("success", False)

            errors: tuple[str, ...] = ()
            if result_data.get("error"):
                errors = (result_data["error"],)

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
        elif any(kw in desc for kw in ["edit", "replace", "change", "modify"]):
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
        cmd_match = re.search(r"[`'\"]([^`'\"]+)[`'\"]", node.description)
        command = cmd_match.group(1) if cmd_match else node.description
        return await self._shell.run(command)

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
