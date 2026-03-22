"""Planner-Grounder DAG pattern for browser workflows.

Provides a two-phase execution model for browser automation tasks:

1. **Planner**: Given a high-level web task description, generates a
   navigation plan -- an ordered list of browser actions (navigate, click,
   fill, extract_text, screenshot).

2. **Grounder**: Executes the plan step-by-step using a BrowserTool
   instance, collecting results and stopping on failure.

Plan steps can be converted to TaskNodes in a sequential sub-DAG for
integration with the DAGExecutor. BROWSER domain tasks are routed through
the planner-grounder pattern, where each action becomes an atomic leaf
node executed by the ToolRegistry.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core_gb.types import Domain, ExecutionResult, TaskNode

logger = logging.getLogger(__name__)

# Maps BrowserAction.action to the corresponding ToolRegistry tool_method.
_ACTION_TO_METHOD: dict[str, str] = {
    "navigate": "browser_navigate",
    "extract_text": "browser_extract_text",
    "screenshot": "browser_screenshot",
    "click": "browser_click",
    "fill": "browser_fill",
}


@dataclass
class BrowserAction:
    """A single browser action in a navigation plan.

    Attributes:
        action: The action type. One of: navigate, extract_text, screenshot,
            click, fill.
        params: Parameters for the action. Keys depend on the action type:
            - navigate: {url}
            - extract_text: {selector}
            - screenshot: {path} (optional)
            - click: {selector}
            - fill: {selector, value}
        description: Human-readable description of what this step does.
    """

    action: str
    params: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class BrowserPlan:
    """An ordered navigation plan for a browser workflow.

    Attributes:
        task: The original high-level task description.
        steps: Ordered list of BrowserAction steps to execute.
    """

    task: str
    steps: list[BrowserAction] = field(default_factory=list)

    def to_task_nodes(self, parent_id: str) -> list[TaskNode]:
        """Convert plan steps to a sequential sub-DAG of TaskNodes.

        Each step becomes an atomic BROWSER-domain TaskNode with the
        appropriate tool_method and tool_params set for direct execution
        by the ToolRegistry. Steps are chained sequentially: each node
        requires its predecessor.

        Args:
            parent_id: The ID of the parent node that spawned this plan.

        Returns:
            Ordered list of TaskNodes forming a sequential sub-DAG.
        """
        if not self.steps:
            return []

        nodes: list[TaskNode] = []
        for i, step in enumerate(self.steps):
            node_id = f"{parent_id}_step_{i}_{uuid.uuid4().hex[:8]}"
            tool_method = _ACTION_TO_METHOD.get(step.action, "")

            requires: list[str] = []
            if i > 0:
                requires = [nodes[i - 1].id]

            node = TaskNode(
                id=node_id,
                description=step.description or f"{step.action}: {step.params}",
                parent_id=parent_id,
                is_atomic=True,
                domain=Domain.BROWSER,
                tool_method=tool_method,
                tool_params=dict(step.params),
                requires=requires,
                complexity=1,
            )
            nodes.append(node)

        return nodes


class BrowserPlanner:
    """Planner-Grounder for browser automation workflows.

    The planner parses a high-level task description and generates a
    BrowserPlan -- an ordered list of browser actions. The grounder
    executes the plan step-by-step using a BrowserTool instance.

    For DAG integration, the planner can convert a plan into TaskNodes
    that the DAGExecutor dispatches sequentially through the ToolRegistry.
    """

    def plan(self, task: str) -> BrowserPlan:
        """Generate a navigation plan from a task description.

        Parses the task to identify:
        - URLs to navigate to
        - Click targets (CSS selectors)
        - Form fields to fill (selector + value)
        - Screenshot requests
        - Text extraction targets

        Args:
            task: High-level description of the browser workflow.

        Returns:
            A BrowserPlan with ordered steps.
        """
        steps: list[BrowserAction] = []
        desc_lower = task.lower()

        # Extract all URLs
        urls = re.findall(r"https?://[^\s,)\"']+", task)

        # Detect action keywords
        has_click = any(kw in desc_lower for kw in ["click"])
        has_fill = any(kw in desc_lower for kw in ["fill", "type into", "enter into"])
        has_screenshot = any(kw in desc_lower for kw in ["screenshot", "capture"])
        has_extract = any(
            kw in desc_lower
            for kw in ["extract", "get the", "get text", "read text", "page text"]
        )

        # Step 1: Navigate to URL if present
        if urls:
            steps.append(BrowserAction(
                action="navigate",
                params={"url": urls[0]},
                description=f"Navigate to {urls[0]}",
            ))

        # Step 2: Fill form fields if requested
        if has_fill:
            fill_matches = re.finditer(
                r"(?:fill|type into|enter into)\s+['\"]([^'\"]+)['\"]\s+(?:with|value)\s+['\"]([^'\"]+)['\"]",
                task,
                re.IGNORECASE,
            )
            for match in fill_matches:
                selector = match.group(1)
                value = match.group(2)
                steps.append(BrowserAction(
                    action="fill",
                    params={"selector": selector, "value": value},
                    description=f"Fill {selector} with '{value}'",
                ))

        # Step 3: Click elements if requested
        if has_click:
            click_matches = re.finditer(
                r"click\s+['\"]([^'\"]+)['\"]",
                task,
                re.IGNORECASE,
            )
            for match in click_matches:
                selector = match.group(1)
                steps.append(BrowserAction(
                    action="click",
                    params={"selector": selector},
                    description=f"Click {selector}",
                ))

        # Step 4: Extract text if requested
        if has_extract:
            extract_match = re.search(
                r"(?:extract|get)\s+(?:text\s+)?(?:from\s+)?['\"]([^'\"]+)['\"]",
                task,
                re.IGNORECASE,
            )
            selector = extract_match.group(1) if extract_match else "body"
            steps.append(BrowserAction(
                action="extract_text",
                params={"selector": selector},
                description=f"Extract text from {selector}",
            ))

        # Step 5: Screenshot if requested
        if has_screenshot:
            steps.append(BrowserAction(
                action="screenshot",
                params={},
                description="Take a screenshot",
            ))

        # Default: if no specific action was detected and we navigated,
        # add an extract_text step
        if urls and not (has_click or has_fill or has_screenshot or has_extract):
            steps.append(BrowserAction(
                action="extract_text",
                params={"selector": "body"},
                description="Extract page text",
            ))

        # If no URL and no actions, default to extracting text from current page
        if not steps:
            steps.append(BrowserAction(
                action="extract_text",
                params={"selector": "body"},
                description="Extract text from current page",
            ))

        logger.info(
            "Generated browser plan with %d steps for task: %s",
            len(steps),
            task[:80],
        )

        return BrowserPlan(task=task, steps=steps)

    async def ground(
        self,
        plan: BrowserPlan,
        browser_tool: Any,
    ) -> dict[str, Any]:
        """Execute a navigation plan step-by-step using a BrowserTool.

        Each step is dispatched to the corresponding BrowserTool method.
        Execution stops on the first failing step. Results from all
        executed steps are collected.

        Args:
            plan: The BrowserPlan to execute.
            browser_tool: A BrowserTool instance (or mock with the same interface).

        Returns:
            Dict with keys:
            - success: bool -- True if all steps succeeded.
            - output: str -- Combined output text from all steps.
            - error: str -- Error message if a step failed (empty on success).
            - step_results: list[dict] -- Per-step results.
        """
        if not plan.steps:
            return {
                "success": True,
                "output": "",
                "error": "",
                "step_results": [],
            }

        step_results: list[dict[str, Any]] = []
        output_parts: list[str] = []

        for i, step in enumerate(plan.steps):
            logger.info(
                "Grounding step %d/%d: %s %s",
                i + 1, len(plan.steps), step.action, step.params,
            )

            result = await self._dispatch_action(step, browser_tool)
            step_results.append(result)

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                logger.warning(
                    "Step %d/%d failed: %s",
                    i + 1, len(plan.steps), error_msg,
                )
                return {
                    "success": False,
                    "output": "\n".join(output_parts),
                    "error": error_msg,
                    "step_results": step_results,
                    "failed_step": i,
                }

            # Collect output from meaningful result fields
            if "text" in result:
                output_parts.append(result["text"])
            elif "title" in result:
                output_parts.append(result["title"])

        return {
            "success": True,
            "output": "\n".join(output_parts),
            "error": "",
            "step_results": step_results,
        }

    async def plan_and_ground(
        self,
        task: str,
        browser_tool: Any,
    ) -> dict[str, Any]:
        """Generate a plan and execute it in one call.

        Convenience method that combines plan() and ground().

        Args:
            task: High-level description of the browser workflow.
            browser_tool: A BrowserTool instance for execution.

        Returns:
            Dict with the same keys as ground(), plus:
            - plan: The generated BrowserPlan.
        """
        plan = self.plan(task)
        result = await self.ground(plan, browser_tool)
        result["plan"] = plan
        return result

    async def execute(
        self,
        node: TaskNode,
        browser_tool: Any,
    ) -> ExecutionResult:
        """Execute a BROWSER domain TaskNode through the planner-grounder.

        This is the integration point for the ToolRegistry: instead of
        dispatching browser tasks via description parsing, they are
        routed through the planner-grounder pattern for structured,
        step-by-step execution.

        Args:
            node: The BROWSER domain TaskNode to execute.
            browser_tool: A BrowserTool instance for execution.

        Returns:
            An ExecutionResult for integration with the DAG executor.
        """
        start = time.perf_counter()

        try:
            result = await self.plan_and_ground(node.description, browser_tool)
            elapsed = (time.perf_counter() - start) * 1000

            output = result.get("output", "")
            success = result.get("success", False)
            errors: tuple[str, ...] = ()
            if result.get("error"):
                errors = (result["error"],)

            plan: BrowserPlan | None = result.get("plan")
            step_count = len(plan.steps) if plan else 0

            return ExecutionResult(
                root_id=node.id,
                output=output,
                success=success,
                total_nodes=step_count,
                total_tokens=0,
                total_latency_ms=elapsed,
                total_cost=0.0,
                model_used="tool:browser_planner",
                errors=errors,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Browser planner execution failed for node %s: %s",
                node.id, exc,
            )
            return ExecutionResult(
                root_id=node.id,
                output="",
                success=False,
                total_nodes=0,
                total_tokens=0,
                total_latency_ms=elapsed,
                total_cost=0.0,
                model_used="tool:browser_planner",
                errors=(str(exc),),
            )

    async def _dispatch_action(
        self,
        step: BrowserAction,
        browser_tool: Any,
    ) -> dict[str, Any]:
        """Dispatch a single BrowserAction to the appropriate BrowserTool method.

        Args:
            step: The action to execute.
            browser_tool: The BrowserTool instance.

        Returns:
            The result dict from the BrowserTool method.
        """
        action = step.action
        params = step.params

        if action == "navigate":
            return await browser_tool.navigate(params.get("url", ""))
        elif action == "extract_text":
            return await browser_tool.extract_text(params.get("selector", "body"))
        elif action == "screenshot":
            return await browser_tool.screenshot(params.get("path"))
        elif action == "click":
            return await browser_tool.click(params.get("selector", ""))
        elif action == "fill":
            return await browser_tool.fill(
                params.get("selector", ""),
                params.get("value", ""),
            )
        else:
            return {
                "success": False,
                "error": f"Unknown browser action: {action}",
            }
