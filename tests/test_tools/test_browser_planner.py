"""Tests for tools_gb.browser_planner -- Planner-Grounder DAG pattern for browser workflows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import Domain, ExecutionResult, TaskNode, TaskStatus
from tools_gb.browser_planner import (
    BrowserAction,
    BrowserPlan,
    BrowserPlanner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_page() -> AsyncMock:
    """Build a mock Playwright page with standard async methods."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value="<html><body><p>Hello World</p></body></html>")
    page.inner_text = AsyncMock(return_value="Hello World")
    page.screenshot = AsyncMock(return_value=b"\x89PNG fake screenshot bytes")
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")
    page.wait_for_load_state = AsyncMock()
    page.close = AsyncMock()
    page.set_default_timeout = MagicMock()
    return page


def _mock_browser_tool(page: AsyncMock | None = None) -> AsyncMock:
    """Build a mock BrowserTool that returns structured results."""
    tool = AsyncMock()
    tool.navigate = AsyncMock(return_value={
        "success": True, "url": "https://example.com", "title": "Example Page",
    })
    tool.extract_text = AsyncMock(return_value={
        "success": True, "text": "Hello World", "url": "https://example.com",
    })
    tool.screenshot = AsyncMock(return_value={
        "success": True, "data": b"\x89PNG", "path": None, "url": "https://example.com",
    })
    tool.click = AsyncMock(return_value={
        "success": True, "selector": "button#submit", "url": "https://example.com",
    })
    tool.fill = AsyncMock(return_value={
        "success": True, "selector": "input#email", "url": "https://example.com",
    })
    tool.close = AsyncMock()
    tool.audit = MagicMock()
    tool.audit.entries = []
    return tool


# ---------------------------------------------------------------------------
# Test: BrowserAction dataclass
# ---------------------------------------------------------------------------


class TestBrowserAction:
    def test_action_creation(self) -> None:
        """BrowserAction stores action type and params."""
        action = BrowserAction(
            action="navigate",
            params={"url": "https://example.com"},
            description="Navigate to example.com",
        )
        assert action.action == "navigate"
        assert action.params == {"url": "https://example.com"}
        assert action.description == "Navigate to example.com"

    def test_action_default_description(self) -> None:
        """BrowserAction has empty description by default."""
        action = BrowserAction(action="click", params={"selector": "#btn"})
        assert action.description == ""


# ---------------------------------------------------------------------------
# Test: BrowserPlan dataclass
# ---------------------------------------------------------------------------


class TestBrowserPlan:
    def test_plan_creation(self) -> None:
        """BrowserPlan holds a task description and ordered list of steps."""
        steps = [
            BrowserAction(action="navigate", params={"url": "https://example.com"}),
            BrowserAction(action="extract_text", params={"selector": "body"}),
        ]
        plan = BrowserPlan(task="Get text from example.com", steps=steps)
        assert plan.task == "Get text from example.com"
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "navigate"
        assert plan.steps[1].action == "extract_text"

    def test_plan_to_task_nodes(self) -> None:
        """BrowserPlan.to_task_nodes converts steps to a sequential sub-DAG."""
        steps = [
            BrowserAction(
                action="navigate",
                params={"url": "https://example.com"},
                description="Navigate to example.com",
            ),
            BrowserAction(
                action="extract_text",
                params={"selector": "#content"},
                description="Extract text from content div",
            ),
            BrowserAction(
                action="screenshot",
                params={},
                description="Take a screenshot",
            ),
        ]
        plan = BrowserPlan(task="Get content from example.com", steps=steps)
        nodes = plan.to_task_nodes(parent_id="parent_001")

        assert len(nodes) == 3

        # First node has no requires (entry point)
        assert nodes[0].requires == []
        assert nodes[0].domain == Domain.BROWSER
        assert nodes[0].is_atomic is True
        assert nodes[0].tool_method == "browser_navigate"
        assert nodes[0].tool_params == {"url": "https://example.com"}

        # Second node requires the first
        assert nodes[1].requires == [nodes[0].id]
        assert nodes[1].tool_method == "browser_extract_text"
        assert nodes[1].tool_params == {"selector": "#content"}

        # Third node requires the second
        assert nodes[2].requires == [nodes[1].id]
        assert nodes[2].tool_method == "browser_screenshot"

        # All nodes share the parent
        for node in nodes:
            assert node.parent_id == "parent_001"

    def test_plan_to_task_nodes_empty(self) -> None:
        """Empty plan produces empty node list."""
        plan = BrowserPlan(task="Nothing to do", steps=[])
        nodes = plan.to_task_nodes(parent_id="p")
        assert nodes == []


# ---------------------------------------------------------------------------
# Test: BrowserPlanner.plan -- generate navigation plan
# ---------------------------------------------------------------------------


class TestPlannerPlan:
    def test_plan_navigate_and_extract(self) -> None:
        """Planner generates navigate + extract_text for a simple URL task."""
        planner = BrowserPlanner()
        plan = planner.plan("Go to https://example.com and get the page text")

        assert isinstance(plan, BrowserPlan)
        assert len(plan.steps) >= 2

        # First step should be navigate
        assert plan.steps[0].action == "navigate"
        assert plan.steps[0].params["url"] == "https://example.com"

        # Should have an extract_text step
        extract_steps = [s for s in plan.steps if s.action == "extract_text"]
        assert len(extract_steps) >= 1

    def test_plan_navigate_click_extract(self) -> None:
        """Planner generates navigate + click + extract for a click task."""
        planner = BrowserPlanner()
        plan = planner.plan(
            "Go to https://example.com, click 'button#submit', then extract text from '#result'"
        )

        assert len(plan.steps) >= 3
        actions = [s.action for s in plan.steps]
        assert "navigate" in actions
        assert "click" in actions
        assert "extract_text" in actions

    def test_plan_navigate_fill(self) -> None:
        """Planner generates navigate + fill for a form fill task."""
        planner = BrowserPlanner()
        plan = planner.plan(
            "Go to https://example.com/login, fill 'input#email' with 'user@test.com'"
        )

        assert len(plan.steps) >= 2
        actions = [s.action for s in plan.steps]
        assert "navigate" in actions
        assert "fill" in actions

        fill_step = next(s for s in plan.steps if s.action == "fill")
        assert fill_step.params["selector"] == "input#email"
        assert fill_step.params["value"] == "user@test.com"

    def test_plan_screenshot(self) -> None:
        """Planner generates navigate + screenshot for a screenshot task."""
        planner = BrowserPlanner()
        plan = planner.plan("Take a screenshot of https://example.com")

        actions = [s.action for s in plan.steps]
        assert "navigate" in actions
        assert "screenshot" in actions

    def test_plan_no_url_returns_extract(self) -> None:
        """Planner with no URL generates extract_text on current page."""
        planner = BrowserPlanner()
        plan = planner.plan("Get the text from the page")

        assert len(plan.steps) >= 1
        assert plan.steps[0].action == "extract_text"

    def test_plan_multiple_urls(self) -> None:
        """Planner with multiple URLs navigates to the first one."""
        planner = BrowserPlanner()
        plan = planner.plan(
            "Navigate to https://example.com then go to https://other.com"
        )

        nav_steps = [s for s in plan.steps if s.action == "navigate"]
        # Should navigate to at least the first URL
        assert len(nav_steps) >= 1
        assert nav_steps[0].params["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# Test: BrowserPlanner.ground -- execute plan steps
# ---------------------------------------------------------------------------


class TestPlannerGround:
    async def test_ground_success(self) -> None:
        """Grounder executes all plan steps and returns aggregated result."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()
        plan = BrowserPlan(
            task="Get text from example.com",
            steps=[
                BrowserAction(
                    action="navigate",
                    params={"url": "https://example.com"},
                ),
                BrowserAction(
                    action="extract_text",
                    params={"selector": "body"},
                ),
            ],
        )

        result = await planner.ground(plan, browser_tool)

        assert result["success"] is True
        assert "Hello World" in result["output"]
        browser_tool.navigate.assert_awaited_once_with("https://example.com")
        browser_tool.extract_text.assert_awaited_once_with("body")

    async def test_ground_stops_on_failure(self) -> None:
        """Grounder stops execution on the first failing step."""
        browser_tool = _mock_browser_tool()
        browser_tool.navigate = AsyncMock(return_value={
            "success": False, "error": "net::ERR_NAME_NOT_RESOLVED",
            "url": "https://bad.example.com",
        })

        planner = BrowserPlanner()
        plan = BrowserPlan(
            task="Get text from bad site",
            steps=[
                BrowserAction(
                    action="navigate",
                    params={"url": "https://bad.example.com"},
                ),
                BrowserAction(
                    action="extract_text",
                    params={"selector": "body"},
                ),
            ],
        )

        result = await planner.ground(plan, browser_tool)

        assert result["success"] is False
        assert "ERR_NAME_NOT_RESOLVED" in result["error"]
        # extract_text should NOT have been called
        browser_tool.extract_text.assert_not_awaited()

    async def test_ground_click_step(self) -> None:
        """Grounder correctly dispatches click actions."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()
        plan = BrowserPlan(
            task="Click a button",
            steps=[
                BrowserAction(action="click", params={"selector": "button#go"}),
            ],
        )

        result = await planner.ground(plan, browser_tool)

        assert result["success"] is True
        browser_tool.click.assert_awaited_once_with("button#go")

    async def test_ground_fill_step(self) -> None:
        """Grounder correctly dispatches fill actions."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()
        plan = BrowserPlan(
            task="Fill a form",
            steps=[
                BrowserAction(
                    action="fill",
                    params={"selector": "input#name", "value": "Alice"},
                ),
            ],
        )

        result = await planner.ground(plan, browser_tool)

        assert result["success"] is True
        browser_tool.fill.assert_awaited_once_with("input#name", "Alice")

    async def test_ground_screenshot_step(self) -> None:
        """Grounder correctly dispatches screenshot actions."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()
        plan = BrowserPlan(
            task="Take a screenshot",
            steps=[
                BrowserAction(action="screenshot", params={}),
            ],
        )

        result = await planner.ground(plan, browser_tool)

        assert result["success"] is True
        browser_tool.screenshot.assert_awaited_once()

    async def test_ground_collects_step_results(self) -> None:
        """Grounder collects all step results in the output."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()
        plan = BrowserPlan(
            task="Navigate and extract",
            steps=[
                BrowserAction(
                    action="navigate",
                    params={"url": "https://example.com"},
                ),
                BrowserAction(action="extract_text", params={"selector": "body"}),
            ],
        )

        result = await planner.ground(plan, browser_tool)

        assert result["success"] is True
        assert len(result["step_results"]) == 2
        assert result["step_results"][0]["success"] is True
        assert result["step_results"][1]["success"] is True

    async def test_ground_empty_plan(self) -> None:
        """Grounder with empty plan returns success with no output."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()
        plan = BrowserPlan(task="Nothing", steps=[])

        result = await planner.ground(plan, browser_tool)

        assert result["success"] is True
        assert result["output"] == ""


# ---------------------------------------------------------------------------
# Test: BrowserPlanner.plan_and_ground -- end-to-end
# ---------------------------------------------------------------------------


class TestPlanAndGround:
    async def test_plan_and_ground_full_workflow(self) -> None:
        """plan_and_ground generates plan then executes it."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()

        result = await planner.plan_and_ground(
            "Go to https://example.com and get the page text",
            browser_tool,
        )

        assert result["success"] is True
        assert "plan" in result
        assert isinstance(result["plan"], BrowserPlan)
        browser_tool.navigate.assert_awaited_once()
        browser_tool.extract_text.assert_awaited()


# ---------------------------------------------------------------------------
# Test: BrowserPlanner.execute_as_dag -- DAG integration
# ---------------------------------------------------------------------------


class TestExecuteAsDag:
    def test_plan_generates_sub_dag_nodes(self) -> None:
        """Plan steps are converted to sequential TaskNodes for the DAG executor."""
        planner = BrowserPlanner()
        plan = planner.plan("Go to https://example.com and extract text")

        nodes = plan.to_task_nodes(parent_id="browser_task_001")

        # All nodes are atomic, in the BROWSER domain
        for node in nodes:
            assert node.is_atomic is True
            assert node.domain == Domain.BROWSER
            assert node.parent_id == "browser_task_001"

        # Sequential chain: each node requires its predecessor
        for i in range(1, len(nodes)):
            assert nodes[i].requires == [nodes[i - 1].id]

    def test_sub_dag_nodes_have_tool_routing(self) -> None:
        """Sub-DAG nodes have proper tool_method and tool_params for direct execution."""
        planner = BrowserPlanner()
        plan = BrowserPlan(
            task="Login flow",
            steps=[
                BrowserAction(
                    action="navigate",
                    params={"url": "https://example.com/login"},
                ),
                BrowserAction(
                    action="fill",
                    params={"selector": "input#email", "value": "user@test.com"},
                ),
                BrowserAction(
                    action="click",
                    params={"selector": "button#login"},
                ),
                BrowserAction(
                    action="extract_text",
                    params={"selector": "#welcome"},
                ),
            ],
        )

        nodes = plan.to_task_nodes(parent_id="login_task")

        assert nodes[0].tool_method == "browser_navigate"
        assert nodes[0].tool_params == {"url": "https://example.com/login"}

        assert nodes[1].tool_method == "browser_fill"
        assert nodes[1].tool_params == {"selector": "input#email", "value": "user@test.com"}

        assert nodes[2].tool_method == "browser_click"
        assert nodes[2].tool_params == {"selector": "button#login"}

        assert nodes[3].tool_method == "browser_extract_text"
        assert nodes[3].tool_params == {"selector": "#welcome"}


# ---------------------------------------------------------------------------
# Test: Integration with ToolRegistry routing
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    async def test_browser_planner_execute(self) -> None:
        """BrowserPlanner.execute wraps plan_and_ground into an ExecutionResult."""
        browser_tool = _mock_browser_tool()
        planner = BrowserPlanner()

        node = TaskNode(
            id="browser_task_001",
            description="Go to https://example.com and get the page text",
            is_atomic=True,
            domain=Domain.BROWSER,
        )

        result = await planner.execute(node, browser_tool)

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.root_id == "browser_task_001"
        assert result.model_used == "tool:browser_planner"
        browser_tool.navigate.assert_awaited()

    async def test_browser_planner_execute_failure(self) -> None:
        """BrowserPlanner.execute returns failure result when grounding fails."""
        browser_tool = _mock_browser_tool()
        browser_tool.navigate = AsyncMock(return_value={
            "success": False,
            "error": "Policy violation: blocked domain",
            "url": "https://evil.com",
        })

        planner = BrowserPlanner()
        node = TaskNode(
            id="browser_task_002",
            description="Go to https://evil.com and get content",
            is_atomic=True,
            domain=Domain.BROWSER,
        )

        result = await planner.execute(node, browser_tool)

        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert len(result.errors) > 0
