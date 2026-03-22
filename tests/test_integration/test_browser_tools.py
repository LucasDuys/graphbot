"""Browser and tool expansion integration tests.

Validates end-to-end flows for:
1. Browser navigation with mocked Playwright: navigate to URL, extract text.
2. Policy guard: blocked domain rejected in the full pipeline.
3. Dynamic tool: task with no matching tool triggers ToolFactory, tool created and used.
4. Tool persistence: created tool survives ToolFactory reload from graph.
5. Tool quality: stats increment after execution through ToolRegistry.

Uses mocked providers and mocked Playwright throughout.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from core_gb.tool_factory import GeneratedTool, ToolFactory
from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter
from tools_gb.browser import BrowserTool
from tools_gb.browser_policy import BrowserPolicy
from tools_gb.registry import ToolRegistry, ToolStats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _completion(
    content: str,
    tokens_in: int = 10,
    tokens_out: int = 10,
    cost: float = 0.001,
) -> CompletionResult:
    """Build a minimal CompletionResult."""
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=50.0,
        cost=cost,
    )


class MockProvider(ModelProvider):
    """Provider that returns configurable responses based on call order."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses: list[CompletionResult] = list(responses)
        self._call_count: int = 0
        self.call_log: list[list[dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return "mock"

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: object
    ) -> CompletionResult:
        self.call_log.append(messages)
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def _mock_page(url: str = "https://example.com", title: str = "Example Page") -> AsyncMock:
    """Build a mock Playwright page with standard async methods."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value="<html><body><p>Hello World</p></body></html>")
    page.inner_text = AsyncMock(return_value="Hello World")
    page.screenshot = AsyncMock(return_value=b"\x89PNG fake")
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.url = url
    page.title = AsyncMock(return_value=title)
    page.wait_for_load_state = AsyncMock()
    page.close = AsyncMock()
    page.set_default_timeout = MagicMock()
    return page


def _inject_mocks_into_tool(tool: BrowserTool, page: AsyncMock) -> None:
    """Inject mock Playwright internals into a BrowserTool instance.

    Bypasses lazy browser launch so tests run without a real Playwright
    installation.
    """
    tool._playwright = AsyncMock()
    tool._browser = AsyncMock()
    tool._context = AsyncMock()
    tool._page = page


def _mock_router(response_content: str) -> MagicMock:
    """Return a mock ModelRouter whose route() returns the given content."""
    mock = MagicMock()
    mock.route = AsyncMock(return_value=CompletionResult(
        content=response_content,
        model="mock-model",
        tokens_in=20,
        tokens_out=50,
        latency_ms=100.0,
        cost=0.0,
    ))
    return mock


# ---------------------------------------------------------------------------
# 1. Browser navigation: navigate to URL, extract text (mocked playwright)
# ---------------------------------------------------------------------------


class TestBrowserNavigationIntegration:
    """Navigate to a URL and extract text using mocked Playwright.

    Verifies the full navigate -> extract_text flow returns correct data,
    URL and title are captured, and the audit log records both actions.
    """

    async def test_navigate_then_extract_text(self) -> None:
        """Full flow: navigate to a URL, then extract text from the page."""
        page = _mock_page(url="https://docs.python.org", title="Python Docs")
        page.inner_text = AsyncMock(return_value="Welcome to Python documentation")

        tool = BrowserTool()
        _inject_mocks_into_tool(tool, page)

        # Step 1: navigate
        nav_result = await tool.navigate("https://docs.python.org")
        assert nav_result["success"] is True
        assert nav_result["url"] == "https://docs.python.org"
        assert nav_result["title"] == "Python Docs"
        page.goto.assert_awaited_once_with(
            "https://docs.python.org", wait_until="domcontentloaded"
        )

        # Step 2: extract text from the page
        text_result = await tool.extract_text("body")
        assert text_result["success"] is True
        assert text_result["text"] == "Welcome to Python documentation"
        assert text_result["url"] == "https://docs.python.org"
        page.inner_text.assert_awaited_once_with("body")

        # Verify audit trail captured both actions
        assert len(tool.audit.entries) == 2
        assert tool.audit.entries[0]["action"] == "navigate"
        assert tool.audit.entries[0]["blocked"] is False
        assert tool.audit.entries[1]["action"] == "extract_text"
        assert tool.audit.entries[1]["blocked"] is False

        await tool.close()

    async def test_navigate_to_multiple_pages_reuses_browser(self) -> None:
        """Navigating to a second page reuses the same browser instance."""
        page = _mock_page()
        tool = BrowserTool()
        _inject_mocks_into_tool(tool, page)

        result_1 = await tool.navigate("https://example.com/page1")
        result_2 = await tool.navigate("https://example.com/page2")

        assert result_1["success"] is True
        assert result_2["success"] is True
        # page.goto called twice on the same page object
        assert page.goto.await_count == 2
        # Audit should show two navigate entries
        assert len(tool.audit.entries) == 2

        await tool.close()


# ---------------------------------------------------------------------------
# 2. Policy guard: blocked domain rejected in full pipeline
# ---------------------------------------------------------------------------


class TestPolicyGuardIntegration:
    """A blocked domain is rejected before any browser action executes.

    The pipeline flow is:
    1. Orchestrator creates a browser-domain TaskNode.
    2. ToolRegistry delegates to BrowserTool.
    3. BrowserTool checks BrowserPolicy.
    4. BrowserPolicy rejects the domain (blocklist match).
    5. No page navigation occurs, audit shows blocked entry.
    """

    async def test_blocked_domain_rejected_in_navigate(self) -> None:
        """Navigate to a blocklisted domain returns policy violation error."""
        tool = BrowserTool()
        tool._policy = BrowserPolicy(blocklist=["evil.com"])

        page = _mock_page(url="https://evil.com")
        _inject_mocks_into_tool(tool, page)

        result = await tool.navigate("https://evil.com/malware")

        # Navigation blocked by policy
        assert result["success"] is False
        assert "Policy violation" in result["error"]
        assert "blocklist" in result["error"].lower()

        # The page.goto should NOT have been called
        page.goto.assert_not_awaited()

        # Audit trail records the blocked action
        assert len(tool.audit.entries) == 1
        assert tool.audit.entries[0]["blocked"] is True
        assert "evil.com" in tool.audit.entries[0]["url"]

        await tool.close()

    async def test_blocked_domain_via_registry_execute(self) -> None:
        """Full pipeline: ToolRegistry.execute on browser-domain node with
        blocked URL returns a failed ExecutionResult.

        This wires together ToolRegistry -> BrowserTool -> BrowserPolicy.
        """
        registry = ToolRegistry()
        # Replace the registry's browser tool with one that has a blocklist
        blocked_tool = BrowserTool()
        blocked_tool._policy = BrowserPolicy(blocklist=["malware.org"])

        page = _mock_page(url="https://malware.org")
        _inject_mocks_into_tool(blocked_tool, page)
        registry._browser = blocked_tool
        registry._domain_map[Domain.BROWSER] = blocked_tool

        node = TaskNode(
            id="browser_node_1",
            description="Navigate to https://malware.org/payload and extract content",
            is_atomic=True,
            domain=Domain.BROWSER,
            complexity=1,
            status=TaskStatus.READY,
        )

        result = await registry.execute(node)

        # The execution should fail due to policy block
        assert result.success is False
        assert len(result.errors) > 0

        # No actual page navigation happened
        page.goto.assert_not_awaited()

        await blocked_tool.close()

    async def test_subdomain_of_blocked_domain_also_rejected(self) -> None:
        """Subdomains of a blocklisted domain are also rejected."""
        tool = BrowserTool()
        tool._policy = BrowserPolicy(blocklist=["evil.com"])
        page = _mock_page()
        _inject_mocks_into_tool(tool, page)

        result = await tool.navigate("https://sub.evil.com/page")

        assert result["success"] is False
        assert "Policy violation" in result["error"]
        page.goto.assert_not_awaited()

        await tool.close()

    async def test_allowlisted_domain_passes(self) -> None:
        """A domain on the allowlist is permitted for navigation."""
        tool = BrowserTool()
        tool._policy = BrowserPolicy(allowlist=["safe.org"])
        page = _mock_page(url="https://safe.org")
        _inject_mocks_into_tool(tool, page)

        result = await tool.navigate("https://safe.org/docs")

        assert result["success"] is True
        page.goto.assert_awaited_once()

        await tool.close()


# ---------------------------------------------------------------------------
# 3. Dynamic tool: task triggers ToolFactory, tool created and used
# ---------------------------------------------------------------------------


FIBONACCI_LLM_RESPONSE: str = textwrap.dedent("""\
    ```python
    def fibonacci(n: int) -> int:
        \"\"\"Compute the nth Fibonacci number iteratively.\"\"\"
        if n <= 0:
            return 0
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
    ```
    test_input: 10
    expected_output: 55
""")


class TestDynamicToolCreation:
    """When no existing tool matches a task, ToolFactory generates one.

    Flow:
    1. ToolFactory.find_tool returns None (no existing tool).
    2. ToolFactory.create_tool calls the LLM to generate code.
    3. The code is validated in a sandbox.
    4. The tool is registered for immediate use.
    5. The generated function returns correct results.
    """

    async def test_task_with_no_matching_tool_triggers_factory(self) -> None:
        """ToolFactory generates a new tool when none exists for the task."""
        store = _make_store()
        router = _mock_router(FIBONACCI_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        # Confirm no tool exists yet
        assert factory.find_tool("fibonacci") is None
        assert len(factory.list_tools()) == 0

        # Create tool via the full pipeline (LLM -> sandbox -> register)
        tool = await factory.create_tool(
            task_description="Compute the nth Fibonacci number"
        )

        # Tool was created and registered
        assert tool is not None
        assert tool.name == "fibonacci"
        assert callable(tool.func)

        # Tool returns correct results
        assert tool.func(1) == 1
        assert tool.func(10) == 55

        # Tool is now findable by keyword search
        found = factory.find_tool("fibonacci number")
        assert found is not None
        assert found.name == "fibonacci"

        # LLM was called exactly once
        router.route.assert_awaited_once()

        store.close()

    async def test_factory_reuses_existing_tool_no_llm_call(self) -> None:
        """If a matching tool exists, create_tool returns it without calling LLM."""
        store = _make_store()
        router = _mock_router(FIBONACCI_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        # First call creates
        tool_1 = await factory.create_tool(
            task_description="Compute the nth Fibonacci number"
        )
        assert tool_1 is not None
        assert router.route.await_count == 1

        # Second call with similar description reuses
        tool_2 = await factory.create_tool(
            task_description="Calculate the Fibonacci number for position n"
        )
        assert tool_2 is not None
        assert tool_2.name == tool_1.name
        # No second LLM call
        assert router.route.await_count == 1

        store.close()

    async def test_factory_rejects_unsafe_code_returns_none(self) -> None:
        """If the LLM generates code with imports, the factory rejects it."""
        malicious_response = textwrap.dedent("""\
            ```python
            import os
            def nuke(path: str) -> str:
                os.system("rm -rf " + path)
                return "done"
            ```
            test_input: "/"
            expected_output: "done"
        """)

        store = _make_store()
        router = _mock_router(malicious_response)
        factory = ToolFactory(router=router, store=store)

        tool = await factory.create_tool(
            task_description="Delete system files"
        )

        assert tool is None
        assert len(factory.list_tools()) == 0

        store.close()


# ---------------------------------------------------------------------------
# 4. Tool persistence: created tool survives ToolFactory reload
# ---------------------------------------------------------------------------


class TestToolPersistence:
    """A tool created by ToolFactory is persisted as a Skill node in the
    knowledge graph. A new ToolFactory instance can reload it.

    Flow:
    1. Factory A creates a tool and persists it to the graph.
    2. Factory B (pointing to the same graph) calls load_from_graph().
    3. Factory B has the tool available without any LLM call.
    4. The reloaded tool produces the same results as the original.
    """

    async def test_tool_survives_factory_reload(self) -> None:
        """Created tool is persisted and reloaded by a new factory instance."""
        store = _make_store()
        router = _mock_router(FIBONACCI_LLM_RESPONSE)

        # Factory A: create and persist
        factory_a = ToolFactory(router=router, store=store)
        tool = await factory_a.create_tool(
            task_description="Compute the nth Fibonacci number"
        )
        assert tool is not None
        assert tool.func(10) == 55

        # Verify Skill node exists in graph
        rows = store.query(
            "MATCH (s:Skill) WHERE s.name = $name RETURN s.*",
            params={"name": "fibonacci"},
        )
        assert len(rows) == 1

        # Factory B: brand new instance, same graph store
        factory_b = ToolFactory(router=router, store=store)
        assert len(factory_b.list_tools()) == 0

        loaded_count = factory_b.load_from_graph()

        # Tool was reloaded
        assert loaded_count == 1
        reloaded_tool = factory_b.get_tool("fibonacci")
        assert reloaded_tool is not None

        # Reloaded tool produces the same results
        assert reloaded_tool.func(1) == 1
        assert reloaded_tool.func(10) == 55
        assert reloaded_tool.func(0) == 0

        store.close()

    async def test_reload_ignores_corrupted_skill_nodes(self) -> None:
        """Skills with invalid source code are skipped during reload."""
        store = _make_store()
        router = _mock_router(FIBONACCI_LLM_RESPONSE)

        # Manually insert a Skill node with invalid source code
        store.create_node("Skill", {
            "id": "corrupt-001",
            "name": "broken_tool",
            "description": "This tool has syntax errors",
            "path": "def broken_tool(x)\n    return x + 1",  # missing colon
        })

        factory = ToolFactory(router=router, store=store)
        loaded_count = factory.load_from_graph()

        # Corrupted tool was skipped
        assert loaded_count == 0
        assert factory.get_tool("broken_tool") is None

        store.close()

    async def test_reload_preserves_multiple_tools(self) -> None:
        """Multiple tools persist and reload correctly."""
        store = _make_store()

        add_response = textwrap.dedent("""\
            ```python
            def add_numbers(a: int, b: int) -> int:
                \"\"\"Add two numbers.\"\"\"
                return a + b
            ```
            test_input: 3, 4
            expected_output: 7
        """)

        multiply_response = textwrap.dedent("""\
            ```python
            def multiply_numbers(a: int, b: int) -> int:
                \"\"\"Multiply two numbers.\"\"\"
                return a * b
            ```
            test_input: 3, 4
            expected_output: 12
        """)

        # Create two tools with separate factory calls
        router_add = _mock_router(add_response)
        factory_a = ToolFactory(router=router_add, store=store)
        tool_add = await factory_a.create_tool("Add two numbers together")
        assert tool_add is not None

        router_mul = _mock_router(multiply_response)
        factory_a._router = router_mul
        # Clear existing tools so find_tool does not return add_numbers
        factory_a._tools.clear()
        tool_mul = await factory_a.create_tool("Multiply two numbers")
        assert tool_mul is not None

        # Reload from a fresh factory
        factory_b = ToolFactory(router=_mock_router(""), store=store)
        loaded = factory_b.load_from_graph()

        assert loaded == 2
        assert factory_b.get_tool("add_numbers") is not None
        assert factory_b.get_tool("multiply_numbers") is not None
        assert factory_b.get_tool("add_numbers").func(10, 20) == 30
        assert factory_b.get_tool("multiply_numbers").func(5, 6) == 30

        store.close()


# ---------------------------------------------------------------------------
# 5. Tool quality: stats increment after execution
# ---------------------------------------------------------------------------


class TestToolQualityStats:
    """ToolRegistry tracks per-tool quality statistics (success/failure counts
    and average latency) across executions.

    Validates:
    - Stats start at zero before any execution.
    - Successful execution increments success_count.
    - Failed execution increments failure_count.
    - Average latency is computed correctly.
    - Degraded threshold is detected.
    """

    async def test_stats_increment_on_successful_browser_execution(self) -> None:
        """Executing a browser-domain node successfully increments stats."""
        registry = ToolRegistry()

        # Set up a browser tool with mocked page
        page = _mock_page(url="https://example.com")
        page.inner_text = AsyncMock(return_value="Page content")
        tool = BrowserTool()
        _inject_mocks_into_tool(tool, page)
        registry._browser = tool
        registry._domain_map[Domain.BROWSER] = tool

        node = TaskNode(
            id="stat_test_1",
            description="Navigate to https://example.com and extract content",
            is_atomic=True,
            domain=Domain.BROWSER,
            complexity=1,
            status=TaskStatus.READY,
        )

        # Stats should be empty before execution
        stats_before = registry.get_stats()
        assert "browser" not in stats_before

        result = await registry.execute(node)

        assert result.success is True

        # Stats should now reflect one successful execution
        stats_after = registry.get_stats()
        assert "browser" in stats_after
        browser_stats: ToolStats = stats_after["browser"]
        assert browser_stats.success_count == 1
        assert browser_stats.failure_count == 0
        assert browser_stats.total_count == 1
        assert browser_stats.success_rate == 1.0
        assert browser_stats.avg_latency_ms > 0.0
        assert browser_stats.is_degraded is False

        await tool.close()

    async def test_stats_increment_on_failed_execution(self) -> None:
        """Executing a browser-domain node that fails increments failure count."""
        registry = ToolRegistry()

        # Set up a browser tool that will fail on navigate
        page = _mock_page(url="https://broken.example.com")
        page.goto.side_effect = Exception("net::ERR_CONNECTION_REFUSED")
        tool = BrowserTool()
        _inject_mocks_into_tool(tool, page)
        registry._browser = tool
        registry._domain_map[Domain.BROWSER] = tool

        node = TaskNode(
            id="stat_test_2",
            description="Navigate to https://broken.example.com",
            is_atomic=True,
            domain=Domain.BROWSER,
            complexity=1,
            status=TaskStatus.READY,
        )

        result = await registry.execute(node)

        # Execution should report failure
        assert result.success is False

        stats = registry.get_stats()
        assert "browser" in stats
        browser_stats: ToolStats = stats["browser"]
        assert browser_stats.failure_count == 1
        assert browser_stats.success_count == 0
        assert browser_stats.success_rate == 0.0

        await tool.close()

    async def test_stats_accumulate_across_multiple_executions(self) -> None:
        """Multiple executions accumulate in the same stat bucket."""
        registry = ToolRegistry()

        page = _mock_page(url="https://example.com")
        page.inner_text = AsyncMock(return_value="Content here")
        tool = BrowserTool()
        _inject_mocks_into_tool(tool, page)
        registry._browser = tool
        registry._domain_map[Domain.BROWSER] = tool

        node_ok = TaskNode(
            id="stat_ok",
            description="Navigate to https://example.com and read text",
            is_atomic=True,
            domain=Domain.BROWSER,
            complexity=1,
            status=TaskStatus.READY,
        )

        # Run two successful executions
        await registry.execute(node_ok)
        await registry.execute(node_ok)

        stats = registry.get_stats()
        browser_stats: ToolStats = stats["browser"]
        assert browser_stats.success_count == 2
        assert browser_stats.failure_count == 0
        assert browser_stats.total_count == 2
        assert browser_stats.success_rate == 1.0
        assert browser_stats.avg_latency_ms > 0.0

        await tool.close()

    async def test_stats_track_method_level_keys(self) -> None:
        """When using tool_method routing, stats are keyed by method name."""
        registry = ToolRegistry()

        page = _mock_page(url="https://example.com")
        tool = BrowserTool()
        _inject_mocks_into_tool(tool, page)
        registry._browser = tool
        registry._domain_map[Domain.BROWSER] = tool

        node = TaskNode(
            id="method_stat",
            description="Navigate to example.com",
            is_atomic=True,
            domain=Domain.BROWSER,
            complexity=1,
            status=TaskStatus.READY,
            tool_method="browser_navigate",
            tool_params={"url": "https://example.com"},
        )

        result = await registry.execute(node)
        assert result.success is True

        stats = registry.get_stats()
        assert "browser_navigate" in stats
        assert stats["browser_navigate"].success_count == 1

        await tool.close()

    def test_degraded_threshold_detection(self) -> None:
        """When success rate drops below threshold, tool is flagged degraded."""
        stats = ToolStats(success_count=1, failure_count=4)

        # 1 out of 5 = 20% success rate, well below 50% threshold
        assert stats.total_count == 5
        assert stats.success_rate == 0.2
        assert stats.is_degraded is True

    def test_healthy_stats_not_degraded(self) -> None:
        """Tools with good success rates are not flagged as degraded."""
        stats = ToolStats(success_count=9, failure_count=1)

        assert stats.success_rate == 0.9
        assert stats.is_degraded is False

    def test_zero_executions_not_degraded(self) -> None:
        """Tools with no executions are not considered degraded."""
        stats = ToolStats()

        assert stats.total_count == 0
        assert stats.is_degraded is False
