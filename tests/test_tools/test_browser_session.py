"""Tests for browser session caching within DAG executions.

Covers T154 acceptance criteria:
- Browser sessions cached and reused within a DAG execution
- Session lifecycle tied to DAG execution scope
- Two browser nodes in same DAG share session
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools_gb.browser import BrowserTool
from tools_gb.browser_session import BrowserSessionCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_page() -> AsyncMock:
    """Build a mock Playwright page with standard async methods."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.inner_text = AsyncMock(return_value="Hello World")
    page.screenshot = AsyncMock(return_value=b"\x89PNG fake")
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")
    page.wait_for_load_state = AsyncMock()
    page.close = AsyncMock()
    page.set_default_timeout = MagicMock()
    return page


def _mock_browser_context(page: AsyncMock) -> AsyncMock:
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    return context


def _mock_browser(context: AsyncMock) -> AsyncMock:
    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()
    browser.is_connected = MagicMock(return_value=True)
    return browser


def _mock_playwright(browser: AsyncMock) -> AsyncMock:
    pw = AsyncMock()
    pw.chromium = AsyncMock()
    pw.chromium.launch = AsyncMock(return_value=browser)
    pw.stop = AsyncMock()
    return pw


@pytest.fixture
def mocks() -> dict[str, AsyncMock]:
    page = _mock_page()
    context = _mock_browser_context(page)
    browser = _mock_browser(context)
    pw = _mock_playwright(browser)
    return {"playwright": pw, "browser": browser, "context": context, "page": page}


# ---------------------------------------------------------------------------
# Test: BrowserSessionCache lifecycle
# ---------------------------------------------------------------------------


class TestBrowserSessionCacheLifecycle:
    """Session cache creation, reuse, and cleanup."""

    def test_create_session_cache(self) -> None:
        """BrowserSessionCache can be created with default settings."""
        cache = BrowserSessionCache()
        assert cache._sessions == {}
        assert cache._default_headless is True

    def test_create_with_custom_headless(self) -> None:
        """BrowserSessionCache respects headless parameter."""
        cache = BrowserSessionCache(headless=False)
        assert cache._default_headless is False

    async def test_get_session_creates_new(self, mocks: dict[str, AsyncMock]) -> None:
        """First call to get_session creates a new BrowserTool instance."""
        cache = BrowserSessionCache()
        execution_id = "dag_exec_001"

        session = cache.get_session(execution_id)

        assert isinstance(session, BrowserTool)
        assert execution_id in cache._sessions

    async def test_get_session_reuses_existing(self) -> None:
        """Second call with same execution_id returns the same BrowserTool."""
        cache = BrowserSessionCache()
        execution_id = "dag_exec_001"

        session1 = cache.get_session(execution_id)
        session2 = cache.get_session(execution_id)

        assert session1 is session2

    async def test_different_executions_get_different_sessions(self) -> None:
        """Different execution IDs get independent BrowserTool instances."""
        cache = BrowserSessionCache()

        session_a = cache.get_session("dag_exec_001")
        session_b = cache.get_session("dag_exec_002")

        assert session_a is not session_b

    async def test_close_session_cleans_up(self, mocks: dict[str, AsyncMock]) -> None:
        """close_session removes session and calls close on the BrowserTool."""
        cache = BrowserSessionCache()
        execution_id = "dag_exec_001"

        session = cache.get_session(execution_id)
        # Inject mocks into the session so close() has something to clean up
        session._playwright = mocks["playwright"]
        session._browser = mocks["browser"]
        session._context = mocks["context"]
        session._page = mocks["page"]

        await cache.close_session(execution_id)

        assert execution_id not in cache._sessions
        mocks["page"].close.assert_awaited_once()

    async def test_close_session_nonexistent_is_safe(self) -> None:
        """Closing a session that does not exist does not raise."""
        cache = BrowserSessionCache()
        await cache.close_session("nonexistent")  # Should not raise

    async def test_close_all_sessions(self, mocks: dict[str, AsyncMock]) -> None:
        """close_all cleans up every tracked session."""
        cache = BrowserSessionCache()

        session1 = cache.get_session("exec_1")
        session2 = cache.get_session("exec_2")

        # Inject mocks
        for session in [session1, session2]:
            session._playwright = mocks["playwright"]
            session._browser = mocks["browser"]
            session._context = mocks["context"]
            session._page = mocks["page"]

        await cache.close_all()

        assert cache._sessions == {}


class TestBrowserSessionCacheSharing:
    """Two browser nodes in the same DAG share a session."""

    async def test_two_nodes_share_browser(self, mocks: dict[str, AsyncMock]) -> None:
        """Two nodes using the same execution_id share the same BrowserTool."""
        cache = BrowserSessionCache()
        execution_id = "dag_exec_shared"

        # Node 1 gets a session and navigates
        session_for_node1 = cache.get_session(execution_id)
        session_for_node1._playwright = mocks["playwright"]
        session_for_node1._browser = mocks["browser"]
        session_for_node1._context = mocks["context"]
        session_for_node1._page = mocks["page"]

        result1 = await session_for_node1.navigate("https://example.com/page1")
        assert result1["success"] is True

        # Node 2 gets the same session (browser already launched)
        session_for_node2 = cache.get_session(execution_id)
        assert session_for_node2 is session_for_node1

        result2 = await session_for_node2.navigate("https://example.com/page2")
        assert result2["success"] is True

        # Browser was only launched once (same instance reused)
        assert mocks["page"].goto.await_count == 2

    def test_has_session(self) -> None:
        """has_session returns True only for active sessions."""
        cache = BrowserSessionCache()
        assert cache.has_session("exec_1") is False

        cache.get_session("exec_1")
        assert cache.has_session("exec_1") is True

    def test_active_session_count(self) -> None:
        """active_count returns the number of tracked sessions."""
        cache = BrowserSessionCache()
        assert cache.active_count == 0

        cache.get_session("exec_1")
        assert cache.active_count == 1

        cache.get_session("exec_2")
        assert cache.active_count == 2
