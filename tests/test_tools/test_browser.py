"""Tests for tools_gb.browser -- BrowserTool with mocked Playwright."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools_gb.browser import BrowserTool


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
    # set_default_timeout is synchronous on real Playwright pages
    page.set_default_timeout = MagicMock()
    return page


def _mock_browser_context(page: AsyncMock) -> AsyncMock:
    """Build a mock browser context that yields the given page."""
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    return context


def _mock_browser(context: AsyncMock) -> AsyncMock:
    """Build a mock browser that yields the given context."""
    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()
    browser.is_connected = MagicMock(return_value=True)
    return browser


def _mock_playwright(browser: AsyncMock) -> AsyncMock:
    """Build a mock async playwright instance."""
    pw = AsyncMock()
    pw.chromium = AsyncMock()
    pw.chromium.launch = AsyncMock(return_value=browser)
    pw.stop = AsyncMock()
    return pw


@pytest.fixture
def mocks() -> dict[str, AsyncMock]:
    """Build a full set of nested Playwright mocks."""
    page = _mock_page()
    context = _mock_browser_context(page)
    browser = _mock_browser(context)
    pw = _mock_playwright(browser)
    return {
        "playwright": pw,
        "browser": browser,
        "context": context,
        "page": page,
    }


@pytest.fixture
def tool() -> BrowserTool:
    return BrowserTool()


# ---------------------------------------------------------------------------
# Test: navigate
# ---------------------------------------------------------------------------


class TestNavigate:
    async def test_navigate_success(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Navigate to a URL, verify page.goto called with correct URL."""
        with patch(
            "tools_gb.browser.async_playwright",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mocks["playwright"]),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            # Force lazy init by patching
            tool._playwright = mocks["playwright"]
            tool._browser = mocks["browser"]
            tool._context = mocks["context"]
            tool._page = mocks["page"]

            result = await tool.navigate("https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        mocks["page"].goto.assert_awaited_once_with(
            "https://example.com", wait_until="domcontentloaded"
        )

    async def test_navigate_invalid_url(self, tool: BrowserTool) -> None:
        """Non-http(s) URLs return error without launching browser."""
        result = await tool.navigate("ftp://files.example.com")
        assert result["success"] is False
        assert "Invalid URL" in result["error"]

    async def test_navigate_empty_url(self, tool: BrowserTool) -> None:
        """Empty URL returns error."""
        result = await tool.navigate("")
        assert result["success"] is False
        assert "Invalid URL" in result["error"]

    async def test_navigate_error_handling(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Navigation failure returns structured error."""
        mocks["page"].goto.side_effect = Exception("net::ERR_NAME_NOT_RESOLVED")
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.navigate("https://nonexistent.invalid")

        assert result["success"] is False
        assert "ERR_NAME_NOT_RESOLVED" in result["error"]


# ---------------------------------------------------------------------------
# Test: extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    async def test_extract_text_success(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Extract text from current page body."""
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.extract_text()

        assert result["success"] is True
        assert result["text"] == "Hello World"
        assert result["url"] == "https://example.com"

    async def test_extract_text_no_page(self, tool: BrowserTool) -> None:
        """extract_text without navigating first returns error."""
        result = await tool.extract_text()

        assert result["success"] is False
        assert "No page" in result["error"]

    async def test_extract_text_with_selector(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Extract text from a specific CSS selector."""
        mocks["page"].inner_text = AsyncMock(return_value="Specific Content")
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.extract_text(selector="#main")

        assert result["success"] is True
        assert result["text"] == "Specific Content"
        mocks["page"].inner_text.assert_awaited_once_with("#main")


# ---------------------------------------------------------------------------
# Test: screenshot
# ---------------------------------------------------------------------------


class TestScreenshot:
    async def test_screenshot_success(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Take a screenshot, returns raw bytes."""
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.screenshot()

        assert result["success"] is True
        assert isinstance(result["data"], bytes)
        assert len(result["data"]) > 0
        mocks["page"].screenshot.assert_awaited_once()

    async def test_screenshot_with_path(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock], tmp_path: object
    ) -> None:
        """Screenshot with a file path passes it to Playwright."""
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.screenshot(path="/tmp/test.png")

        assert result["success"] is True
        mocks["page"].screenshot.assert_awaited_once_with(
            path="/tmp/test.png", full_page=True
        )

    async def test_screenshot_no_page(self, tool: BrowserTool) -> None:
        """Screenshot without navigating first returns error."""
        result = await tool.screenshot()

        assert result["success"] is False
        assert "No page" in result["error"]


# ---------------------------------------------------------------------------
# Test: click
# ---------------------------------------------------------------------------


class TestClick:
    async def test_click_success(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Click a CSS selector."""
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.click("button#submit")

        assert result["success"] is True
        mocks["page"].click.assert_awaited_once_with("button#submit")

    async def test_click_no_page(self, tool: BrowserTool) -> None:
        """Click without a page returns error."""
        result = await tool.click("button#submit")

        assert result["success"] is False
        assert "No page" in result["error"]

    async def test_click_selector_not_found(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Click on non-existent selector returns error."""
        mocks["page"].click.side_effect = Exception(
            "Timeout waiting for selector 'button#missing'"
        )
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.click("button#missing")

        assert result["success"] is False
        assert "button#missing" in result["error"]


# ---------------------------------------------------------------------------
# Test: fill
# ---------------------------------------------------------------------------


class TestFill:
    async def test_fill_success(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Fill a form field with a value."""
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.fill("input#email", "user@example.com")

        assert result["success"] is True
        mocks["page"].fill.assert_awaited_once_with("input#email", "user@example.com")

    async def test_fill_no_page(self, tool: BrowserTool) -> None:
        """Fill without a page returns error."""
        result = await tool.fill("input#email", "user@example.com")

        assert result["success"] is False
        assert "No page" in result["error"]

    async def test_fill_selector_not_found(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """Fill on non-existent selector returns error."""
        mocks["page"].fill.side_effect = Exception(
            "Timeout waiting for selector 'input#missing'"
        )
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        result = await tool.fill("input#missing", "value")

        assert result["success"] is False
        assert "input#missing" in result["error"]


# ---------------------------------------------------------------------------
# Test: lazy browser launch
# ---------------------------------------------------------------------------


class TestLazyLaunch:
    async def test_browser_launched_lazily(self, mocks: dict[str, AsyncMock]) -> None:
        """Browser is not launched until first action that needs it."""
        tool = BrowserTool()
        assert tool._browser is None
        assert tool._page is None

        mock_async_pw = AsyncMock()
        mock_async_pw.__aenter__ = AsyncMock(return_value=mocks["playwright"])
        mock_async_pw.__aexit__ = AsyncMock(return_value=False)

        with patch("tools_gb.browser.async_playwright", return_value=mock_async_pw):
            result = await tool.navigate("https://example.com")

        assert result["success"] is True
        mocks["playwright"].chromium.launch.assert_awaited_once()
        mocks["browser"].new_context.assert_awaited_once()
        mocks["context"].new_page.assert_awaited_once()

    async def test_browser_reused_across_calls(
        self, mocks: dict[str, AsyncMock]
    ) -> None:
        """Second navigate call reuses the existing browser instance."""
        tool = BrowserTool()

        mock_async_pw = AsyncMock()
        mock_async_pw.__aenter__ = AsyncMock(return_value=mocks["playwright"])
        mock_async_pw.__aexit__ = AsyncMock(return_value=False)

        with patch("tools_gb.browser.async_playwright", return_value=mock_async_pw):
            await tool.navigate("https://example.com/page1")
            await tool.navigate("https://example.com/page2")

        # Browser launched only once
        mocks["playwright"].chromium.launch.assert_awaited_once()
        # But goto called twice
        assert mocks["page"].goto.await_count == 2


# ---------------------------------------------------------------------------
# Test: close / cleanup
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_cleans_up(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """close() shuts down page, context, browser, and playwright."""
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        await tool.close()

        mocks["page"].close.assert_awaited_once()
        mocks["context"].close.assert_awaited_once()
        mocks["browser"].close.assert_awaited_once()
        mocks["playwright"].stop.assert_awaited_once()

        # All references cleared
        assert tool._playwright is None
        assert tool._browser is None
        assert tool._context is None
        assert tool._page is None

    async def test_close_idempotent(self, tool: BrowserTool) -> None:
        """Calling close() when no browser is open does not raise."""
        await tool.close()  # Should not raise

    async def test_close_handles_errors(
        self, tool: BrowserTool, mocks: dict[str, AsyncMock]
    ) -> None:
        """close() handles errors during cleanup gracefully."""
        mocks["page"].close.side_effect = Exception("already closed")
        tool._playwright = mocks["playwright"]
        tool._browser = mocks["browser"]
        tool._context = mocks["context"]
        tool._page = mocks["page"]

        # Should not raise
        await tool.close()

        # Still clears references even after error
        assert tool._page is None
        assert tool._browser is None
