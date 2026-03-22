"""Browser automation tools for GraphBot DAG leaf execution.

Uses Playwright async API for headless browser control. The browser is
launched lazily on the first action that requires it and reused across
subsequent calls within the same BrowserTool instance. Designed for
integration with the DAG executor where a single BrowserTool instance
serves all browser-domain leaves in one execution run.
"""

from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, Playwright
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


class BrowserTool:
    """Headless browser automation for DAG leaf execution.

    Actions: navigate, extract_text, screenshot, click, fill.
    Browser is launched lazily on first use and reused within the instance.
    Call close() for graceful cleanup when done.
    """

    def __init__(self, headless: bool = True, timeout: int = 30_000) -> None:
        self._headless = headless
        self._timeout = timeout
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._pw_context_manager: Any = None

    async def _ensure_browser(self) -> Page:
        """Launch browser lazily on first use, return the active page.

        If the browser is already running, returns the existing page.
        """
        if self._page is not None:
            return self._page

        logger.info("Launching headless browser (lazy init)")
        self._pw_context_manager = async_playwright()
        self._playwright = await self._pw_context_manager.__aenter__()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self._timeout)
        return self._page

    async def navigate(self, url: str) -> dict[str, Any]:
        """Navigate to a URL.

        Returns: {success, url, title, error}
        """
        if not url or not url.startswith(("http://", "https://")):
            return {"success": False, "error": f"Invalid URL: {url!r}", "url": url}

        try:
            page = await self._ensure_browser()
            await page.goto(url, wait_until="domcontentloaded")
            title = await page.title()
            logger.info("Navigated to %s (%s)", url, title)
            return {
                "success": True,
                "url": url,
                "title": title,
            }
        except Exception as exc:
            logger.error("Navigation to %s failed: %s", url, exc)
            return {"success": False, "error": str(exc), "url": url}

    async def extract_text(self, selector: str = "body") -> dict[str, Any]:
        """Extract visible text content from the current page.

        Args:
            selector: CSS selector to extract text from. Defaults to "body".

        Returns: {success, text, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        try:
            text = await self._page.inner_text(selector)
            return {
                "success": True,
                "text": text,
                "url": self._page.url,
            }
        except Exception as exc:
            logger.error("Text extraction failed (selector=%s): %s", selector, exc)
            return {"success": False, "error": str(exc)}

    async def screenshot(self, path: str | None = None) -> dict[str, Any]:
        """Take a screenshot of the current page.

        Args:
            path: Optional file path to save the screenshot. If None,
                  returns raw PNG bytes in the result dict.

        Returns: {success, data, path, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        try:
            kwargs: dict[str, Any] = {"full_page": True}
            if path is not None:
                kwargs["path"] = path

            data = await self._page.screenshot(**kwargs)
            logger.info("Screenshot taken (%d bytes)", len(data))
            return {
                "success": True,
                "data": data,
                "path": path,
                "url": self._page.url,
            }
        except Exception as exc:
            logger.error("Screenshot failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def click(self, selector: str) -> dict[str, Any]:
        """Click an element matching the CSS selector.

        Returns: {success, selector, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        try:
            await self._page.click(selector)
            logger.info("Clicked selector: %s", selector)
            return {
                "success": True,
                "selector": selector,
                "url": self._page.url,
            }
        except Exception as exc:
            logger.error("Click failed (selector=%s): %s", selector, exc)
            return {"success": False, "error": str(exc)}

    async def fill(self, selector: str, value: str) -> dict[str, Any]:
        """Fill a form field with a value.

        Args:
            selector: CSS selector for the input element.
            value: Text to type into the field.

        Returns: {success, selector, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        try:
            await self._page.fill(selector, value)
            logger.info("Filled selector %s with %d chars", selector, len(value))
            return {
                "success": True,
                "selector": selector,
                "url": self._page.url,
            }
        except Exception as exc:
            logger.error("Fill failed (selector=%s): %s", selector, exc)
            return {"success": False, "error": str(exc)}

    async def close(self) -> None:
        """Gracefully shut down browser resources.

        Closes page, context, browser, and Playwright in order.
        Safe to call multiple times (idempotent).
        """
        for name, resource in [
            ("page", self._page),
            ("context", self._context),
            ("browser", self._browser),
        ]:
            if resource is not None:
                try:
                    await resource.close()
                except Exception as exc:
                    logger.warning("Error closing %s: %s", name, exc)

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                logger.warning("Error stopping playwright: %s", exc)

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._pw_context_manager = None
        logger.info("Browser resources cleaned up")
