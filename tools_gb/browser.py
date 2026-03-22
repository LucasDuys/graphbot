"""Browser automation tools for GraphBot DAG leaf execution.

Uses Playwright async API for headless browser control. The browser is
launched lazily on the first action that requires it and reused across
subsequent calls within the same BrowserTool instance. Designed for
integration with the DAG executor where a single BrowserTool instance
serves all browser-domain leaves in one execution run.

Policy guards (T153):
- Domain allowlist/blocklist configurable via BROWSER_ALLOWLIST / BROWSER_BLOCKLIST
- Form submissions blocked by default (opt-in via BROWSER_ALLOW_FORMS=true)
- Every browser action is audit-logged with URL, action type, and timestamp
"""

from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, Playwright
from playwright.async_api import async_playwright

from tools_gb.browser_policy import BrowserAuditLogger, BrowserPolicy

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
        self._policy: BrowserPolicy = BrowserPolicy.from_env()
        self.audit: BrowserAuditLogger = BrowserAuditLogger()

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

        Checks domain policy before navigation. Logs the action to audit trail.

        Returns: {success, url, title, error}
        """
        if not url or not url.startswith(("http://", "https://")):
            self.audit.log("navigate", url or "", blocked=True, reason="invalid URL")
            return {"success": False, "error": f"Invalid URL: {url!r}", "url": url}

        # Policy check: domain allowlist/blocklist
        violation = self._policy.check_url(url)
        if violation is not None:
            self.audit.log("navigate", url, blocked=True, reason=violation.reason)
            return {"success": False, "error": f"Policy violation: {violation.reason}", "url": url}

        try:
            page = await self._ensure_browser()
            await page.goto(url, wait_until="domcontentloaded")
            title = await page.title()
            logger.info("Navigated to %s (%s)", url, title)
            self.audit.log("navigate", url)
            return {
                "success": True,
                "url": url,
                "title": title,
            }
        except Exception as exc:
            logger.error("Navigation to %s failed: %s", url, exc)
            self.audit.log("navigate", url, blocked=True, reason=str(exc))
            return {"success": False, "error": str(exc), "url": url}

    async def extract_text(self, selector: str = "body") -> dict[str, Any]:
        """Extract visible text content from the current page.

        Checks current page URL against domain policy. Logs the action.

        Args:
            selector: CSS selector to extract text from. Defaults to "body".

        Returns: {success, text, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        current_url = self._page.url
        violation = self._policy.check_url(current_url)
        if violation is not None:
            self.audit.log(
                "extract_text", current_url,
                selector=selector, blocked=True, reason=violation.reason,
            )
            return {"success": False, "error": f"Policy violation: {violation.reason}"}

        try:
            text = await self._page.inner_text(selector)
            self.audit.log("extract_text", current_url, selector=selector)
            return {
                "success": True,
                "text": text,
                "url": current_url,
            }
        except Exception as exc:
            logger.error("Text extraction failed (selector=%s): %s", selector, exc)
            self.audit.log(
                "extract_text", current_url,
                selector=selector, blocked=True, reason=str(exc),
            )
            return {"success": False, "error": str(exc)}

    async def screenshot(self, path: str | None = None) -> dict[str, Any]:
        """Take a screenshot of the current page.

        Checks current page URL against domain policy. Logs the action.

        Args:
            path: Optional file path to save the screenshot. If None,
                  returns raw PNG bytes in the result dict.

        Returns: {success, data, path, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        current_url = self._page.url
        violation = self._policy.check_url(current_url)
        if violation is not None:
            self.audit.log("screenshot", current_url, blocked=True, reason=violation.reason)
            return {"success": False, "error": f"Policy violation: {violation.reason}"}

        try:
            kwargs: dict[str, Any] = {"full_page": True}
            if path is not None:
                kwargs["path"] = path

            data = await self._page.screenshot(**kwargs)
            logger.info("Screenshot taken (%d bytes)", len(data))
            self.audit.log("screenshot", current_url)
            return {
                "success": True,
                "data": data,
                "path": path,
                "url": current_url,
            }
        except Exception as exc:
            logger.error("Screenshot failed: %s", exc)
            self.audit.log("screenshot", current_url, blocked=True, reason=str(exc))
            return {"success": False, "error": str(exc)}

    async def click(self, selector: str) -> dict[str, Any]:
        """Click an element matching the CSS selector.

        Checks current page URL against domain policy. Logs the action.

        Returns: {success, selector, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        current_url = self._page.url
        violation = self._policy.check_url(current_url)
        if violation is not None:
            self.audit.log(
                "click", current_url,
                selector=selector, blocked=True, reason=violation.reason,
            )
            return {"success": False, "error": f"Policy violation: {violation.reason}"}

        try:
            await self._page.click(selector)
            logger.info("Clicked selector: %s", selector)
            self.audit.log("click", current_url, selector=selector)
            return {
                "success": True,
                "selector": selector,
                "url": current_url,
            }
        except Exception as exc:
            logger.error("Click failed (selector=%s): %s", selector, exc)
            self.audit.log(
                "click", current_url,
                selector=selector, blocked=True, reason=str(exc),
            )
            return {"success": False, "error": str(exc)}

    async def fill(self, selector: str, value: str) -> dict[str, Any]:
        """Fill a form field with a value.

        Checks form policy (blocked by default) and current page URL against
        domain policy. Logs the action.

        Args:
            selector: CSS selector for the input element.
            value: Text to type into the field.

        Returns: {success, selector, url, error}
        """
        if self._page is None:
            return {"success": False, "error": "No page open. Call navigate() first."}

        current_url = self._page.url

        # Form policy check (blocked by default)
        form_violation = self._policy.check_form_action()
        if form_violation is not None:
            self.audit.log(
                "fill", current_url,
                selector=selector, blocked=True, reason=form_violation.reason,
            )
            return {"success": False, "error": f"Policy violation: {form_violation.reason}"}

        # Domain policy check
        url_violation = self._policy.check_url(current_url)
        if url_violation is not None:
            self.audit.log(
                "fill", current_url,
                selector=selector, blocked=True, reason=url_violation.reason,
            )
            return {"success": False, "error": f"Policy violation: {url_violation.reason}"}

        try:
            await self._page.fill(selector, value)
            logger.info("Filled selector %s with %d chars", selector, len(value))
            self.audit.log("fill", current_url, selector=selector)
            return {
                "success": True,
                "selector": selector,
                "url": current_url,
            }
        except Exception as exc:
            logger.error("Fill failed (selector=%s): %s", selector, exc)
            self.audit.log(
                "fill", current_url,
                selector=selector, blocked=True, reason=str(exc),
            )
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
