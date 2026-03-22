"""Browser session caching for DAG execution scope.

Provides a BrowserSessionCache that manages BrowserTool instances keyed by
DAG execution ID. Sessions are created lazily on first access and reused
across all browser-domain nodes within the same execution run. The cache
owner (typically the DAG executor) is responsible for calling close_session()
or close_all() when the execution completes.
"""

from __future__ import annotations

import logging
from typing import Any

from tools_gb.browser import BrowserTool

logger = logging.getLogger(__name__)


class BrowserSessionCache:
    """Cache of BrowserTool instances keyed by DAG execution ID.

    Within a single DAG execution, all browser-domain leaf nodes share the
    same BrowserTool (and therefore the same Playwright browser context).
    This avoids launching multiple browser instances for the same execution.

    Args:
        headless: Default headless mode for new BrowserTool instances.
        timeout: Default page timeout in milliseconds.
    """

    def __init__(self, headless: bool = True, timeout: int = 30_000) -> None:
        self._sessions: dict[str, BrowserTool] = {}
        self._default_headless: bool = headless
        self._default_timeout: int = timeout

    def get_session(self, execution_id: str) -> BrowserTool:
        """Get or create a BrowserTool for the given execution.

        If a session already exists for this execution_id, returns the
        existing instance. Otherwise creates a new BrowserTool with the
        cache's default settings.

        Args:
            execution_id: Unique identifier for the DAG execution run.

        Returns:
            A BrowserTool instance bound to this execution scope.
        """
        if execution_id in self._sessions:
            logger.debug("Reusing browser session for execution %s", execution_id)
            return self._sessions[execution_id]

        logger.info("Creating new browser session for execution %s", execution_id)
        tool = BrowserTool(
            headless=self._default_headless,
            timeout=self._default_timeout,
        )
        self._sessions[execution_id] = tool
        return tool

    def has_session(self, execution_id: str) -> bool:
        """Check whether a session exists for the given execution ID."""
        return execution_id in self._sessions

    @property
    def active_count(self) -> int:
        """Number of currently tracked sessions."""
        return len(self._sessions)

    async def close_session(self, execution_id: str) -> None:
        """Close and remove the browser session for a specific execution.

        If no session exists for this execution_id, this is a no-op.

        Args:
            execution_id: The execution whose browser session to close.
        """
        tool = self._sessions.pop(execution_id, None)
        if tool is None:
            return

        try:
            await tool.close()
            logger.info("Closed browser session for execution %s", execution_id)
        except Exception as exc:
            logger.warning(
                "Error closing browser session for execution %s: %s",
                execution_id,
                exc,
            )

    async def close_all(self) -> None:
        """Close all tracked browser sessions.

        Used during shutdown or when the executor finishes all work.
        Errors during individual session cleanup are logged but do not
        prevent other sessions from being cleaned up.
        """
        execution_ids = list(self._sessions.keys())
        for execution_id in execution_ids:
            await self.close_session(execution_id)

        logger.info("All browser sessions cleaned up (%d total)", len(execution_ids))
