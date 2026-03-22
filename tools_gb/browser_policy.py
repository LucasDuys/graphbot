"""Browser policy guards -- domain filtering, form control, and audit logging.

Provides configurable security controls for the BrowserTool:
- Domain allowlist/blocklist via BROWSER_ALLOWLIST / BROWSER_BLOCKLIST env vars
- Form submission blocked by default, opt-in via BROWSER_ALLOW_FORMS=true
- Action audit logging with URL, action type, and timestamp
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _parse_list_env(env_var: str) -> list[str]:
    """Parse a comma-separated environment variable into a list of trimmed strings.

    Returns an empty list if the variable is not set or empty.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _extract_domain(url: str) -> str:
    """Extract the domain (hostname) from a URL, lowercased.

    Returns empty string if URL cannot be parsed.
    """
    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def _domain_matches(hostname: str, pattern: str) -> bool:
    """Check if hostname matches a domain pattern.

    Matches exact domain or any subdomain. For example, pattern "evil.com"
    matches "evil.com" and "sub.evil.com" but not "notevil.com".
    """
    pattern = pattern.lower()
    hostname = hostname.lower()
    return hostname == pattern or hostname.endswith("." + pattern)


@dataclass
class PolicyViolation:
    """Represents a policy check failure."""

    reason: str
    action: str = ""
    url: str = ""


class BrowserPolicy:
    """Configurable policy for browser actions.

    Controls:
    - Domain allowlist: when non-empty, only listed domains are permitted.
    - Domain blocklist: listed domains are always rejected (takes precedence).
    - Form control: form submissions (fill) are blocked by default.
    """

    def __init__(
        self,
        allowlist: list[str] | None = None,
        blocklist: list[str] | None = None,
        allow_forms: bool = False,
    ) -> None:
        self._allowlist: list[str] = [d.lower() for d in (allowlist or [])]
        self._blocklist: list[str] = [d.lower() for d in (blocklist or [])]
        self._allow_forms: bool = allow_forms

    @classmethod
    def from_env(cls) -> BrowserPolicy:
        """Create a BrowserPolicy from environment variables.

        Reads:
        - BROWSER_ALLOWLIST: comma-separated domains (e.g. "example.com,docs.python.org")
        - BROWSER_BLOCKLIST: comma-separated domains (e.g. "evil.com,malware.org")
        - BROWSER_ALLOW_FORMS: "true" to enable form submissions (default: disabled)
        """
        allowlist = _parse_list_env("BROWSER_ALLOWLIST")
        blocklist = _parse_list_env("BROWSER_BLOCKLIST")
        allow_forms = os.environ.get("BROWSER_ALLOW_FORMS", "").strip().lower() == "true"
        return cls(allowlist=allowlist, blocklist=blocklist, allow_forms=allow_forms)

    def check_url(self, url: str) -> PolicyViolation | None:
        """Check if a URL is permitted by the domain policy.

        Evaluation order:
        1. Blocklist check (always takes precedence)
        2. Allowlist check (if non-empty, domain must be listed)

        Returns:
            PolicyViolation if blocked, None if permitted.
        """
        hostname = _extract_domain(url)
        if not hostname:
            return PolicyViolation(
                reason=f"Cannot extract domain from URL: {url!r}",
                url=url,
            )

        # 1. Blocklist takes precedence
        for blocked_domain in self._blocklist:
            if _domain_matches(hostname, blocked_domain):
                return PolicyViolation(
                    reason=(
                        f"Domain '{hostname}' is on the blocklist "
                        f"(matched pattern: {blocked_domain})"
                    ),
                    url=url,
                )

        # 2. Allowlist check (if set)
        if self._allowlist:
            for allowed_domain in self._allowlist:
                if _domain_matches(hostname, allowed_domain):
                    return None
            return PolicyViolation(
                reason=(
                    f"Domain '{hostname}' is not on allowlist "
                    f"(allowed: {', '.join(self._allowlist)})"
                ),
                url=url,
            )

        return None

    def check_form_action(self) -> PolicyViolation | None:
        """Check if form submissions are permitted.

        Returns:
            PolicyViolation if forms are blocked, None if permitted.
        """
        if not self._allow_forms:
            return PolicyViolation(
                reason="Form submissions are blocked by default. "
                "Set BROWSER_ALLOW_FORMS=true to enable.",
            )
        return None


class BrowserAuditLogger:
    """Audit logger for browser actions.

    Records every browser action with URL, action type, timestamp,
    and optional metadata. Entries are stored in-memory and also
    emitted via Python's logging module.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def log(
        self,
        action: str,
        url: str,
        *,
        selector: str | None = None,
        blocked: bool = False,
        reason: str | None = None,
    ) -> None:
        """Record a browser action.

        Args:
            action: Action type (navigate, click, fill, extract_text, screenshot).
            url: The URL involved in the action.
            selector: Optional CSS selector (for click/fill actions).
            blocked: Whether the action was blocked by policy.
            reason: Reason for blocking (if blocked is True).
        """
        entry: dict[str, Any] = {
            "action": action,
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blocked": blocked,
        }
        if selector is not None:
            entry["selector"] = selector
        if reason is not None:
            entry["reason"] = reason

        self.entries.append(entry)

        # Emit to Python logging
        if blocked:
            logger.warning(
                "BROWSER AUDIT [BLOCKED] action=%s url=%s reason=%s",
                action,
                url,
                reason,
            )
        else:
            logger.info(
                "BROWSER AUDIT action=%s url=%s",
                action,
                url,
            )
