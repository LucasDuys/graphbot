"""Tests for browser policy guards -- domain filtering, form control, audit logging.

Covers:
- Blocked domain rejected
- Allowed domain passes
- Form submission blocked by default
- Form submission allowed when BROWSER_ALLOW_FORMS=true
- Audit logging of every browser action
- Allowlist/blocklist precedence (blocklist wins)
- Empty allowlist means all domains permitted (subject to blocklist)
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from tools_gb.browser_policy import (
    BrowserAuditLogger,
    BrowserPolicy,
    PolicyViolation,
)


# ---------------------------------------------------------------------------
# BrowserPolicy -- domain allowlist / blocklist
# ---------------------------------------------------------------------------


class TestDomainBlocklist:
    """BROWSER_BLOCKLIST rejects matching domains."""

    def test_blocked_domain_rejected(self) -> None:
        policy = BrowserPolicy(blocklist=["evil.com", "malware.org"])
        violation = policy.check_url("https://evil.com/page")
        assert violation is not None
        assert "evil.com" in violation.reason

    def test_blocked_subdomain_rejected(self) -> None:
        policy = BrowserPolicy(blocklist=["evil.com"])
        violation = policy.check_url("https://sub.evil.com/page")
        assert violation is not None
        assert "evil.com" in violation.reason

    def test_unblocked_domain_passes(self) -> None:
        policy = BrowserPolicy(blocklist=["evil.com"])
        violation = policy.check_url("https://safe-site.com/page")
        assert violation is None


class TestDomainAllowlist:
    """BROWSER_ALLOWLIST permits only listed domains."""

    def test_allowed_domain_passes(self) -> None:
        policy = BrowserPolicy(allowlist=["example.com", "docs.python.org"])
        violation = policy.check_url("https://example.com/page")
        assert violation is None

    def test_allowed_subdomain_passes(self) -> None:
        policy = BrowserPolicy(allowlist=["example.com"])
        violation = policy.check_url("https://sub.example.com/page")
        assert violation is None

    def test_disallowed_domain_rejected(self) -> None:
        policy = BrowserPolicy(allowlist=["example.com"])
        violation = policy.check_url("https://other-site.com/page")
        assert violation is not None
        assert "not on allowlist" in violation.reason.lower()

    def test_empty_allowlist_permits_all(self) -> None:
        policy = BrowserPolicy(allowlist=[], blocklist=[])
        violation = policy.check_url("https://any-site.com/page")
        assert violation is None


class TestAllowlistBlocklistPrecedence:
    """Blocklist takes precedence over allowlist."""

    def test_blocklist_wins_over_allowlist(self) -> None:
        policy = BrowserPolicy(
            allowlist=["evil.com", "example.com"],
            blocklist=["evil.com"],
        )
        violation = policy.check_url("https://evil.com/page")
        assert violation is not None
        assert "blocklist" in violation.reason.lower()


# ---------------------------------------------------------------------------
# BrowserPolicy -- form control
# ---------------------------------------------------------------------------


class TestFormBlocking:
    """Forms blocked by default, opt-in via allow_forms."""

    def test_fill_blocked_by_default(self) -> None:
        policy = BrowserPolicy()
        violation = policy.check_form_action()
        assert violation is not None
        assert "form" in violation.reason.lower()

    def test_fill_allowed_when_opted_in(self) -> None:
        policy = BrowserPolicy(allow_forms=True)
        violation = policy.check_form_action()
        assert violation is None


# ---------------------------------------------------------------------------
# BrowserPolicy -- loading from environment
# ---------------------------------------------------------------------------


class TestPolicyFromEnv:
    """BrowserPolicy.from_env() reads BROWSER_ALLOWLIST, BROWSER_BLOCKLIST, BROWSER_ALLOW_FORMS."""

    def test_from_env_parses_allowlist(self) -> None:
        env = {"BROWSER_ALLOWLIST": "example.com, docs.python.org"}
        with patch.dict("os.environ", env, clear=False):
            policy = BrowserPolicy.from_env()
        assert policy._allowlist == ["example.com", "docs.python.org"]

    def test_from_env_parses_blocklist(self) -> None:
        env = {"BROWSER_BLOCKLIST": "evil.com,malware.org"}
        with patch.dict("os.environ", env, clear=False):
            policy = BrowserPolicy.from_env()
        assert policy._blocklist == ["evil.com", "malware.org"]

    def test_from_env_allow_forms_true(self) -> None:
        env = {"BROWSER_ALLOW_FORMS": "true"}
        with patch.dict("os.environ", env, clear=False):
            policy = BrowserPolicy.from_env()
        assert policy._allow_forms is True

    def test_from_env_allow_forms_false_by_default(self) -> None:
        env: dict[str, str] = {}
        with patch.dict("os.environ", env, clear=True):
            policy = BrowserPolicy.from_env()
        assert policy._allow_forms is False

    def test_from_env_empty_lists(self) -> None:
        env: dict[str, str] = {}
        with patch.dict("os.environ", env, clear=True):
            policy = BrowserPolicy.from_env()
        assert policy._allowlist == []
        assert policy._blocklist == []


# ---------------------------------------------------------------------------
# BrowserAuditLogger -- action audit logging
# ---------------------------------------------------------------------------


class TestAuditLogger:
    """Every browser action is logged with URL, action type, and timestamp."""

    def test_log_entry_recorded(self) -> None:
        audit = BrowserAuditLogger()
        audit.log("navigate", "https://example.com/page")
        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry["action"] == "navigate"
        assert entry["url"] == "https://example.com/page"
        assert "timestamp" in entry

    def test_multiple_entries(self) -> None:
        audit = BrowserAuditLogger()
        audit.log("navigate", "https://a.com")
        audit.log("click", "https://a.com", selector="#btn")
        audit.log("fill", "https://a.com", selector="#input")
        assert len(audit.entries) == 3
        assert audit.entries[1]["action"] == "click"
        assert audit.entries[1]["selector"] == "#btn"

    def test_log_emits_to_python_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        audit = BrowserAuditLogger()
        with caplog.at_level(logging.INFO, logger="tools_gb.browser_policy"):
            audit.log("navigate", "https://example.com")
        assert any("navigate" in rec.message and "example.com" in rec.message for rec in caplog.records)

    def test_blocked_action_logged(self) -> None:
        audit = BrowserAuditLogger()
        audit.log("navigate", "https://evil.com", blocked=True, reason="blocklist")
        entry = audit.entries[0]
        assert entry["blocked"] is True
        assert entry["reason"] == "blocklist"


# ---------------------------------------------------------------------------
# Integration: BrowserTool with policy guards
# ---------------------------------------------------------------------------


class TestBrowserToolPolicyIntegration:
    """Browser actions check policy before executing."""

    @pytest.mark.asyncio
    async def test_navigate_blocked_domain(self) -> None:
        from tools_gb.browser import BrowserTool

        env = {"BROWSER_BLOCKLIST": "evil.com"}
        with patch.dict("os.environ", env, clear=False):
            tool = BrowserTool()
        result = await tool.navigate("https://evil.com/malware")
        assert result["success"] is False
        assert "blocked" in result["error"].lower() or "policy" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_navigate_allowed_domain(self) -> None:
        from tools_gb.browser import BrowserTool

        env = {"BROWSER_ALLOWLIST": "example.com"}
        with patch.dict("os.environ", env, clear=False):
            tool = BrowserTool()

        # Mock browser launch so we don't need real playwright
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.url = "https://example.com/page"
        tool._ensure_browser = AsyncMock(return_value=mock_page)

        result = await tool.navigate("https://example.com/page")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_fill_blocked_by_default(self) -> None:
        from tools_gb.browser import BrowserTool

        env: dict[str, str] = {}
        with patch.dict("os.environ", env, clear=True):
            tool = BrowserTool()

        # Simulate having a page open
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/form"
        tool._page = mock_page

        result = await tool.fill("#email", "test@test.com")
        assert result["success"] is False
        assert "form" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fill_allowed_when_opted_in(self) -> None:
        from tools_gb.browser import BrowserTool

        env = {"BROWSER_ALLOW_FORMS": "true"}
        with patch.dict("os.environ", env, clear=False):
            tool = BrowserTool()

        mock_page = AsyncMock()
        mock_page.url = "https://example.com/form"
        mock_page.fill = AsyncMock()
        tool._page = mock_page

        result = await tool.fill("#email", "test@test.com")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_navigate_not_on_allowlist(self) -> None:
        from tools_gb.browser import BrowserTool

        env = {"BROWSER_ALLOWLIST": "safe.com"}
        with patch.dict("os.environ", env, clear=False):
            tool = BrowserTool()
        result = await tool.navigate("https://other.com/page")
        assert result["success"] is False
        assert "allowlist" in result["error"].lower() or "not on" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_click_on_blocked_domain_page(self) -> None:
        """Click should also check current page URL against policy."""
        from tools_gb.browser import BrowserTool

        env = {"BROWSER_BLOCKLIST": "evil.com"}
        with patch.dict("os.environ", env, clear=False):
            tool = BrowserTool()

        mock_page = AsyncMock()
        mock_page.url = "https://evil.com/dashboard"
        tool._page = mock_page

        result = await tool.click("#btn")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_audit_log_populated_on_navigate(self) -> None:
        from tools_gb.browser import BrowserTool

        env: dict[str, str] = {}
        with patch.dict("os.environ", env, clear=False):
            tool = BrowserTool()

        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.url = "https://example.com"
        tool._ensure_browser = AsyncMock(return_value=mock_page)

        await tool.navigate("https://example.com")
        assert len(tool.audit.entries) >= 1
        assert tool.audit.entries[0]["action"] == "navigate"

    @pytest.mark.asyncio
    async def test_audit_log_records_blocked_action(self) -> None:
        from tools_gb.browser import BrowserTool

        env = {"BROWSER_BLOCKLIST": "evil.com"}
        with patch.dict("os.environ", env, clear=False):
            tool = BrowserTool()

        await tool.navigate("https://evil.com/bad")
        assert len(tool.audit.entries) >= 1
        assert tool.audit.entries[0]["blocked"] is True
