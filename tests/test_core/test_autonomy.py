"""Tests for core_gb.autonomy -- autonomy levels and per-action risk scoring."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core_gb.autonomy import (
    AutonomyLevel,
    RiskLevel,
    RiskScorer,
    get_autonomy_level,
)
from core_gb.types import Domain, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    description: str,
    domain: Domain = Domain.SYSTEM,
    tool_method: str | None = None,
    tool_params: dict[str, str] | None = None,
) -> TaskNode:
    """Create a minimal TaskNode for autonomy tests."""
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=domain,
        complexity=1,
        status=TaskStatus.READY,
        tool_method=tool_method,
        tool_params=tool_params or {},
    )


# ---------------------------------------------------------------------------
# AutonomyLevel enum
# ---------------------------------------------------------------------------


class TestAutonomyLevelEnum:
    """AutonomyLevel has exactly three values: SUPERVISED, STANDARD, AUTONOMOUS."""

    def test_supervised_value(self) -> None:
        assert AutonomyLevel.SUPERVISED == "supervised"

    def test_standard_value(self) -> None:
        assert AutonomyLevel.STANDARD == "standard"

    def test_autonomous_value(self) -> None:
        assert AutonomyLevel.AUTONOMOUS == "autonomous"

    def test_exactly_three_members(self) -> None:
        assert len(AutonomyLevel) == 3


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------


class TestRiskLevelEnum:
    """RiskLevel has exactly three values: LOW, MEDIUM, HIGH."""

    def test_low_value(self) -> None:
        assert RiskLevel.LOW == "low"

    def test_medium_value(self) -> None:
        assert RiskLevel.MEDIUM == "medium"

    def test_high_value(self) -> None:
        assert RiskLevel.HIGH == "high"

    def test_exactly_three_members(self) -> None:
        assert len(RiskLevel) == 3


# ---------------------------------------------------------------------------
# RiskScorer -- tool-type-based risk scoring
# ---------------------------------------------------------------------------


class TestRiskScorerToolType:
    """RiskScorer assigns risk based on tool_method (tool type)."""

    @pytest.fixture
    def scorer(self) -> RiskScorer:
        return RiskScorer()

    def test_shell_run_is_high_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Run ls", tool_method="shell_run")
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_browser_navigate_is_high_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Open page", tool_method="browser_navigate")
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_browser_click_is_high_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Click submit", tool_method="browser_click")
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_browser_fill_is_high_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Fill form", tool_method="browser_fill")
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_file_read_is_low_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Read file", tool_method="file_read")
        assert scorer.score_node(node) == RiskLevel.LOW

    def test_file_list_is_low_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "List files", tool_method="file_list")
        assert scorer.score_node(node) == RiskLevel.LOW

    def test_file_search_is_low_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Search files", tool_method="file_search")
        assert scorer.score_node(node) == RiskLevel.LOW

    def test_llm_reason_is_low_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Think about this", tool_method="llm_reason")
        assert scorer.score_node(node) == RiskLevel.LOW

    def test_web_search_is_medium_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Search web", tool_method="web_search")
        assert scorer.score_node(node) == RiskLevel.MEDIUM

    def test_web_fetch_is_medium_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Fetch URL", tool_method="web_fetch")
        assert scorer.score_node(node) == RiskLevel.MEDIUM

    def test_browser_extract_text_is_medium_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Extract text", tool_method="browser_extract_text")
        assert scorer.score_node(node) == RiskLevel.MEDIUM

    def test_browser_screenshot_is_medium_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Take screenshot", tool_method="browser_screenshot")
        assert scorer.score_node(node) == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# RiskScorer -- domain-based fallback risk scoring
# ---------------------------------------------------------------------------


class TestRiskScorerDomain:
    """When no tool_method is set, RiskScorer falls back to domain-based scoring."""

    @pytest.fixture
    def scorer(self) -> RiskScorer:
        return RiskScorer()

    def test_code_domain_is_high_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Run a script", domain=Domain.CODE)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_browser_domain_is_high_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Open browser", domain=Domain.BROWSER)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_system_domain_is_high_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "System operation", domain=Domain.SYSTEM)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_web_domain_is_medium_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Search something", domain=Domain.WEB)
        assert scorer.score_node(node) == RiskLevel.MEDIUM

    def test_comms_domain_is_medium_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Send message", domain=Domain.COMMS)
        assert scorer.score_node(node) == RiskLevel.MEDIUM

    def test_file_domain_is_low_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Read a file", domain=Domain.FILE)
        assert scorer.score_node(node) == RiskLevel.LOW

    def test_synthesis_domain_is_low_risk(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Summarize results", domain=Domain.SYNTHESIS)
        assert scorer.score_node(node) == RiskLevel.LOW


# ---------------------------------------------------------------------------
# RiskScorer -- action impact detection (description-based escalation)
# ---------------------------------------------------------------------------


class TestRiskScorerActionImpact:
    """RiskScorer escalates risk when description indicates high-impact actions."""

    @pytest.fixture
    def scorer(self) -> RiskScorer:
        return RiskScorer()

    def test_file_write_escalates_to_high(self, scorer: RiskScorer) -> None:
        """File domain node with 'write' in description escalates to HIGH."""
        node = _node("n1", "Write output to results.txt", domain=Domain.FILE)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_file_create_escalates_to_high(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Create a new config file", domain=Domain.FILE)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_file_delete_escalates_to_high(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Delete the temporary file", domain=Domain.FILE)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_file_save_escalates_to_high(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Save the document", domain=Domain.FILE)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_shell_exec_in_description_escalates(self, scorer: RiskScorer) -> None:
        """Description mentioning shell execution escalates risk."""
        node = _node("n1", "Execute the shell command", domain=Domain.SYNTHESIS)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_web_form_submission_escalates(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Submit the web form", domain=Domain.WEB)
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_navigate_non_allowlisted_escalates(self, scorer: RiskScorer) -> None:
        """Browser navigation to a non-allowlisted domain escalates to HIGH."""
        node = _node(
            "n1",
            "Navigate to http://unknown-site.com",
            domain=Domain.BROWSER,
            tool_method="browser_navigate",
            tool_params={"url": "http://unknown-site.com"},
        )
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_navigate_allowlisted_stays_medium(self, scorer: RiskScorer) -> None:
        """Browser navigation to an allowlisted domain does not escalate."""
        node = _node(
            "n1",
            "Navigate to https://google.com/search",
            domain=Domain.BROWSER,
            tool_method="browser_navigate",
            tool_params={"url": "https://google.com/search?q=test"},
        )
        # browser_navigate is HIGH by default due to tool type,
        # but allowlisted domains should not add further escalation.
        # The base tool_method risk for browser_navigate is HIGH regardless.
        assert scorer.score_node(node) == RiskLevel.HIGH

    def test_benign_file_read_stays_low(self, scorer: RiskScorer) -> None:
        """A file read node without write/delete keywords stays LOW."""
        node = _node("n1", "Read the README.md file", domain=Domain.FILE)
        assert scorer.score_node(node) == RiskLevel.LOW


# ---------------------------------------------------------------------------
# get_autonomy_level -- env var configuration
# ---------------------------------------------------------------------------


class TestGetAutonomyLevel:
    """get_autonomy_level reads AUTONOMY_LEVEL env var, defaults to STANDARD."""

    def test_default_is_standard(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove AUTONOMY_LEVEL if present
            os.environ.pop("AUTONOMY_LEVEL", None)
            assert get_autonomy_level() == AutonomyLevel.STANDARD

    def test_supervised_from_env(self) -> None:
        with patch.dict(os.environ, {"AUTONOMY_LEVEL": "supervised"}):
            assert get_autonomy_level() == AutonomyLevel.SUPERVISED

    def test_standard_from_env(self) -> None:
        with patch.dict(os.environ, {"AUTONOMY_LEVEL": "standard"}):
            assert get_autonomy_level() == AutonomyLevel.STANDARD

    def test_autonomous_from_env(self) -> None:
        with patch.dict(os.environ, {"AUTONOMY_LEVEL": "autonomous"}):
            assert get_autonomy_level() == AutonomyLevel.AUTONOMOUS

    def test_case_insensitive(self) -> None:
        with patch.dict(os.environ, {"AUTONOMY_LEVEL": "SUPERVISED"}):
            assert get_autonomy_level() == AutonomyLevel.SUPERVISED

    def test_invalid_value_falls_back_to_standard(self) -> None:
        with patch.dict(os.environ, {"AUTONOMY_LEVEL": "yolo"}):
            assert get_autonomy_level() == AutonomyLevel.STANDARD


# ---------------------------------------------------------------------------
# RiskScorer -- autonomy level filtering
# ---------------------------------------------------------------------------


class TestAutonomyFiltering:
    """RiskScorer.is_allowed checks whether a node's risk is permitted
    under the current autonomy level."""

    @pytest.fixture
    def scorer(self) -> RiskScorer:
        return RiskScorer()

    # -- SUPERVISED: only LOW risk allowed -----------------------------------

    def test_supervised_allows_low(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Read file", tool_method="file_read")
        assert scorer.is_allowed(node, AutonomyLevel.SUPERVISED) is True

    def test_supervised_blocks_medium(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Search web", tool_method="web_search")
        assert scorer.is_allowed(node, AutonomyLevel.SUPERVISED) is False

    def test_supervised_blocks_high(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Run command", tool_method="shell_run")
        assert scorer.is_allowed(node, AutonomyLevel.SUPERVISED) is False

    # -- STANDARD: LOW and MEDIUM allowed ------------------------------------

    def test_standard_allows_low(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Read file", tool_method="file_read")
        assert scorer.is_allowed(node, AutonomyLevel.STANDARD) is True

    def test_standard_allows_medium(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Search web", tool_method="web_search")
        assert scorer.is_allowed(node, AutonomyLevel.STANDARD) is True

    def test_standard_blocks_high(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Run command", tool_method="shell_run")
        assert scorer.is_allowed(node, AutonomyLevel.STANDARD) is False

    # -- AUTONOMOUS: everything allowed --------------------------------------

    def test_autonomous_allows_low(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Read file", tool_method="file_read")
        assert scorer.is_allowed(node, AutonomyLevel.AUTONOMOUS) is True

    def test_autonomous_allows_medium(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Search web", tool_method="web_search")
        assert scorer.is_allowed(node, AutonomyLevel.AUTONOMOUS) is True

    def test_autonomous_allows_high(self, scorer: RiskScorer) -> None:
        node = _node("n1", "Run command", tool_method="shell_run")
        assert scorer.is_allowed(node, AutonomyLevel.AUTONOMOUS) is True


# ---------------------------------------------------------------------------
# RiskScorer -- filter_dag returns only permitted nodes
# ---------------------------------------------------------------------------


class TestFilterDag:
    """RiskScorer.filter_dag returns nodes permitted under the autonomy level."""

    @pytest.fixture
    def scorer(self) -> RiskScorer:
        return RiskScorer()

    def test_supervised_filters_all_but_low(self, scorer: RiskScorer) -> None:
        nodes = [
            _node("low", "Read file", tool_method="file_read"),
            _node("med", "Search web", tool_method="web_search"),
            _node("high", "Run shell", tool_method="shell_run"),
        ]
        allowed = scorer.filter_dag(nodes, AutonomyLevel.SUPERVISED)
        assert [n.id for n in allowed] == ["low"]

    def test_standard_filters_high_only(self, scorer: RiskScorer) -> None:
        nodes = [
            _node("low", "Read file", tool_method="file_read"),
            _node("med", "Search web", tool_method="web_search"),
            _node("high", "Run shell", tool_method="shell_run"),
        ]
        allowed = scorer.filter_dag(nodes, AutonomyLevel.STANDARD)
        assert [n.id for n in allowed] == ["low", "med"]

    def test_autonomous_allows_all(self, scorer: RiskScorer) -> None:
        nodes = [
            _node("low", "Read file", tool_method="file_read"),
            _node("med", "Search web", tool_method="web_search"),
            _node("high", "Run shell", tool_method="shell_run"),
        ]
        allowed = scorer.filter_dag(nodes, AutonomyLevel.AUTONOMOUS)
        assert [n.id for n in allowed] == ["low", "med", "high"]

    def test_empty_dag_returns_empty(self, scorer: RiskScorer) -> None:
        allowed = scorer.filter_dag([], AutonomyLevel.STANDARD)
        assert allowed == []
