"""Autonomy levels and per-action risk scoring for DAG node execution.

Provides three autonomy tiers (SUPERVISED, STANDARD, AUTONOMOUS) configurable
via the AUTONOMY_LEVEL environment variable.  Each TaskNode is assigned a
RiskLevel (LOW, MEDIUM, HIGH) based on its tool type, domain, and action
impact.  The autonomy level determines which risk tiers are permitted to
execute without human approval.

Risk assignment priority:
  1. Explicit tool_method (highest specificity)
  2. Action impact keywords in description (escalation)
  3. Domain-based fallback (lowest specificity)
"""

from __future__ import annotations

import logging
import os
import re
from enum import Enum
from urllib.parse import urlparse

from core_gb.types import Domain, TaskNode

logger = logging.getLogger(__name__)


class AutonomyLevel(str, Enum):
    """Tier 2 autonomy levels controlling which risk classes auto-execute."""

    SUPERVISED = "supervised"
    STANDARD = "standard"
    AUTONOMOUS = "autonomous"


class RiskLevel(str, Enum):
    """Risk classification for a single node action."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Risk mappings
# ---------------------------------------------------------------------------

# tool_method -> base risk level
_TOOL_RISK: dict[str, RiskLevel] = {
    # High risk: shell execution, browser interaction with state changes
    "shell_run": RiskLevel.HIGH,
    "browser_navigate": RiskLevel.HIGH,
    "browser_click": RiskLevel.HIGH,
    "browser_fill": RiskLevel.HIGH,
    # Medium risk: read-only network access, passive browser operations
    "web_search": RiskLevel.MEDIUM,
    "web_fetch": RiskLevel.MEDIUM,
    "browser_extract_text": RiskLevel.MEDIUM,
    "browser_screenshot": RiskLevel.MEDIUM,
    # Low risk: local reads, reasoning
    "file_read": RiskLevel.LOW,
    "file_list": RiskLevel.LOW,
    "file_search": RiskLevel.LOW,
    "llm_reason": RiskLevel.LOW,
}

# Domain -> fallback risk level (used when no tool_method is set)
_DOMAIN_RISK: dict[Domain, RiskLevel] = {
    Domain.CODE: RiskLevel.HIGH,
    Domain.BROWSER: RiskLevel.HIGH,
    Domain.SYSTEM: RiskLevel.HIGH,
    Domain.WEB: RiskLevel.MEDIUM,
    Domain.COMMS: RiskLevel.MEDIUM,
    Domain.FILE: RiskLevel.LOW,
    Domain.SYNTHESIS: RiskLevel.LOW,
}

# Patterns in task description that indicate high-impact actions.
# When matched, the node's risk is escalated to HIGH regardless of
# tool_method or domain base risk.
_HIGH_IMPACT_PATTERNS: list[re.Pattern[str]] = [
    # File mutations
    re.compile(r"\b(write|create|save|delete|remove|overwrite|modify|append)\b", re.IGNORECASE),
    # Shell execution mentioned in description
    re.compile(r"\b(shell\s+exec|execute\s+(the\s+)?shell|run\s+(the\s+)?shell)\b", re.IGNORECASE),
    # Web form submission
    re.compile(r"\bsubmit\b.*\bform\b", re.IGNORECASE),
]

# Allowlisted domains for browser_navigate.  Navigation to these domains
# does not trigger additional escalation beyond the base tool_method risk.
# (browser_navigate is already HIGH by default, but the allowlist is used
# for future downgrade logic if needed.)
_BROWSER_DOMAIN_ALLOWLIST: frozenset[str] = frozenset({
    "google.com",
    "www.google.com",
    "bing.com",
    "www.bing.com",
    "duckduckgo.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "github.com",
    "stackoverflow.com",
})

# Autonomy level -> maximum permitted risk level (inclusive)
_AUTONOMY_MAX_RISK: dict[AutonomyLevel, RiskLevel] = {
    AutonomyLevel.SUPERVISED: RiskLevel.LOW,
    AutonomyLevel.STANDARD: RiskLevel.MEDIUM,
    AutonomyLevel.AUTONOMOUS: RiskLevel.HIGH,
}

# Numeric ordering for risk comparison
_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}


def get_autonomy_level() -> AutonomyLevel:
    """Read the autonomy level from the AUTONOMY_LEVEL environment variable.

    Defaults to STANDARD if the variable is unset or contains an invalid value.
    Matching is case-insensitive.
    """
    raw = os.environ.get("AUTONOMY_LEVEL", "standard").strip().lower()
    try:
        return AutonomyLevel(raw)
    except ValueError:
        logger.warning(
            "Invalid AUTONOMY_LEVEL '%s', falling back to 'standard'", raw
        )
        return AutonomyLevel.STANDARD


class RiskScorer:
    """Assigns a risk level to each TaskNode and enforces autonomy filtering.

    Scoring priority:
      1. tool_method lookup (explicit tool type)
      2. Description-based escalation (high-impact action keywords)
      3. Domain-based fallback
    """

    def score_node(self, task: TaskNode) -> RiskLevel:
        """Score a single TaskNode and return its risk level.

        Args:
            task: The TaskNode to evaluate.

        Returns:
            RiskLevel indicating how dangerous this action is.
        """
        risk = self._base_risk(task)
        escalated = self._escalate_by_impact(task, risk)

        if escalated != risk:
            logger.debug(
                "Node '%s' risk escalated from %s to %s (impact keywords)",
                task.id,
                risk.value,
                escalated.value,
            )

        return escalated

    def is_allowed(self, task: TaskNode, autonomy: AutonomyLevel) -> bool:
        """Check whether a node is permitted under the given autonomy level.

        Args:
            task: The TaskNode to check.
            autonomy: The current autonomy level.

        Returns:
            True if the node's risk level is within the autonomy ceiling.
        """
        risk = self.score_node(task)
        max_risk = _AUTONOMY_MAX_RISK[autonomy]
        return _RISK_ORDER[risk] <= _RISK_ORDER[max_risk]

    def filter_dag(
        self, nodes: list[TaskNode], autonomy: AutonomyLevel
    ) -> list[TaskNode]:
        """Return only nodes permitted under the given autonomy level.

        Args:
            nodes: Full list of TaskNodes from decomposition.
            autonomy: The current autonomy level.

        Returns:
            Filtered list of nodes whose risk is within the autonomy ceiling.
        """
        return [node for node in nodes if self.is_allowed(node, autonomy)]

    # -- Internal helpers ----------------------------------------------------

    def _base_risk(self, task: TaskNode) -> RiskLevel:
        """Determine the base risk from tool_method or domain fallback."""
        if task.tool_method:
            return _TOOL_RISK.get(task.tool_method, RiskLevel.MEDIUM)
        return _DOMAIN_RISK.get(task.domain, RiskLevel.MEDIUM)

    def _escalate_by_impact(self, task: TaskNode, current: RiskLevel) -> RiskLevel:
        """Escalate risk to HIGH if description indicates a high-impact action.

        Actions that always escalate to HIGH:
          - File write/create/save/delete
          - Shell execution mentioned in description
          - Web form submission
          - Browser navigation to non-allowlisted domain
        """
        if current == RiskLevel.HIGH:
            return current

        desc = task.description

        # Check high-impact keyword patterns
        for pattern in _HIGH_IMPACT_PATTERNS:
            if pattern.search(desc):
                return RiskLevel.HIGH

        # Check browser navigation to non-allowlisted domain
        if task.tool_method == "browser_navigate":
            url = task.tool_params.get("url", "")
            if url and not self._is_allowlisted_domain(url):
                return RiskLevel.HIGH

        return current

    @staticmethod
    def _is_allowlisted_domain(url: str) -> bool:
        """Check if a URL's domain is in the browser allowlist."""
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            # Strip leading 'www.' for matching flexibility
            bare = hostname.lstrip("www.") if hostname.startswith("www.") else hostname
            return hostname in _BROWSER_DOMAIN_ALLOWLIST or bare in _BROWSER_DOMAIN_ALLOWLIST
        except Exception:
            return False
