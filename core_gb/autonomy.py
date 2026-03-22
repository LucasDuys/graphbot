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

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from core_gb.types import Domain, TaskNode

if TYPE_CHECKING:
    from models.router import ModelRouter

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


# ---------------------------------------------------------------------------
# Multi-model cross-validation for high-risk plans
# ---------------------------------------------------------------------------

# Complexity levels used to select different models for cross-validation.
# Using levels 3 (mid-tier) and 5 (top-tier) ensures two distinct models
# review the plan, providing genuine independent validation.
_CROSS_VALIDATION_COMPLEXITY_A: int = 3
_CROSS_VALIDATION_COMPLEXITY_B: int = 5

_CROSS_VALIDATION_SYSTEM_PROMPT: str = (
    "You are a safety reviewer for an AI task execution engine. "
    "You will be given a plan consisting of one or more task nodes. "
    "Each node has a description, domain, and risk level. "
    "Evaluate whether this plan is safe to execute autonomously.\n\n"
    "Respond with ONLY valid JSON in this exact format:\n"
    '{"safe": true, "reason": "brief explanation"}\n'
    "or\n"
    '{"safe": false, "reason": "brief explanation of why it is unsafe"}\n\n'
    "A plan is UNSAFE if it could cause data loss, system damage, "
    "unauthorized access, privacy violations, or other harm. "
    "Be conservative: when in doubt, mark as unsafe."
)


@dataclass(frozen=True)
class CrossValidationResult:
    """Result of multi-model cross-validation for a high-risk plan.

    Attributes:
        approved: Whether the plan passed cross-validation (both models agree safe).
        risk_level: The highest risk level found across all nodes in the plan.
        model_a_response: Raw response content from the first model.
        model_b_response: Raw response content from the second model.
        explanation: Human-readable summary of the cross-validation outcome.
    """

    approved: bool
    risk_level: RiskLevel
    model_a_response: str = ""
    model_b_response: str = ""
    explanation: str = ""


class CrossValidator:
    """Multi-model cross-validation for high-risk execution plans.

    For plans where the highest node risk is HIGH, queries two different
    models (selected via different complexity levels through ModelRouter)
    to independently evaluate plan safety. Both models must agree the plan
    is safe for execution to proceed.

    Plans with risk level below HIGH are approved without model calls.
    On any error (network, JSON parse, missing fields), the validator
    defaults to blocking (fail-safe).
    """

    def __init__(self, scorer: RiskScorer, router: ModelRouter) -> None:
        self._scorer = scorer
        self._router = router

    async def validate_plan(self, nodes: list[TaskNode]) -> CrossValidationResult:
        """Validate a plan through multi-model cross-validation.

        Only triggers for plans where the maximum risk level across all
        nodes is HIGH. Lower-risk plans are approved immediately.

        Args:
            nodes: The list of TaskNodes forming the execution plan.

        Returns:
            CrossValidationResult indicating whether the plan is approved.
        """
        if not nodes:
            return CrossValidationResult(
                approved=True,
                risk_level=RiskLevel.LOW,
                explanation="Empty plan trivially approved.",
            )

        # Determine the highest risk level across all nodes.
        max_risk = self._compute_max_risk(nodes)

        # Only cross-validate HIGH risk plans.
        if max_risk != RiskLevel.HIGH:
            return CrossValidationResult(
                approved=True,
                risk_level=max_risk,
                explanation=f"Plan risk level is {max_risk.value}; cross-validation not required.",
            )

        # Build the validation prompt describing the plan.
        messages = self._build_validation_messages(nodes)

        # Query two different models via different complexity levels.
        model_a_response = await self._query_model(
            messages, _CROSS_VALIDATION_COMPLEXITY_A
        )
        model_b_response = await self._query_model(
            messages, _CROSS_VALIDATION_COMPLEXITY_B
        )

        # Parse responses and determine outcome.
        return self._evaluate_responses(model_a_response, model_b_response, max_risk)

    def _compute_max_risk(self, nodes: list[TaskNode]) -> RiskLevel:
        """Compute the highest risk level across all nodes in the plan."""
        max_order = 0
        max_risk = RiskLevel.LOW
        for node in nodes:
            risk = self._scorer.score_node(node)
            order = _RISK_ORDER[risk]
            if order > max_order:
                max_order = order
                max_risk = risk
        return max_risk

    def _build_validation_messages(self, nodes: list[TaskNode]) -> list[dict[str, str]]:
        """Build the LLM prompt messages for plan validation."""
        plan_description_parts: list[str] = []
        for node in nodes:
            risk = self._scorer.score_node(node)
            plan_description_parts.append(
                f"- Node '{node.id}': {node.description} "
                f"[domain={node.domain.value}, risk={risk.value}]"
            )

        plan_text = "\n".join(plan_description_parts)
        user_content = (
            f"Evaluate the safety of this execution plan:\n\n{plan_text}"
        )

        return [
            {"role": "system", "content": _CROSS_VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def _query_model(
        self, messages: list[dict[str, str]], complexity: int
    ) -> str:
        """Query a model at the specified complexity level for plan validation.

        Returns the raw response content, or an error marker string on failure.
        """
        from core_gb.types import TaskNode as _TN

        route_node = _TN(
            id="_cross_validate",
            description="cross-validation safety review",
            complexity=complexity,
        )
        try:
            result = await self._router.route(
                route_node, messages,
                response_format={"type": "json_object"},
            )
            return result.content
        except Exception as exc:
            logger.warning(
                "Cross-validation model query failed (complexity=%d): %s",
                complexity, exc,
            )
            return f'{{"error": "{exc}"}}'

    def _evaluate_responses(
        self,
        response_a: str,
        response_b: str,
        risk_level: RiskLevel,
    ) -> CrossValidationResult:
        """Parse both model responses and determine approval or block.

        Both models must return {"safe": true} for approval. Any error in
        parsing or a missing "safe" field is treated as unsafe (fail-safe).
        """
        safe_a, reason_a = self._parse_safety_response(response_a)
        safe_b, reason_b = self._parse_safety_response(response_b)

        if safe_a and safe_b:
            return CrossValidationResult(
                approved=True,
                risk_level=risk_level,
                model_a_response=response_a,
                model_b_response=response_b,
                explanation="Both models agree the plan is safe. Approved for execution.",
            )

        # Build explanation with details about which model(s) flagged the plan.
        disagreement_parts: list[str] = []
        if not safe_a:
            disagreement_parts.append(f"Model A (complexity {_CROSS_VALIDATION_COMPLEXITY_A}): {reason_a}")
        if not safe_b:
            disagreement_parts.append(f"Model B (complexity {_CROSS_VALIDATION_COMPLEXITY_B}): {reason_b}")

        explanation = (
            "Models disagree on plan safety. Execution blocked. "
            + " | ".join(disagreement_parts)
        )

        return CrossValidationResult(
            approved=False,
            risk_level=risk_level,
            model_a_response=response_a,
            model_b_response=response_b,
            explanation=explanation,
        )

    @staticmethod
    def _parse_safety_response(response: str) -> tuple[bool, str]:
        """Parse a model's safety review response.

        Returns (is_safe, reason). On parse failure, returns (False, error_message)
        to enforce fail-safe behaviour.
        """
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return False, f"Failed to parse response as JSON: {response[:200]}"

        if "error" in data:
            return False, f"Model returned error: {data['error']}"

        if "safe" not in data:
            return False, "Response missing 'safe' field"

        is_safe = bool(data["safe"])
        reason = str(data.get("reason", ""))
        return is_safe, reason
