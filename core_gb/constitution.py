"""Constitutional principles for plan verification.

Defines five core principles (no harm, no deception, no unauthorized access,
respect privacy, minimize side effects) and a ConstitutionalChecker that
evaluates a list of TaskNodes against all principles before execution.

Each principle is a callable check function that scans node descriptions
and tool_params for violations, returning a list of (principle_name, reason)
tuples. The ConstitutionalChecker aggregates violations from all principles
into a ConstitutionalVerdict.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from core_gb.types import Domain, TaskNode, TaskStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class ConstitutionalVerdict:
    """Result of constitutional review of a plan.

    Attributes:
        passed: Whether the plan passed all constitutional checks.
        violations: List of (principle_name, reason) tuples describing
            which principles were violated and why.
    """

    passed: bool
    violations: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Principle type alias
# ---------------------------------------------------------------------------

# A principle check function takes a list of TaskNodes and returns a list
# of (principle_name, reason) tuples for any violations found.
PrincipleCheck = Callable[[list[TaskNode]], list[tuple[str, str]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_texts(nodes: list[TaskNode]) -> list[tuple[str, str]]:
    """Collect all scannable text from nodes with their node IDs.

    Returns a list of (node_id, text) pairs from descriptions and tool_params.
    """
    texts: list[tuple[str, str]] = []
    for node in nodes:
        texts.append((node.id, node.description))
        if node.tool_params:
            for value in node.tool_params.values():
                texts.append((node.id, value))
    return texts


def _scan_texts(
    nodes: list[TaskNode],
    patterns: list[tuple[re.Pattern[str], str]],
    principle_name: str,
) -> list[tuple[str, str]]:
    """Scan all node texts against a list of patterns.

    Args:
        nodes: TaskNodes to scan.
        patterns: List of (compiled_regex, description) pairs.
        principle_name: Name of the principle for violation reporting.

    Returns:
        List of (principle_name, reason) tuples for each match.
    """
    violations: list[tuple[str, str]] = []
    texts = _collect_texts(nodes)
    for node_id, text in texts:
        for pattern, description in patterns:
            if pattern.search(text):
                reason = f"Node '{node_id}': {description}"
                violations.append((principle_name, reason))
    return violations


# ---------------------------------------------------------------------------
# Principle 1: No Harm
# ---------------------------------------------------------------------------

_HARM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"\b(destroy|wipe out|obliterate|nuke|annihilate)\b.*\b(data|file|system|server|database|record)",
        re.IGNORECASE,
    ), "plan includes destructive intent toward data or systems"),
    (re.compile(
        r"\b(corrupt|sabotage|cripple|disable|break)\b.*\b(database|system|server|service|network)",
        re.IGNORECASE,
    ), "plan includes sabotage of systems or services"),
    (re.compile(
        r"\b(harm|injure|hurt|attack|assault)\b.*\b(someone|person|people|user|individual|victim)",
        re.IGNORECASE,
    ), "plan includes intent to harm persons"),
    (re.compile(
        r"\bgenerate\s+instructions\s+to\s+harm\b",
        re.IGNORECASE,
    ), "plan generates harmful instructions"),
    (re.compile(
        r"\b(denial.of.service|ddos|dos\s+attack|flood)\b.*\b(server|target|service|network|site)",
        re.IGNORECASE,
    ), "plan includes denial-of-service attack"),
    (re.compile(
        r"\bdelete\s+all\b.*\b(data|files|records|backups|user)\b",
        re.IGNORECASE,
    ), "plan deletes all data or records"),
    (re.compile(
        r"\bdestroy\s+all\b.*\b(backup|data|file|record|log)",
        re.IGNORECASE,
    ), "plan destroys all backups or data"),
]


def check_no_harm(nodes: list[TaskNode]) -> list[tuple[str, str]]:
    """Check plan for actions that cause physical or digital harm.

    Scans node descriptions and tool_params for patterns indicating
    destructive operations, sabotage, or intent to harm persons.

    Args:
        nodes: The list of TaskNodes to check.

    Returns:
        List of (principle_name, reason) tuples for violations found.
    """
    return _scan_texts(nodes, _HARM_PATTERNS, "no_harm")


# ---------------------------------------------------------------------------
# Principle 2: No Deception
# ---------------------------------------------------------------------------

_DECEPTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"\b(impersonat\w*|pretend\s+to\s+be|pose\s+as|masquerad\w*)",
        re.IGNORECASE,
    ), "plan involves impersonation or false identity"),
    (re.compile(
        r"\bfake\b.*\b(profile|identity|account|credential|document|login|page)",
        re.IGNORECASE,
    ), "plan creates fake identities or documents"),
    (re.compile(
        r"\b(fabricat\w*|forg\w*|counterfeit\w*)\b.*\b(evidence|document|record|data|credential|proof)",
        re.IGNORECASE,
    ), "plan fabricates or forges evidence"),
    (re.compile(
        r"\bphish(ing)?\b",
        re.IGNORECASE,
    ), "plan involves phishing"),
    (re.compile(
        r"\b(mislead|deceive|trick|defraud|dupe|scam)\b",
        re.IGNORECASE,
    ), "plan involves deception or fraud"),
    (re.compile(
        r"\b(misinformation|disinformation|propaganda)\b",
        re.IGNORECASE,
    ), "plan spreads misinformation"),
]


def check_deception(nodes: list[TaskNode]) -> list[tuple[str, str]]:
    """Check plan for deceptive actions.

    Scans node descriptions and tool_params for patterns indicating
    impersonation, phishing, fabrication, or other deceptive intent.

    Args:
        nodes: The list of TaskNodes to check.

    Returns:
        List of (principle_name, reason) tuples for violations found.
    """
    return _scan_texts(nodes, _DECEPTION_PATTERNS, "no_deception")


# ---------------------------------------------------------------------------
# Principle 3: No Unauthorized Access
# ---------------------------------------------------------------------------

_UNAUTHORIZED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"\bbypass\b.*\b(auth|firewall|security|access.control|permission|login|verification)",
        re.IGNORECASE,
    ), "plan bypasses authentication or security controls"),
    (re.compile(
        r"\b(escalat\w*|elevat\w*)\b.*\b(privilege|permission|access|role)",
        re.IGNORECASE,
    ), "plan escalates privileges without authorization"),
    (re.compile(
        r"\bbrute\s*force\b",
        re.IGNORECASE,
    ), "plan uses brute force to crack credentials"),
    (re.compile(
        r"\bexploit\b.*\b(vulnerabilit|bug|flaw|weakness|injection|xss|csrf|overflow)",
        re.IGNORECASE,
    ), "plan exploits security vulnerabilities"),
    (re.compile(
        r"\b(crack|hack\s+into|break\s+into)\b.*\b(password|account|system|server|network)",
        re.IGNORECASE,
    ), "plan involves cracking passwords or hacking into systems"),
    (re.compile(
        r"\bunauthorized\s+(access|entry|login)\b",
        re.IGNORECASE,
    ), "plan involves unauthorized access"),
]


def check_unauthorized_access(nodes: list[TaskNode]) -> list[tuple[str, str]]:
    """Check plan for unauthorized access attempts.

    Scans node descriptions and tool_params for patterns indicating
    authentication bypass, privilege escalation, brute force attacks,
    or exploitation of vulnerabilities.

    Args:
        nodes: The list of TaskNodes to check.

    Returns:
        List of (principle_name, reason) tuples for violations found.
    """
    return _scan_texts(nodes, _UNAUTHORIZED_PATTERNS, "no_unauthorized_access")


# ---------------------------------------------------------------------------
# Principle 4: Respect Privacy
# ---------------------------------------------------------------------------

_PRIVACY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"\b(harvest|collect|gather|scrape|mine|exfiltrate)\b.*\b(personal|private|user)\b.*\b(data|info|record|detail|email)",
        re.IGNORECASE,
    ), "plan harvests personal or private data"),
    (re.compile(
        r"\bscrape\b.*\b(private|personal|medical|financial|health)\b.*\b(record|data|info|detail)",
        re.IGNORECASE,
    ), "plan scrapes private records"),
    (re.compile(
        r"\b(surveillance|spy|eavesdrop|wiretap|keylog)\b",
        re.IGNORECASE,
    ), "plan involves surveillance or monitoring"),
    (re.compile(
        r"\btrack\b.*\b(user|location|activit|browsing|movement)\b.*\bwithout\b",
        re.IGNORECASE,
    ), "plan tracks users without consent"),
    (re.compile(
        r"\bmonitor\b.*\b(employee|user|person)\b.*\b(activit|browsing|email|communication)",
        re.IGNORECASE,
    ), "plan monitors user activities"),
    (re.compile(
        r"\bwithout\s+(their\s+)?(consent|knowledge|permission|authorization)\b",
        re.IGNORECASE,
    ), "plan operates without user consent"),
]


def check_privacy(nodes: list[TaskNode]) -> list[tuple[str, str]]:
    """Check plan for privacy violations.

    Scans node descriptions and tool_params for patterns indicating
    unauthorized data collection, surveillance, tracking, or operations
    performed without user consent.

    Args:
        nodes: The list of TaskNodes to check.

    Returns:
        List of (principle_name, reason) tuples for violations found.
    """
    return _scan_texts(nodes, _PRIVACY_PATTERNS, "respect_privacy")


# ---------------------------------------------------------------------------
# Principle 5: Minimize Side Effects
# ---------------------------------------------------------------------------

_SIDE_EFFECT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"\b(mass|bulk|blast)\b.*\b(email|message|notification|spam)\b",
        re.IGNORECASE,
    ), "plan sends mass unsolicited communications"),
    (re.compile(
        r"\bunsolicited\b.*\b(email|message|notification|contact)",
        re.IGNORECASE,
    ), "plan sends unsolicited messages"),
    (re.compile(
        r"\bmodify\b.*\bglobal\b.*\b(config|setting|system|environment)",
        re.IGNORECASE,
    ), "plan modifies global configuration"),
    (re.compile(
        r"\b(wipe|clear|purge|erase)\b.*\b(all\s+)?(log|audit|trail|history)\b",
        re.IGNORECASE,
    ), "plan wipes logs or audit trails"),
    (re.compile(
        r"\bcover\s+(your\s+|the\s+)?tracks\b",
        re.IGNORECASE,
    ), "plan attempts to cover tracks"),
    (re.compile(
        r"\baffecting\s+all\s+users\b",
        re.IGNORECASE,
    ), "plan has broad unintended impact on all users"),
]


def check_side_effects(nodes: list[TaskNode]) -> list[tuple[str, str]]:
    """Check plan for unintended broad side effects.

    Scans node descriptions and tool_params for patterns indicating
    mass communications, global configuration changes, log wiping,
    or other actions with broad unintended impact.

    Args:
        nodes: The list of TaskNodes to check.

    Returns:
        List of (principle_name, reason) tuples for violations found.
    """
    return _scan_texts(nodes, _SIDE_EFFECT_PATTERNS, "minimize_side_effects")


# ---------------------------------------------------------------------------
# Principles registry
# ---------------------------------------------------------------------------

PRINCIPLES: list[tuple[str, PrincipleCheck]] = [
    ("no_harm", check_no_harm),
    ("no_deception", check_deception),
    ("no_unauthorized_access", check_unauthorized_access),
    ("respect_privacy", check_privacy),
    ("minimize_side_effects", check_side_effects),
]


# ---------------------------------------------------------------------------
# ConstitutionalChecker
# ---------------------------------------------------------------------------


class ConstitutionalChecker:
    """Verifies a plan (list of TaskNodes) against all constitutional principles.

    Each principle is a callable check function that scans node descriptions
    and tool_params for violations. The checker aggregates all violations
    into a single ConstitutionalVerdict.

    Also provides check_text() for pre-decomposition scanning of raw user
    messages against all constitutional principles (zero LLM cost).
    """

    def __init__(self) -> None:
        self._principles = PRINCIPLES

    def check_text(self, text: str) -> ConstitutionalVerdict:
        """Check raw user message text against all constitutional principles.

        Wraps the text in a minimal TaskNode and runs all principle checks.
        This is a zero-cost pre-decomposition check -- no LLM calls needed.

        Args:
            text: The raw user message string.

        Returns:
            ConstitutionalVerdict with pass/fail and list of violations.
        """
        if not text:
            return ConstitutionalVerdict(passed=True, violations=[])

        # Wrap text in a minimal TaskNode for principle check compatibility
        proxy_node = TaskNode(
            id="__pre_decomposition__",
            description=text,
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
        )
        return self.check_plan([proxy_node])

    def check_plan(self, nodes: list[TaskNode]) -> ConstitutionalVerdict:
        """Check a plan against all constitutional principles.

        Runs every registered principle check function against the full list
        of TaskNodes. Collects all violations and returns a ConstitutionalVerdict
        with passed=False if any violations are found.

        Args:
            nodes: The list of TaskNodes to verify.

        Returns:
            ConstitutionalVerdict with pass/fail and list of violations.
        """
        if not nodes:
            return ConstitutionalVerdict(passed=True, violations=[])

        all_violations: list[tuple[str, str]] = []

        for principle_name, check_fn in self._principles:
            violations = check_fn(nodes)
            if violations:
                all_violations.extend(violations)
                logger.warning(
                    "Constitutional principle '%s' violated: %d violation(s)",
                    principle_name,
                    len(violations),
                )

        if all_violations:
            logger.warning(
                "Plan failed constitutional review with %d total violation(s)",
                len(all_violations),
            )
            return ConstitutionalVerdict(passed=False, violations=all_violations)

        return ConstitutionalVerdict(passed=True, violations=[])
