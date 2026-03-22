"""DAG-level intent classifier and dangerous command blocking.

Analyzes a full decomposition plan (list of TaskNodes) before execution begins.
Catches dangerous shell commands, destructive operations, and pipe-to-shell
patterns that could cause harm if executed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core_gb.types import TaskNode

logger = logging.getLogger(__name__)


@dataclass
class SafetyVerdict:
    """Result of DAG-level safety classification.

    Attributes:
        blocked: Whether the DAG should be blocked from execution.
        reason: Human-readable explanation of why the DAG was blocked.
        flagged_nodes: List of node IDs that triggered the block.
    """

    blocked: bool
    reason: str
    flagged_nodes: list[str] = field(default_factory=list)


# Dangerous command patterns for DAG-level scanning.
# These extend the shell-level BLOCKED_PATTERNS from tools_gb/shell.py
# to catch dangerous intent at the plan level (before execution).
DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Destructive file operations
    (re.compile(r"\brm\s+-[rf]{1,2}\s+/", re.IGNORECASE), "recursive file deletion at root"),
    (re.compile(r"\brm\s+-[rf]{1,2}\s+~", re.IGNORECASE), "recursive file deletion in home"),
    (re.compile(r"\bdel\s+/[fq]\s+[A-Z]:\\", re.IGNORECASE), "Windows forced file deletion"),

    # Disk formatting and overwriting
    (re.compile(r"\bmkfs\b", re.IGNORECASE), "filesystem creation (mkfs)"),
    (re.compile(r"\bdd\s+if=", re.IGNORECASE), "raw disk write (dd)"),
    (re.compile(r"\bformat\s+[A-Z]:", re.IGNORECASE), "disk format command"),

    # System shutdown and reboot
    (re.compile(r"\b(shutdown|reboot|poweroff|halt)\b", re.IGNORECASE), "system shutdown/reboot"),

    # Dangerous permission changes
    (re.compile(r"\bchmod\s+777\b", re.IGNORECASE), "world-writable permissions (chmod 777)"),

    # Pipe-to-shell (remote code execution)
    (re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b", re.IGNORECASE), "curl piped to shell"),
    (re.compile(r"\bwget\b.*\|\s*(ba)?sh\b", re.IGNORECASE), "wget piped to shell"),

    # Fork bomb
    (re.compile(r":\(\)\s*\{", re.IGNORECASE), "fork bomb pattern"),
]


class IntentClassifier:
    """Analyzes a DAG of TaskNodes for dangerous intent before execution.

    Scans every node's description and tool_params against known dangerous
    command patterns. Returns a SafetyVerdict indicating whether the DAG
    should be blocked.
    """

    def __init__(self) -> None:
        self._patterns = DANGEROUS_PATTERNS

    def classify_dag(self, nodes: list[TaskNode]) -> SafetyVerdict:
        """Classify a full decomposition plan for safety.

        Scans all node descriptions and tool parameters against dangerous
        command patterns. Returns a SafetyVerdict with blocked=True if any
        node is flagged.

        Args:
            nodes: The full list of TaskNodes from decomposition.

        Returns:
            SafetyVerdict with blocking decision and details.
        """
        if not nodes:
            return SafetyVerdict(blocked=False, reason="", flagged_nodes=[])

        flagged_nodes: list[str] = []
        reasons: list[str] = []

        for node in nodes:
            node_reasons = self._scan_node(node)
            if node_reasons:
                flagged_nodes.append(node.id)
                for reason in node_reasons:
                    reasons.append(f"Node '{node.id}': {reason}")

        if flagged_nodes:
            combined_reason = "; ".join(reasons)
            logger.warning(
                "DAG blocked by safety classifier: %s", combined_reason
            )
            return SafetyVerdict(
                blocked=True,
                reason=combined_reason,
                flagged_nodes=flagged_nodes,
            )

        return SafetyVerdict(blocked=False, reason="", flagged_nodes=[])

    def _scan_node(self, node: TaskNode) -> list[str]:
        """Scan a single node for dangerous patterns.

        Checks both the node description and any tool_params (especially
        shell command parameters).

        Returns:
            List of reasons the node was flagged (empty if safe).
        """
        reasons: list[str] = []

        # Scan the node description
        reasons.extend(self._scan_text(node.description))

        # Scan tool parameters if present (shell commands embedded in params)
        if node.tool_params:
            for param_value in node.tool_params.values():
                reasons.extend(self._scan_text(param_value))

        return reasons

    def _scan_text(self, text: str) -> list[str]:
        """Scan a text string against all dangerous patterns.

        Returns:
            List of matched pattern descriptions (empty if clean).
        """
        reasons: list[str] = []
        for pattern, description in self._patterns:
            if pattern.search(text):
                reasons.append(description)
        return reasons
