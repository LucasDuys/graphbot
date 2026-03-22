"""DAG-level intent classifier and dangerous command blocking.

Analyzes a full decomposition plan (list of TaskNodes) before execution begins.
Catches dangerous shell commands, destructive operations, pipe-to-shell
patterns, and multi-step composition attacks that could cause harm if executed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

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


# ---------------------------------------------------------------------------
# Composition attack signal detectors
# ---------------------------------------------------------------------------
# Each detector returns True if a given text exhibits a particular signal.
# Composition attacks are sequences of individually benign steps that combine
# into a dangerous operation (e.g., download + chmod + execute = malware).

_DOWNLOAD_RE = re.compile(
    r"\b(curl|wget|fetch|download|git\s+clone|pip\s+install|npm\s+install)\b",
    re.IGNORECASE,
)
_MAKE_EXECUTABLE_RE = re.compile(
    r"\bchmod\s+\+x\b", re.IGNORECASE,
)
_EXECUTE_FILE_RE = re.compile(
    r"(\./\w|run\s+\.|execute\s+\.|bash\s+\w|sh\s+\w|python\s+/tmp/|node\s+/tmp/)",
    re.IGNORECASE,
)
_SENSITIVE_READ_RE = re.compile(
    r"\b(cat|read|head|tail|less|more|type|print|dump|show|display)\b.*"
    r"(/etc/(passwd|shadow|hosts|sudoers)"
    r"|\.ssh/(id_rsa|id_ed25519|authorized_keys|known_hosts)"
    r"|\.env|\.aws/credentials|\.gnupg"
    r"|/var/log"
    r"|environment\s+variables|env\s+command"
    r"|\benv\b)",
    re.IGNORECASE,
)
_NETWORK_SEND_RE = re.compile(
    r"\b(curl|wget|nc|ncat|netcat|send|upload|post|http|ftp|scp|rsync)\b",
    re.IGNORECASE,
)


def _has_download(text: str) -> bool:
    """Check if text indicates a file download operation."""
    return bool(_DOWNLOAD_RE.search(text))


def _has_make_executable(text: str) -> bool:
    """Check if text indicates making a file executable."""
    return bool(_MAKE_EXECUTABLE_RE.search(text))


def _has_execute_file(text: str) -> bool:
    """Check if text indicates executing a local file."""
    return bool(_EXECUTE_FILE_RE.search(text))


def _has_sensitive_read(text: str) -> bool:
    """Check if text indicates reading a sensitive file or data."""
    return bool(_SENSITIVE_READ_RE.search(text))


def _has_network_send(text: str) -> bool:
    """Check if text indicates sending data over the network."""
    return bool(_NETWORK_SEND_RE.search(text))


# Composition attack rules: (signal_checkers, description).
# A rule matches when ALL listed signals are found across the DAG nodes.
COMPOSITION_RULES: list[tuple[list[Callable[[str], bool]], str]] = [
    # Malware install: download + make executable + run
    (
        [_has_download, _has_make_executable, _has_execute_file],
        "Composition attack: download + make executable + execute (potential malware install)",
    ),
    # Malware install (no chmod): download + execute from temp/download path
    (
        [_has_download, _has_execute_file],
        "Composition attack: download + execute (potential malware install)",
    ),
    # Data exfiltration: read sensitive data + send via network
    (
        [_has_sensitive_read, _has_network_send],
        "Composition attack: read sensitive data + network send (potential data exfiltration)",
    ),
]


class IntentClassifier:
    """Analyzes a DAG of TaskNodes for dangerous intent before execution.

    Scans every node's description and tool_params against known dangerous
    command patterns. Also detects composition attacks: sequences of
    individually benign steps that together form a dangerous operation.
    Returns a SafetyVerdict indicating whether the DAG should be blocked.
    """

    def __init__(self) -> None:
        self._patterns = DANGEROUS_PATTERNS
        self._composition_rules = COMPOSITION_RULES

    def classify_dag(self, nodes: list[TaskNode]) -> SafetyVerdict:
        """Classify a full decomposition plan for safety.

        Scans all node descriptions and tool parameters against dangerous
        command patterns. Also checks for composition attacks across the
        full set of nodes. Returns a SafetyVerdict with blocked=True if
        any node is flagged or a composition attack is detected.

        Args:
            nodes: The full list of TaskNodes from decomposition.

        Returns:
            SafetyVerdict with blocking decision and details.
        """
        if not nodes:
            return SafetyVerdict(blocked=False, reason="", flagged_nodes=[])

        flagged_nodes: list[str] = []
        reasons: list[str] = []

        # Phase 1: Single-node dangerous pattern detection
        for node in nodes:
            node_reasons = self._scan_node(node)
            if node_reasons:
                flagged_nodes.append(node.id)
                for reason in node_reasons:
                    reasons.append(f"Node '{node.id}': {reason}")

        # Phase 2: Multi-node composition attack detection
        composition_reasons = self._scan_composition(nodes)
        if composition_reasons:
            # Flag all nodes involved in the composition
            all_node_ids = [n.id for n in nodes]
            for node_id in all_node_ids:
                if node_id not in flagged_nodes:
                    flagged_nodes.append(node_id)
            reasons.extend(composition_reasons)

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

    def _collect_node_texts(self, node: TaskNode) -> list[str]:
        """Collect all scannable text from a node (description + tool_params).

        Returns:
            List of text strings to scan for composition signals.
        """
        texts: list[str] = [node.description]
        if node.tool_params:
            texts.extend(node.tool_params.values())
        return texts

    def _scan_composition(self, nodes: list[TaskNode]) -> list[str]:
        """Scan the full DAG for composition attacks.

        Composition attacks are sequences of individually benign steps that
        together form a dangerous operation. Each composition rule defines
        a set of signal checkers; if ALL signals are found across the nodes
        (each signal in at least one node's text), the rule triggers.

        Returns:
            List of composition attack descriptions (empty if clean).
        """
        if len(nodes) < 2:
            return []

        # Gather all text across all nodes
        all_texts: list[str] = []
        for node in nodes:
            all_texts.extend(self._collect_node_texts(node))

        reasons: list[str] = []
        for signal_checkers, description in self._composition_rules:
            # Each signal must be found in at least one text fragment
            all_signals_found = True
            for checker in signal_checkers:
                signal_found = any(checker(text) for text in all_texts)
                if not signal_found:
                    all_signals_found = False
                    break
            if all_signals_found:
                reasons.append(description)
                logger.warning("Composition attack detected: %s", description)

        return reasons
