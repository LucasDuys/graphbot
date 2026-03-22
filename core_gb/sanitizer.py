"""Output sanitizer for DAG data forwarding.

Strips potential prompt injection patterns from node outputs before they are
forwarded to dependent nodes in the DAG executor. This prevents adversarial
outputs from one node from hijacking the behavior of downstream nodes.

Patterns detected and removed:
- System prompt overrides ("You are now...", "Ignore previous instructions")
- Instruction injection patterns ("[SYSTEM]:", "<system>", "New instructions:")
- Role-play attempts ("Act as...", "Pretend you are...", "Roleplay as...")
- Jailbreak/developer mode triggers
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Each pattern matches a full line that should be removed from the output.
# Patterns are case-insensitive and anchored to line content.
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # System prompt overrides
    (
        re.compile(r"^.*\byou are now\b.*$", re.IGNORECASE | re.MULTILINE),
        "system prompt override (you are now)",
    ),
    (
        re.compile(
            r"^.*\bignore\s+(all\s+)?previous\s+instructions\b.*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "instruction override (ignore previous)",
    ),
    (
        re.compile(
            r"^.*\bdisregard\s+(all\s+)?prior\s+instructions\b.*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "instruction override (disregard prior)",
    ),
    (
        re.compile(
            r"^.*\bforget\s+(your\s+)?previous\s+instructions\b.*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "instruction override (forget previous)",
    ),
    (
        re.compile(
            r"^.*\boverride\s+(your\s+)?(system\s+)?instructions\b.*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "instruction override (override instructions)",
    ),
    # Instruction injection markers
    (
        re.compile(r"^.*\[SYSTEM\]\s*:.*$", re.IGNORECASE | re.MULTILINE),
        "system tag injection ([SYSTEM]:)",
    ),
    (
        re.compile(r"^.*<system>.*</system>.*$", re.IGNORECASE | re.MULTILINE),
        "system tag injection (<system>)",
    ),
    (
        re.compile(r"^.*<system>.*$", re.IGNORECASE | re.MULTILINE),
        "system tag injection (<system>)",
    ),
    (
        re.compile(
            r"^.*\bnew\s+instructions\s*:.*$", re.IGNORECASE | re.MULTILINE,
        ),
        "instruction injection (new instructions:)",
    ),
    # Role-play injection
    (
        re.compile(r"^.*\bact\s+as\b.*$", re.IGNORECASE | re.MULTILINE),
        "role-play injection (act as)",
    ),
    (
        re.compile(r"^.*\bpretend\s+you\s+are\b.*$", re.IGNORECASE | re.MULTILINE),
        "role-play injection (pretend you are)",
    ),
    (
        re.compile(r"^.*\broleplay\s+as\b.*$", re.IGNORECASE | re.MULTILINE),
        "role-play injection (roleplay as)",
    ),
    (
        re.compile(r"^.*\bjailbreak\s+mode\b.*$", re.IGNORECASE | re.MULTILINE),
        "jailbreak mode trigger",
    ),
    (
        re.compile(r"^.*\bdeveloper\s+mode\b.*$", re.IGNORECASE | re.MULTILINE),
        "developer mode trigger",
    ),
]


class OutputSanitizer:
    """Sanitizes node outputs by removing prompt injection patterns.

    Used by the DAG executor to clean data before forwarding it from a
    completed node to its dependent nodes. Operates line-by-line: lines
    matching injection patterns are removed entirely, preserving all other
    content.
    """

    def __init__(self) -> None:
        self._patterns = _INJECTION_PATTERNS

    def sanitize(self, text: str) -> str:
        """Remove prompt injection patterns from text.

        Lines matching injection patterns are removed. Remaining lines are
        joined back together. If the text is empty or has no injections,
        it is returned unchanged.

        Args:
            text: The raw output text from a completed node.

        Returns:
            Sanitized text with injection patterns removed.
        """
        result, _ = self._do_sanitize(text)
        return result

    def sanitize_with_flag(self, text: str) -> tuple[str, bool]:
        """Sanitize text and report whether any modifications were made.

        Args:
            text: The raw output text from a completed node.

        Returns:
            Tuple of (sanitized_text, was_modified).
        """
        return self._do_sanitize(text)

    def _do_sanitize(self, text: str) -> tuple[str, bool]:
        """Internal sanitization logic. Returns (sanitized_text, was_modified)."""
        if not text:
            return text, False

        modified = False
        result = text

        for pattern, description in self._patterns:
            new_result = pattern.sub("", result)
            if new_result != result:
                logger.warning(
                    "Output sanitizer removed injection pattern: %s", description
                )
                modified = True
                result = new_result

        if modified:
            # Clean up: remove empty lines left by removals, collapse multiple
            # blank lines into single blank lines
            lines = result.split("\n")
            cleaned: list[str] = []
            prev_blank = False
            for line in lines:
                stripped = line.strip()
                if stripped == "":
                    if not prev_blank:
                        cleaned.append("")
                    prev_blank = True
                else:
                    cleaned.append(line)
                    prev_blank = False

            # Strip leading/trailing blank lines
            while cleaned and cleaned[0].strip() == "":
                cleaned.pop(0)
            while cleaned and cleaned[-1].strip() == "":
                cleaned.pop()

            result = "\n".join(cleaned)

        return result, modified
