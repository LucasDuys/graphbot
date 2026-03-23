"""Aggregation of parallel leaf outputs with optional LLM synthesis.

Deterministic aggregation (zero LLM calls) is the default path. When a
ModelRouter is available and the number of leaf outputs meets the synthesis
threshold, an LLM call produces a coherent prose response from the subtask
results. JSON artifacts are stripped from subtask outputs before synthesis
and before final output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.router import ModelRouter

from core_gb.types import CompletionResult, TaskNode, TaskStatus, Domain

logger = logging.getLogger(__name__)

# Default minimum number of leaf outputs required to trigger LLM synthesis.
# Tasks with fewer outputs skip synthesis and use deterministic aggregation
# with JSON artifact cleanup.
DEFAULT_SYNTHESIS_THRESHOLD: int = 3

# ---------------------------------------------------------------------------
# JSON artifact stripping
# ---------------------------------------------------------------------------

# Matches a top-level JSON object (greedy, handles nested braces).
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")

# Matches a top-level JSON array.
_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]")

# Matches JSON-style quoted key patterns like "key": "value" or "key": 123.
_QUOTED_KEY_RE = re.compile(r'"(\w+)"\s*:\s*')


def strip_json_artifacts(text: str) -> str:
    """Remove raw JSON artifacts from text, preserving human-readable content.

    Strips:
    - Full JSON objects ({...}) -- extracts string values from them
    - Full JSON arrays ([...]) -- extracts items from them
    - JSON key-value syntax ("key": "value") -- extracts the value

    Args:
        text: Raw text potentially containing JSON artifacts.

    Returns:
        Cleaned text with JSON artifacts removed, readable content preserved.
    """
    if not text:
        return text

    result = text

    # Phase 1: Try to parse as a complete JSON object and extract values
    stripped = result.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                values = _extract_json_values(parsed)
                if values:
                    return ". ".join(str(v) for v in values if v)
        except json.JSONDecodeError:
            pass

    # Phase 2: Try to parse as a complete JSON array and extract items
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                items = [str(item) for item in parsed if item]
                if items:
                    return ", ".join(items)
        except json.JSONDecodeError:
            pass

    # Phase 3: Remove embedded JSON objects from mixed content
    def _replace_json_obj(match: re.Match[str]) -> str:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                values = _extract_json_values(parsed)
                if values:
                    return " ".join(str(v) for v in values if v)
        except json.JSONDecodeError:
            pass
        return ""

    result = _JSON_OBJECT_RE.sub(_replace_json_obj, result)

    # Phase 4: Remove embedded JSON arrays from mixed content
    def _replace_json_arr(match: re.Match[str]) -> str:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                items = [str(item) for item in parsed if item]
                if items:
                    return ", ".join(items)
        except json.JSONDecodeError:
            pass
        return ""

    result = _JSON_ARRAY_RE.sub(_replace_json_arr, result)

    # Phase 5: Strip remaining JSON-style quoted key patterns
    result = _QUOTED_KEY_RE.sub("", result)

    # Clean up: collapse multiple spaces, strip stray quotes
    result = re.sub(r"\s{2,}", " ", result)
    result = result.strip()

    return result


def _extract_json_values(obj: dict[str, Any]) -> list[Any]:
    """Recursively extract human-readable values from a JSON dict.

    Prioritizes keys like 'answer', 'content', 'result', 'text' over
    metadata keys like 'confidence', 'status', 'model'.

    Args:
        obj: A parsed JSON dictionary.

    Returns:
        List of extracted values suitable for human reading.
    """
    priority_keys = ("answer", "content", "result", "text", "response", "output")
    skip_keys = ("confidence", "status", "model", "tokens", "cost", "latency")

    values: list[Any] = []

    # First pass: priority keys
    for key in priority_keys:
        if key in obj:
            val = obj[key]
            if isinstance(val, dict):
                values.extend(_extract_json_values(val))
            elif isinstance(val, str) and val:
                values.append(val)
            elif val is not None:
                values.append(str(val))

    if values:
        return values

    # Second pass: all non-skip keys
    for key, val in obj.items():
        if key in skip_keys:
            continue
        if isinstance(val, dict):
            values.extend(_extract_json_values(val))
        elif isinstance(val, str) and val:
            values.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict):
                    values.extend(_extract_json_values(item))
        elif val is not None:
            values.append(str(val))

    return values


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class Aggregator:
    """Combines leaf outputs into a final response.

    Supports two modes:
    1. Deterministic aggregation (zero LLM calls) -- concatenate, template
       fill, merge JSON, or confidence ranking.
    2. LLM synthesis -- uses a ModelRouter to produce coherent prose from
       subtask results. Activated only when a router is provided and the
       number of leaf outputs meets the synthesis threshold.
    """

    def __init__(
        self,
        router: ModelRouter | None = None,
        synthesis_threshold: int = DEFAULT_SYNTHESIS_THRESHOLD,
    ) -> None:
        self._router = router
        self._synthesis_threshold = synthesis_threshold

    def aggregate(
        self,
        template: dict[str, Any] | None,
        leaf_outputs: dict[str, str],
    ) -> str:
        """Aggregate leaf outputs using the specified template.

        This is the synchronous deterministic path -- zero LLM calls.

        Args:
            template: Output template from decomposition. Has:
                - aggregation_type: "concatenate" | "merge_json" | "confidence_ranked" | "template_fill"
                - template: string with {slot_id} placeholders
                - slot_definitions: dict mapping slot_id to description
            leaf_outputs: Maps provides key -> leaf output text

        Returns:
            Aggregated output string. Zero LLM calls.
        """
        if template is None:
            return self._concatenate(leaf_outputs)

        agg_type = template.get("aggregation_type", "concatenate")

        if agg_type == "template_fill":
            return self._template_fill(template, leaf_outputs)
        elif agg_type == "merge_json":
            return self._merge_json(leaf_outputs)
        elif agg_type == "confidence_ranked":
            return self._confidence_ranked(leaf_outputs)
        else:  # concatenate (default)
            return self._concatenate(leaf_outputs)

    async def synthesize(
        self,
        original_question: str,
        leaf_outputs: dict[str, str],
        template: dict[str, Any] | None,
    ) -> str:
        """Aggregate leaf outputs with optional LLM synthesis.

        When the number of leaf outputs meets the synthesis threshold and a
        router is available, sends the cleaned subtask results to an LLM to
        produce a coherent prose response.

        When below the threshold or no router is available, falls back to
        deterministic aggregation with JSON artifact cleanup.

        Args:
            original_question: The original user question that triggered
                the decomposed task.
            leaf_outputs: Maps provides key -> leaf output text.
            template: Optional output template from decomposition.

        Returns:
            A coherent response string, free of JSON artifacts.
        """
        # Strip JSON artifacts from all leaf outputs
        cleaned_outputs: dict[str, str] = {
            key: strip_json_artifacts(value)
            for key, value in leaf_outputs.items()
        }

        # Decide whether to use LLM synthesis
        should_synthesize = (
            self._router is not None
            and len(cleaned_outputs) >= self._synthesis_threshold
        )

        if not should_synthesize:
            # Below threshold or no router: deterministic aggregation on
            # cleaned outputs
            return self.aggregate(template, cleaned_outputs)

        # Build synthesis prompt
        subtask_summary = "\n".join(
            f"- {key.replace('_', ' ').title()}: {value}"
            for key, value in cleaned_outputs.items()
        )

        synthesis_prompt = (
            f"Given the original question and these subtask results, "
            f"write a coherent, natural-language response. Do not include "
            f"any JSON, code blocks, or raw data structures. Write clear "
            f"prose that directly answers the question.\n\n"
            f"Original question: {original_question}\n\n"
            f"Subtask results:\n{subtask_summary}"
        )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a synthesis assistant. Combine subtask results "
                    "into a single coherent response. Write natural prose, "
                    "not bullet points or JSON. Be concise and direct."
                ),
            },
            {
                "role": "user",
                "content": synthesis_prompt,
            },
        ]

        # Use a lightweight task node for routing (complexity 2 = cheap model)
        synthesis_task = TaskNode(
            id="synthesis",
            description="Synthesize subtask results",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=2,
            status=TaskStatus.READY,
        )

        try:
            completion: CompletionResult = await self._router.route(
                synthesis_task, messages,
            )
            synthesized = completion.content
            # Final cleanup: strip any JSON artifacts the LLM might produce
            synthesized = strip_json_artifacts(synthesized)
            logger.debug(
                "LLM synthesis produced %d chars (model=%s, tokens=%d+%d)",
                len(synthesized),
                completion.model,
                completion.tokens_in,
                completion.tokens_out,
            )
            return synthesized
        except Exception as exc:
            logger.warning(
                "LLM synthesis failed, falling back to deterministic "
                "aggregation: %s",
                exc,
            )
            return self.aggregate(template, cleaned_outputs)

    def _concatenate(self, outputs: dict[str, str]) -> str:
        """Join outputs with headers."""
        if not outputs:
            return ""
        if len(outputs) == 1:
            return next(iter(outputs.values()))

        parts = []
        for key, value in outputs.items():
            header = key.replace("_", " ").title()
            parts.append(f"## {header}\n{value}")
        return "\n\n".join(parts)

    def _template_fill(self, template: dict[str, Any], outputs: dict[str, str]) -> str:
        """Fill template slots with leaf outputs."""
        tmpl = template.get("template", "")
        if not tmpl:
            return self._concatenate(outputs)

        result = tmpl
        for key, value in outputs.items():
            result = result.replace("{" + key + "}", value)

        unfilled = re.findall(r"\{(\w+)\}", result)
        for slot in unfilled:
            logger.warning("Unfilled template slot: %s", slot)
            result = result.replace("{" + slot + "}", f"[No data for {slot}]")

        return result

    def _merge_json(self, outputs: dict[str, str]) -> str:
        """Deep merge JSON outputs into a single object."""
        merged: dict[str, Any] = {}
        for key, value in outputs.items():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    merged.update(parsed)
                else:
                    merged[key] = parsed
            except json.JSONDecodeError:
                merged[key] = value
        return json.dumps(merged, indent=2)

    def _confidence_ranked(self, outputs: dict[str, str]) -> str:
        """Rank outputs by confidence score (if embedded) and concatenate top results."""
        scored: list[tuple[float, str, str]] = []
        for key, value in outputs.items():
            confidence = 0.5
            text = value
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    confidence = float(parsed.get("confidence", 0.5))
                    text = parsed.get("answer", parsed.get("content", value))
            except (json.JSONDecodeError, ValueError):
                pass
            scored.append((confidence, key, text))

        scored.sort(key=lambda x: x[0], reverse=True)

        parts = []
        for conf, key, text in scored:
            parts.append(text)
        return "\n\n".join(parts)
