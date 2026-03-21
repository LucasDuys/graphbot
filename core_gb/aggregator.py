"""Deterministic aggregation of parallel leaf outputs. Zero LLM calls."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class Aggregator:
    """Combines leaf outputs into final response using templates. No LLM calls."""

    def aggregate(
        self,
        template: dict[str, Any] | None,
        leaf_outputs: dict[str, str],
    ) -> str:
        """Aggregate leaf outputs using the specified template.

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
