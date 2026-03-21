"""Pattern extraction and matching for the GraphBot execution engine."""

from __future__ import annotations

import json
import re
import uuid

import Levenshtein

from core_gb.types import ExecutionResult, Pattern, TaskNode


class PatternMatcher:
    """Matches incoming tasks against cached patterns using trigger templates."""

    def match(
        self, task: str, patterns: list[Pattern], threshold: float = 0.7
    ) -> tuple[Pattern, dict[str, str]] | None:
        """Match a task against cached patterns.

        Returns (pattern, variable_bindings) or None if no match.
        variable_bindings maps slot names to extracted values.

        Matching strategy:
        1. For each pattern, try to match the trigger template against the task
        2. Extract variable bindings from slots ({slot_0}, {slot_1}, etc.)
        3. Compute similarity between the structural parts
        4. Return best match above threshold
        """
        best_match: Pattern | None = None
        best_score = 0.0
        best_bindings: dict[str, str] = {}

        for pattern in patterns:
            score, bindings = self._score_match(task, pattern)
            if score >= threshold and score > best_score:
                best_score = score
                best_match = pattern
                best_bindings = bindings

        if best_match is not None:
            return best_match, best_bindings
        return None

    def _score_match(
        self, task: str, pattern: Pattern
    ) -> tuple[float, dict[str, str]]:
        trigger = pattern.trigger
        bindings: dict[str, str] = {}

        # Try regex-based matching: replace {slot_N} with (.+?) capture groups
        slot_names = sorted(pattern.variable_slots)
        regex_pattern = re.escape(trigger)
        for slot in slot_names:
            regex_pattern = regex_pattern.replace(
                re.escape("{" + slot + "}"), "(.+?)"
            )

        try:
            match = re.fullmatch(regex_pattern, task, re.IGNORECASE)
            if match:
                for i, slot in enumerate(slot_names):
                    bindings[slot] = match.group(i + 1)
                return 1.0, bindings
        except re.error:
            pass

        # Fallback: Levenshtein similarity on structural parts
        # Strip slot placeholders from trigger to get structural text
        structural = re.sub(r"\{slot_\d+\}", "", trigger).strip()
        structural = re.sub(r"\s+", " ", structural)

        task_lower = task.lower().strip()
        structural_lower = structural.lower().strip()

        if not structural_lower:
            return 0.0, {}

        score: float = Levenshtein.ratio(task_lower, structural_lower)
        return score, bindings


class PatternExtractor:
    """Extracts reusable execution templates from completed task trees."""

    def extract(
        self,
        task: str,
        nodes: list[TaskNode],
        result: ExecutionResult,
    ) -> Pattern | None:
        """Extract a pattern from a successful multi-node execution.

        Returns None if:
        - Execution failed
        - Only 1 leaf node (not worth caching)
        - Cannot generalize the task
        """
        if not result.success:
            return None

        leaf_nodes = [n for n in nodes if n.is_atomic]
        if len(leaf_nodes) < 2:
            return None

        trigger, slots, bindings = self._generalize(task, leaf_nodes)

        tree_template = self._serialize_template(nodes, bindings)

        return Pattern(
            id=str(uuid.uuid4()),
            trigger=trigger,
            description=f"Pattern for: {task[:80]}",
            variable_slots=tuple(sorted(slots)),
            tree_template=tree_template,
            success_count=1,
            avg_tokens=float(result.total_tokens),
            avg_latency_ms=result.total_latency_ms,
        )

    def _generalize(
        self,
        task: str,
        leaves: list[TaskNode],
    ) -> tuple[str, set[str], dict[str, str]]:
        """Generalize a task description by replacing specific entities with slots.

        Strategy:
        1. Find words in leaf descriptions that vary between leaves
        2. Those varying words are likely entity-specific (city names, file names, etc.)
        3. Replace them with numbered slots: {slot_0}, {slot_1}, etc.

        Returns: (trigger_template, slot_names, bindings: {slot_name: original_value})
        """
        leaf_words = [set(leaf.description.lower().split()) for leaf in leaves]

        if leaf_words:
            common = (
                leaf_words[0].intersection(*leaf_words[1:])
                if len(leaf_words) > 1
                else leaf_words[0]
            )
        else:
            common = set()

        bindings: dict[str, str] = {}
        slot_names: set[str] = set()
        trigger = task

        slot_counter = 0
        for leaf in leaves:
            leaf_specific = set(leaf.description.lower().split()) - common
            for word in leaf_specific:
                if len(word) >= 3 and word in task.lower():
                    slot_name = f"slot_{slot_counter}"
                    slot_counter += 1
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    match = pattern.search(trigger)
                    if match:
                        original = match.group()
                        trigger = (
                            trigger[: match.start()]
                            + "{" + slot_name + "}"
                            + trigger[match.end() :]
                        )
                        bindings[slot_name] = original
                        slot_names.add(slot_name)

        return trigger, slot_names, bindings

    def _serialize_template(
        self,
        nodes: list[TaskNode],
        bindings: dict[str, str],
    ) -> str:
        """Serialize the tree structure as a JSON template with variable slots."""
        template_nodes: list[dict[str, object]] = []
        for node in nodes:
            desc = node.description
            for slot, value in bindings.items():
                desc = desc.replace(value, "{" + slot + "}")

            template_nodes.append({
                "description": desc,
                "domain": node.domain.value,
                "is_atomic": node.is_atomic,
                "complexity": node.complexity,
                "provides": list(node.provides),
                "consumes": list(node.consumes),
            })
        return json.dumps(template_nodes)
