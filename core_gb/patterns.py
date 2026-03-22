"""Pattern extraction and matching for the GraphBot execution engine."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

import Levenshtein

from core_gb.types import ExecutionResult, Pattern, TaskNode
from graph.store import GraphStore

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Matches incoming tasks against cached patterns using trigger templates."""

    @staticmethod
    def _success_rate_factor(pattern: Pattern) -> float:
        """Compute a weighting factor based on the pattern's success rate.

        Returns a value in (0.0, 1.0]:
        - 1.0 if the pattern has never been used (no history to penalize)
        - success_count / total_uses otherwise

        This ensures patterns with high failure rates are penalized in scoring
        while brand-new patterns receive no penalty.
        """
        total = pattern.success_count + pattern.failure_count
        if total == 0:
            return 1.0
        return pattern.success_count / total

    # Patterns below this success rate are deprioritized (skipped when
    # alternatives exist, returned with a warning otherwise).
    LOW_SUCCESS_THRESHOLD: float = 0.20

    def _success_rate(self, pattern: Pattern) -> float | None:
        """Compute the success rate for a pattern.

        Returns None if the pattern has never been executed (neutral),
        otherwise returns success_count / (success_count + failure_count).
        """
        total = pattern.success_count + pattern.failure_count
        if total == 0:
            return None
        return pattern.success_count / total

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
        4. Weight the raw score by the pattern's success rate factor
        5. Deprioritize patterns with success_rate < 20%:
           - 0% success rate (all failures): always skip (force decomposition)
           - <20% with alternatives: skip in favor of better patterns
           - <20% with no alternatives: return with warning
        6. Return best match above threshold
        """
        best_match: Pattern | None = None
        best_score = 0.0
        best_bindings: dict[str, str] = {}

        # Track the best low-rate fallback in case no good patterns match
        low_rate_fallback: Pattern | None = None
        low_rate_fallback_score = 0.0
        low_rate_fallback_bindings: dict[str, str] = {}

        for pattern in patterns:
            success_rate = self._success_rate(pattern)

            # 0% success rate with actual failures: skip entirely to force
            # decomposition instead of reusing a known-bad pattern
            if success_rate is not None and success_rate == 0.0:
                logger.debug(
                    "Skipping pattern %s (0%% success rate, %d failures)",
                    pattern.id,
                    pattern.failure_count,
                )
                continue

            raw_score, bindings = self._score_match(task, pattern)
            weighted_score = raw_score * self._success_rate_factor(pattern)

            # Low success rate (<20%): track as fallback but do not select
            # as best match -- prefer alternatives if they exist
            if (
                success_rate is not None
                and success_rate < self.LOW_SUCCESS_THRESHOLD
            ):
                if (
                    weighted_score >= threshold
                    and weighted_score > low_rate_fallback_score
                ):
                    low_rate_fallback = pattern
                    low_rate_fallback_score = weighted_score
                    low_rate_fallback_bindings = bindings
                continue

            if weighted_score >= threshold and weighted_score > best_score:
                best_score = weighted_score
                best_match = pattern
                best_bindings = bindings

        if best_match is not None:
            return best_match, best_bindings

        # No good patterns matched. Fall back to a low-rate pattern if one
        # exists -- it is better than nothing, but log a warning.
        if low_rate_fallback is not None:
            rate = self._success_rate(low_rate_fallback)
            logger.warning(
                "Using pattern %s despite low success rate (%.1f%%). "
                "No better alternatives available.",
                low_rate_fallback.id,
                (rate or 0.0) * 100,
            )
            return low_rate_fallback, low_rate_fallback_bindings

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


class PatternStore:
    """Stores and retrieves patterns from the knowledge graph."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def save(self, pattern: Pattern) -> str:
        """Save a pattern to the graph. Returns the pattern ID."""
        props: dict[str, object] = {
            "id": pattern.id,
            "trigger_template": pattern.trigger,
            "description": pattern.description,
            "variable_slots": json.dumps(list(pattern.variable_slots)),
            "success_count": pattern.success_count,
            "failure_count": pattern.failure_count,
            "avg_tokens": pattern.avg_tokens,
            "avg_latency_ms": pattern.avg_latency_ms,
            "tree_template": pattern.tree_template,
            "created_at": datetime.now(),
        }
        return self._store.create_node("PatternNode", props)

    def load_all(self) -> list[Pattern]:
        """Retrieve all patterns from the graph."""
        rows = self._store.query("MATCH (p:PatternNode) RETURN p.*")
        patterns: list[Pattern] = []
        for row in rows:
            pid = row.get("p.id", "")
            patterns.append(Pattern(
                id=str(pid),
                trigger=str(row.get("p.trigger_template", "")),
                description=str(row.get("p.description", "")),
                variable_slots=tuple(json.loads(row.get("p.variable_slots", "[]"))),
                tree_template=str(row.get("p.tree_template", "")),
                success_count=int(row.get("p.success_count") or 0),
                failure_count=int(row.get("p.failure_count") or 0),
                avg_tokens=float(row.get("p.avg_tokens", 0.0)),
                avg_latency_ms=float(row.get("p.avg_latency_ms", 0.0)),
            ))
        return patterns

    def increment_usage(self, pattern_id: str) -> None:
        """Increment success count and update last_used timestamp."""
        node = self._store.get_node("PatternNode", pattern_id)
        if node is None:
            return
        current_count = int(node.get("success_count", 0))
        self._store.update_node("PatternNode", pattern_id, {
            "success_count": current_count + 1,
            "last_used": datetime.now(),
        })

    def increment_failure(self, pattern_id: str) -> None:
        """Increment failure count and update last_used timestamp."""
        node = self._store.get_node("PatternNode", pattern_id)
        if node is None:
            return
        current_count = int(node.get("failure_count", 0))
        self._store.update_node("PatternNode", pattern_id, {
            "failure_count": current_count + 1,
            "last_used": datetime.now(),
        })
