"""Pattern extraction and matching for the GraphBot execution engine."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

import Levenshtein

from core_gb.types import Domain, ExecutionResult, Pattern, TaskNode
from graph.store import GraphStore

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Matches incoming tasks against cached patterns using trigger templates.

    Supports three matching strategies (scored independently, best wins):
    1. Regex-based structural matching (exact template match with slot extraction)
    2. Levenshtein similarity on structural text
    3. Semantic embedding similarity (requires an EmbeddingService)

    Domain scoping: when a domain is provided, only patterns whose
    source_domain matches or is "general" are considered. This prevents
    cross-domain cache pollution (e.g., shell patterns matching creative tasks).

    Slot validation: after matching, any pattern with variable_slots that
    could not be fully filled via regex extraction is rejected. This forces
    decomposition rather than producing output with unfilled slot placeholders.

    Args:
        embedding_service: Optional EmbeddingService for semantic matching.
            When None, only regex and Levenshtein strategies are used.
    """

    def __init__(
        self,
        embedding_service: "EmbeddingService | None" = None,
    ) -> None:
        self._embedding_service = embedding_service

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
        self,
        task: str,
        patterns: list[Pattern],
        threshold: float = 0.7,
        domain: str | None = None,
    ) -> tuple[Pattern, dict[str, str]] | None:
        """Match a task against cached patterns.

        Returns (pattern, variable_bindings) or None if no match.
        variable_bindings maps slot names to extracted values.

        Args:
            task: The task description to match against patterns.
            patterns: List of candidate patterns to score.
            threshold: Minimum weighted score for a match to be accepted.
            domain: When provided, only patterns whose source_domain matches
                this value or is "general" are considered. Prevents
                cross-domain cache pollution.

        Matching strategy:
        1. Filter patterns by domain (if specified)
        2. For each pattern, try to match the trigger template against the task
        3. Extract variable bindings from slots ({slot_0}, {slot_1}, etc.)
        4. Reject patterns with unfilled slots (force decomposition)
        5. Compute similarity between the structural parts
        6. Weight the raw score by the pattern's success rate factor
        7. Deprioritize patterns with success_rate < 20%:
           - 0% success rate (all failures): always skip (force decomposition)
           - <20% with alternatives: skip in favor of better patterns
           - <20% with no alternatives: return with warning
        8. Return best match above threshold
        """
        # --- Domain scoping ---
        if domain is not None:
            patterns = [
                p for p in patterns
                if p.source_domain == domain or p.source_domain == "general"
            ]

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

            # --- Unfilled slot validation ---
            # If the pattern declares variable slots but they were not all
            # filled by regex extraction, reject the match. This prevents
            # producing output with unfilled slot placeholders like
            # "[No data for python_version]".
            if pattern.variable_slots and not self._slots_filled(
                pattern, bindings
            ):
                logger.debug(
                    "Rejecting pattern %s: unfilled slots %s",
                    pattern.id,
                    set(pattern.variable_slots) - set(bindings.keys()),
                )
                continue

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

    @staticmethod
    def _slots_filled(pattern: Pattern, bindings: dict[str, str]) -> bool:
        """Check whether all variable slots have been filled by extraction.

        Returns True if the pattern has no variable slots or all slots have
        corresponding non-empty bindings. Returns False if any slot is
        missing or empty.
        """
        if not pattern.variable_slots:
            return True
        for slot in pattern.variable_slots:
            if slot not in bindings or not bindings[slot]:
                return False
        return True

    def _score_match(
        self, task: str, pattern: Pattern
    ) -> tuple[float, dict[str, str]]:
        """Score a task against a pattern using three strategies.

        Strategies (scored independently, best score wins):
        1. Regex-based: exact template match with slot extraction -> 1.0
        2. Levenshtein: edit-distance similarity on structural text
        3. Semantic embedding: cosine similarity between embeddings

        Returns:
            (max_score, variable_bindings) where max_score is the best
            of the three strategy scores.
        """
        trigger = pattern.trigger
        bindings: dict[str, str] = {}

        # --- Strategy 1: Regex-based matching ---
        # Replace {slot_N} with (.+?) capture groups
        slot_names = sorted(pattern.variable_slots)
        regex_pattern = re.escape(trigger)
        for slot in slot_names:
            regex_pattern = regex_pattern.replace(
                re.escape("{" + slot + "}"), "(.+?)"
            )

        regex_score = 0.0
        try:
            match = re.fullmatch(regex_pattern, task, re.IGNORECASE)
            if match:
                for i, slot in enumerate(slot_names):
                    bindings[slot] = match.group(i + 1)
                regex_score = 1.0
        except re.error:
            pass

        # --- Strategy 2: Levenshtein similarity on structural parts ---
        # Strip slot placeholders from trigger to get structural text
        structural = re.sub(r"\{slot_\d+\}", "", trigger).strip()
        structural = re.sub(r"\s+", " ", structural)

        task_lower = task.lower().strip()
        structural_lower = structural.lower().strip()

        levenshtein_score = 0.0
        if structural_lower:
            levenshtein_score = Levenshtein.ratio(task_lower, structural_lower)

        # --- Strategy 3: Semantic embedding similarity ---
        embedding_score = 0.0
        if self._embedding_service is not None and structural_lower:
            try:
                sim = self._embedding_service.similarity(
                    task_lower, structural_lower
                )
                # Only accept embedding scores above the configured threshold
                if sim >= self._embedding_service.similarity_threshold:
                    embedding_score = sim
                else:
                    logger.debug(
                        "Embedding similarity %.3f below threshold %.3f for "
                        "pattern %s",
                        sim,
                        self._embedding_service.similarity_threshold,
                        pattern.id,
                    )
            except Exception:
                logger.debug(
                    "Embedding similarity computation failed for pattern %s",
                    pattern.id,
                    exc_info=True,
                )

        # Final score: best of the three strategies
        score = max(regex_score, levenshtein_score, embedding_score)

        # If regex matched perfectly, bindings were already extracted above.
        # For non-regex matches, bindings stay empty (no slot extraction possible
        # from fuzzy/semantic matches alone).
        return score, bindings


class PatternExtractor:
    """Extracts reusable execution templates from completed task trees.

    Domain blocklist: tasks whose atomic leaves execute in tool-dependent
    domains (CODE, BROWSER) are excluded from pattern extraction. Their
    outputs are environment-specific and would cause cache pollution if
    generalized into reusable templates.
    """

    # Domains whose outputs depend on external tool execution and should
    # never produce reusable patterns. CODE maps to shell execution in the
    # tool registry; BROWSER performs live web scraping.
    NON_CACHEABLE_DOMAINS: frozenset[Domain] = frozenset({
        Domain.CODE,
        Domain.BROWSER,
    })

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
        - Any atomic leaf is in a non-cacheable domain (CODE, BROWSER)
        - Cannot generalize the task
        """
        if not result.success:
            return None

        leaf_nodes = [n for n in nodes if n.is_atomic]
        if len(leaf_nodes) < 2:
            return None

        # --- Domain blocklist ---
        leaf_domains = {n.domain for n in leaf_nodes}
        blocked = leaf_domains & self.NON_CACHEABLE_DOMAINS
        if blocked:
            logger.info(
                "Skipping pattern extraction: non-cacheable domain(s) %s "
                "in atomic leaves",
                {d.value for d in blocked},
            )
            return None

        # Determine the primary source domain from the atomic leaves
        source_domain = self._determine_source_domain(leaf_nodes)

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
            source_domain=source_domain,
        )

    @staticmethod
    def _determine_source_domain(leaf_nodes: list[TaskNode]) -> str:
        """Determine the primary domain for a set of leaf nodes.

        If all non-SYNTHESIS leaves share a single domain, that domain is
        used. If leaves span multiple non-SYNTHESIS domains, falls back to
        "general". Pure SYNTHESIS tasks return "synthesis".
        """
        non_synthesis = {
            n.domain.value for n in leaf_nodes
            if n.domain != Domain.SYNTHESIS
        }
        if len(non_synthesis) == 1:
            return non_synthesis.pop()
        if len(non_synthesis) == 0:
            return "synthesis"
        return "general"

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
    """Stores and retrieves patterns from the knowledge graph.

    Implements lazy-load caching: patterns are loaded from the graph once
    and cached in memory. The cache is invalidated on save, delete,
    increment_usage, and increment_failure operations (which modify graph
    state).

    Pollution purge: purge_polluted() scans all patterns and removes those
    containing shell-specific markers, unfilled slot indicators, or other
    artifacts that indicate the pattern was extracted from a tool-dependent
    execution and should not be reused.
    """

    # Markers that indicate a pattern was extracted from tool-dependent
    # execution and should not be reused. Checked by purge_polluted().
    POLLUTION_MARKERS: tuple[str, ...] = (
        "python_version",
        "pip_version",
        "node_version",
        "shell",
        "bash",
        "terminal",
        "cmd",
        "/usr/",
        "/bin/",
        "pip install",
        "npm ",
        "git ",
        ".exe",
        ".sh",
        "sudo",
        "chmod",
        "[no data for",
    )

    def __init__(self, store: GraphStore) -> None:
        self._store = store
        self._cache: list[Pattern] | None = None

    def save(self, pattern: Pattern) -> str:
        """Save a pattern to the graph. Returns the pattern ID.

        Invalidates the in-memory cache so the next load_all() re-queries.
        """
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
            "source_domain": pattern.source_domain,
            "created_at": datetime.now(),
        }
        result = self._store.create_node("PatternNode", props)
        self._cache = None  # Invalidate cache on mutation
        return result

    def load_all(self) -> list[Pattern]:
        """Retrieve all patterns from the graph, with lazy-load caching.

        Results are cached in memory. Subsequent calls return the cached
        list without querying the graph. The cache is invalidated by
        save(), delete(), increment_usage(), increment_failure(), or
        invalidate_cache().
        """
        if self._cache is not None:
            return self._cache

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
                source_domain=str(row.get("p.source_domain") or "general"),
            ))
        self._cache = patterns
        return patterns

    def delete(self, pattern_id: str) -> bool:
        """Delete a single pattern by ID. Returns True if it existed.

        Invalidates the in-memory cache so the next load_all() re-queries.
        """
        result = self._store.delete_node("PatternNode", pattern_id)
        self._cache = None  # Invalidate cache on mutation
        return result

    def purge_polluted(self) -> int:
        """Remove patterns with pollution markers from the store.

        Scans all patterns and deletes any whose tree_template or trigger
        contains known pollution indicators (shell commands, file paths,
        unfilled slot syntax, etc.).

        Returns the number of patterns purged.
        """
        patterns = self.load_all()
        purged = 0
        for pattern in patterns:
            template_lower = pattern.tree_template.lower()
            trigger_lower = pattern.trigger.lower()
            combined = template_lower + " " + trigger_lower

            if any(marker in combined for marker in self.POLLUTION_MARKERS):
                logger.info(
                    "Purging polluted pattern %s: %s",
                    pattern.id,
                    pattern.description,
                )
                self.delete(pattern.id)
                purged += 1

        if purged > 0:
            logger.warning(
                "Purged %d polluted pattern(s) from cache", purged
            )
        return purged

    def invalidate_cache(self) -> None:
        """Force the next load_all() to re-query the graph."""
        self._cache = None

    def increment_usage(self, pattern_id: str) -> None:
        """Increment success count and update last_used timestamp.

        Invalidates the in-memory cache so the next load_all() re-queries.
        """
        node = self._store.get_node("PatternNode", pattern_id)
        if node is None:
            return
        current_count = int(node.get("success_count", 0))
        self._store.update_node("PatternNode", pattern_id, {
            "success_count": current_count + 1,
            "last_used": datetime.now(),
        })
        self._cache = None  # Invalidate cache on mutation

    def increment_failure(self, pattern_id: str) -> None:
        """Increment failure count and update last_used timestamp.

        Invalidates the in-memory cache so the next load_all() re-queries.
        """
        node = self._store.get_node("PatternNode", pattern_id)
        if node is None:
            return
        current_count = int(node.get("failure_count", 0))
        self._store.update_node("PatternNode", pattern_id, {
            "failure_count": current_count + 1,
            "last_used": datetime.now(),
        })
        self._cache = None  # Invalidate cache on mutation
