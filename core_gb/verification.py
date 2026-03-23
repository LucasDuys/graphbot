"""Verification layers for DAG node outputs.

Layer 1 (VerificationLayer1): Rule-based format/type checks.
    Runs rule-based checks on every node output before forwarding:
    - Non-empty output
    - Valid JSON when structured output is expected
    - No LLM refusal phrases
    - Reasonable length for complex tasks
    On failure, triggers a single retry with the issues appended as hints
    to the modified prompt.

Layer 2 (VerificationLayer2): Self-consistency via 3-way sampling.
    Executes the same node 3 times in parallel, compares outputs pairwise
    using string similarity (CISC), and selects the highest-agreement output.
    Only runs on nodes with complexity >= configurable threshold (default 3).
    Cost bound: max 3x overhead (no retries within sampling).
    Fallback: all 3 disagree -> return first with low_confidence=True.

Layer 3 (VerificationLayer3): CRITIC-style knowledge graph verification.
    Extracts entities from LLM output using EntityResolver, checks claimed
    relationships and properties against the knowledge graph, and revises
    the output via re-prompt with graph context when inconsistencies are
    found. Opt-in only: runs when verify=True or complexity >= 5.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph.store import GraphStore
    from models.router import ModelRouter

from core_gb.types import CompletionResult, ExecutionResult, TaskNode

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for the multi-layer verification pipeline.

    Controls which verification layers are applied to DAG node outputs,
    and at what complexity thresholds each layer activates.

    Attributes:
        layer1_enabled: Whether Layer 1 (rule-based format/type checks)
            runs on every node. Default True.
        layer2_threshold: Minimum node complexity for Layer 2
            (self-consistency 3-way sampling) to activate. Default 3.
        layer3_threshold: Minimum node complexity for Layer 3
            (CRITIC-style knowledge graph verification). Default 5.
        layer3_opt_in: Whether Layer 3 is enabled at all. Even when
            complexity >= layer3_threshold, Layer 3 only runs if this
            is True. Default False.
        skip_layer1_for_simple: Whether to skip Layer 1 verification for
            complexity=1 tasks. These are trivially simple tasks where
            format verification adds latency without meaningful benefit.
            Default True.
    """

    layer1_enabled: bool = True
    layer2_threshold: int = 3
    layer3_threshold: int = 5
    layer3_opt_in: bool = False
    skip_layer1_for_simple: bool = True


# Refusal phrases that indicate the LLM declined to answer.
# Each pattern is compiled case-insensitive. We match against the start
# of the output (first ~200 chars) to avoid false positives in longer text
# that legitimately discusses these concepts.
_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bI cannot\b", re.IGNORECASE),
    re.compile(r"\bI'm not able\b", re.IGNORECASE),
    re.compile(r"\bI am not able\b", re.IGNORECASE),
    re.compile(r"\bas an AI\b", re.IGNORECASE),
    re.compile(r"\bI'm sorry,?\s+but I cannot\b", re.IGNORECASE),
    re.compile(r"\bI'm unable to\b", re.IGNORECASE),
    re.compile(r"\bI am unable to\b", re.IGNORECASE),
]

# Minimum word count for complex tasks (complexity >= 2).
# A one-or-two-word answer is almost certainly inadequate for a complex task.
_MIN_WORDS_COMPLEX: int = 5

# Maximum length of text prefix inspected for refusal phrases. Longer text
# that happens to contain "I cannot" in the middle of a legitimate answer
# should not be flagged.
_REFUSAL_CHECK_PREFIX_LEN: int = 200


@dataclass
class VerificationResult:
    """Result of a Layer 1 verification check.

    Attributes:
        passed: Whether all checks passed.
        issues: List of human-readable issue descriptions.
        layer: Verification layer number (always 1 for this class).
        retry_count: Number of retries that were triggered.
    """

    passed: bool
    issues: list[str]
    layer: int = 1
    retry_count: int = 0


def aggregate_verification_stats(
    results: tuple[VerificationResult, ...] | list[VerificationResult],
) -> dict[str, object]:
    """Compute aggregate statistics from a sequence of VerificationResult objects.

    Returns a dict with:
        total_verifications: Total number of verification passes.
        pass_rate: Fraction of verifications that passed (1.0 if none).
        total_retries: Sum of retry_count across all results.
        per_layer: Dict mapping layer number to per-layer breakdown with
            keys 'total', 'passed', and 'retries'.

    Args:
        results: Sequence of VerificationResult objects to aggregate.

    Returns:
        Dict with aggregate stats.
    """
    total = len(results)
    if total == 0:
        return {
            "total_verifications": 0,
            "pass_rate": 1.0,
            "total_retries": 0,
            "per_layer": {},
        }

    passed_count = sum(1 for vr in results if vr.passed)
    total_retries = sum(vr.retry_count for vr in results)

    per_layer: dict[int, dict[str, int]] = {}
    for vr in results:
        if vr.layer not in per_layer:
            per_layer[vr.layer] = {"total": 0, "passed": 0, "retries": 0}
        per_layer[vr.layer]["total"] += 1
        if vr.passed:
            per_layer[vr.layer]["passed"] += 1
        per_layer[vr.layer]["retries"] += vr.retry_count

    return {
        "total_verifications": total,
        "pass_rate": passed_count / total,
        "total_retries": total_retries,
        "per_layer": per_layer,
    }


class VerificationLayer1:
    """Layer 1 format/type verification for DAG node outputs.

    Performs rule-based checks on LLM output:
    1. Non-empty output
    2. Valid JSON when structured output is expected
    3. No refusal phrases
    4. Reasonable length for complex tasks

    On failure, can trigger a single retry with the issues appended as hints.
    """

    def verify(
        self,
        output: str,
        expects_json: bool,
        complexity: int,
    ) -> VerificationResult:
        """Run all Layer 1 checks on a node output.

        Args:
            output: The raw text output from the LLM.
            expects_json: Whether the output should be valid JSON.
            complexity: Task complexity level (1 = simple, higher = complex).

        Returns:
            VerificationResult with pass/fail status and any issues found.
        """
        issues: list[str] = []

        # Check 1: Non-empty output
        if not output or not output.strip():
            issues.append("Output is empty or whitespace-only.")
            # Short-circuit: no point checking further on empty output
            return VerificationResult(passed=False, issues=issues)

        stripped = output.strip()

        # Check 2: Valid JSON when expected
        if expects_json:
            try:
                json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                issues.append(
                    "Expected valid JSON output but received non-JSON text."
                )

        # Check 3: Refusal phrase detection
        # Only check the first N characters to avoid false positives in
        # longer legitimate content.
        prefix = stripped[:_REFUSAL_CHECK_PREFIX_LEN]
        for pattern in _REFUSAL_PATTERNS:
            if pattern.search(prefix):
                issues.append(
                    f"Output contains refusal phrase matching: {pattern.pattern}"
                )
                break  # One refusal issue is enough

        # Check 4: Reasonable length for complex tasks
        if complexity >= 2:
            word_count = len(stripped.split())
            if word_count < _MIN_WORDS_COMPLEX:
                issues.append(
                    f"Output too short for complex task (complexity={complexity}): "
                    f"only {word_count} word(s), expected at least {_MIN_WORDS_COMPLEX}."
                )

        passed = len(issues) == 0
        return VerificationResult(passed=passed, issues=issues)

    async def verify_and_retry(
        self,
        output: str,
        node: TaskNode,
        executor: object,
        expects_json: bool,
    ) -> tuple[str, VerificationResult]:
        """Verify output and retry once on failure.

        If the initial output fails verification, triggers a single retry
        with the issues appended as hints to the prompt.

        Args:
            output: The raw text output from the initial execution.
            node: The TaskNode being executed.
            executor: The executor object with an async execute() method.
            expects_json: Whether the output should be valid JSON.

        Returns:
            Tuple of (final_output, verification_result).
            The verification_result.retry_count indicates whether a retry
            was triggered (0 = no retry, 1 = one retry).
        """
        vr = self.verify(output, expects_json=expects_json, complexity=node.complexity)

        if vr.passed:
            return output, vr

        # Build retry prompt with issues as hints
        logger.info(
            "Verification failed for node %s: %s -- triggering retry",
            node.id,
            "; ".join(vr.issues),
        )

        hints = "\n".join(f"- {issue}" for issue in vr.issues)
        retry_prompt = (
            f"{node.description}\n\n"
            f"[VERIFICATION HINT: Your previous response had issues. "
            f"Please address the following and try again:]\n{hints}"
        )

        provides_keys = list(node.provides) if node.provides else None

        result: ExecutionResult = await executor.execute(
            retry_prompt, node.complexity, provides_keys=provides_keys
        )

        retry_output = result.output if result.success else output

        # Verify the retry output (but do not retry again)
        retry_vr = self.verify(
            retry_output, expects_json=expects_json, complexity=node.complexity
        )
        retry_vr.retry_count = 1

        if not retry_vr.passed:
            logger.warning(
                "Retry also failed verification for node %s: %s",
                node.id,
                "; ".join(retry_vr.issues),
            )

        return retry_output, retry_vr


# ---------------------------------------------------------------------------
# Layer 2: Self-consistency via 3-way sampling
# ---------------------------------------------------------------------------

# Agreement score threshold below which the result is considered low-confidence.
# When all 3 samples disagree, the best pairwise score will be below this.
_LOW_CONFIDENCE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class SamplingResult(CompletionResult):
    """Result from Layer 2 self-consistency verification.

    Extends CompletionResult with metadata about the 3-way sampling process:
    how many samples were taken, the inter-sample agreement score, and whether
    the result should be treated as low-confidence.

    Attributes:
        sample_count: Number of samples taken (1 if below complexity threshold,
            3 otherwise).
        agreement_score: Highest pairwise agreement score among samples
            (0.0 - 1.0).
        low_confidence: True if all 3 samples disagreed and the result is
            unreliable.
    """

    sample_count: int = 1
    agreement_score: float = 1.0
    low_confidence: bool = False


class VerificationLayer2:
    """Layer 2 self-consistency verification via 3-way sampling.

    Executes the same node 3 times in parallel via asyncio.gather, compares
    outputs pairwise using string similarity (CISC -- Confidence-weighted
    Intersample Consistency), and selects the output with the highest total
    agreement.

    Design constraints:
    - Only runs on nodes with complexity >= complexity_threshold (default 3).
    - Cost bound: exactly 3 parallel calls, no retries within sampling.
    - Fallback: if all 3 disagree, returns the first sample with
      low_confidence=True.

    Requires access to a ModelRouter to make the 3 parallel calls.
    """

    def __init__(
        self,
        router: ModelRouter,
        complexity_threshold: int = 3,
    ) -> None:
        self._router = router
        self._complexity_threshold = complexity_threshold

    @property
    def complexity_threshold(self) -> int:
        """Return the complexity threshold for triggering 3-way sampling."""
        return self._complexity_threshold

    async def verify(
        self,
        task: TaskNode,
        messages: list[dict],
        **kwargs: object,
    ) -> SamplingResult:
        """Run self-consistency verification on a task.

        If the task complexity is below the threshold, performs a single call
        and returns the result directly. Otherwise, fans out 3 parallel calls
        and selects the best output using CISC.

        Args:
            task: The TaskNode to execute.
            messages: Chat messages to send to the model.
            **kwargs: Extra arguments forwarded to ModelRouter.route().

        Returns:
            SamplingResult with the selected content and agreement metadata.
        """
        if task.complexity < self._complexity_threshold:
            return await self._single_call(task, messages, **kwargs)

        return await self._three_way_sample(task, messages, **kwargs)

    async def _single_call(
        self,
        task: TaskNode,
        messages: list[dict],
        **kwargs: object,
    ) -> SamplingResult:
        """Execute a single call (below complexity threshold)."""
        result = await self._router.route(task, messages, **kwargs)
        return SamplingResult(
            content=result.content,
            model=result.model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
            cost=result.cost,
            logprobs=result.logprobs,
            sample_count=1,
            agreement_score=1.0,
            low_confidence=False,
        )

    async def _three_way_sample(
        self,
        task: TaskNode,
        messages: list[dict],
        **kwargs: object,
    ) -> SamplingResult:
        """Fan-out 3 parallel calls and select the best by CISC agreement.

        All 3 calls are launched concurrently via asyncio.gather. No retries
        are performed within sampling to maintain the max 3x cost bound.
        """
        results: list[CompletionResult] = await asyncio.gather(
            self._router.route(task, messages, **kwargs),
            self._router.route(task, messages, **kwargs),
            self._router.route(task, messages, **kwargs),
        )

        outputs = [r.content for r in results]
        best_idx, agreement_score = self._select_best(outputs)
        low_confidence = agreement_score < _LOW_CONFIDENCE_THRESHOLD

        # If all disagree, fall back to the first sample
        if low_confidence:
            best_idx = 0

        best = results[best_idx]

        # Aggregate tokens and cost from all 3 samples
        total_tokens_in = sum(r.tokens_in for r in results)
        total_tokens_out = sum(r.tokens_out for r in results)
        total_cost = sum(r.cost for r in results)
        max_latency = max(r.latency_ms for r in results)

        return SamplingResult(
            content=best.content,
            model=best.model,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            latency_ms=max_latency,
            cost=total_cost,
            logprobs=best.logprobs,
            sample_count=3,
            agreement_score=agreement_score,
            low_confidence=low_confidence,
        )

    def _select_best(self, outputs: list[str]) -> tuple[int, float]:
        """Select the output with the highest total pairwise agreement.

        Computes pairwise similarity between all outputs and returns the
        index of the output with the highest sum of pairwise similarities,
        along with the normalized agreement score.

        Args:
            outputs: List of output strings from the 3 samples.

        Returns:
            Tuple of (best_index, agreement_score).
            agreement_score is the mean of the best output's pairwise
            similarities, in [0.0, 1.0].
        """
        n = len(outputs)
        if n == 0:
            return 0, 0.0

        # Compute pairwise similarity matrix
        similarity: list[list[float]] = [
            [0.0] * n for _ in range(n)
        ]
        for i in range(n):
            similarity[i][i] = 1.0
            for j in range(i + 1, n):
                score = self._pairwise_similarity(outputs[i], outputs[j])
                similarity[i][j] = score
                similarity[j][i] = score

        # Sum pairwise similarities for each output (excluding self)
        totals = [
            sum(similarity[i][j] for j in range(n) if j != i)
            for i in range(n)
        ]

        best_idx = max(range(n), key=lambda i: totals[i])
        # Normalize: mean of pairwise similarities with other outputs
        agreement_score = totals[best_idx] / (n - 1) if n > 1 else 1.0

        return best_idx, agreement_score

    @staticmethod
    def _pairwise_similarity(a: str, b: str) -> float:
        """Compute string similarity between two outputs using SequenceMatcher.

        Uses difflib.SequenceMatcher which provides a ratio in [0.0, 1.0]
        based on the longest contiguous matching subsequence. This is a
        reasonable proxy for semantic similarity at low computational cost.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Similarity ratio in [0.0, 1.0].
        """
        return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Layer 3: CRITIC-style knowledge graph verification
# ---------------------------------------------------------------------------

# Minimum confidence score from EntityResolver to consider a mention as
# a genuine entity reference in the output text.
_ENTITY_CONFIDENCE_THRESHOLD: float = 0.7

# Property keys on graph nodes that contain verifiable factual claims.
# When two entities appear in the same output and both exist in the graph,
# we compare these property values against what the output states.
_VERIFIABLE_PROPERTIES: tuple[str, ...] = (
    "language",
    "framework",
    "status",
    "role",
    "institution",
    "type",
    "relationship",
    "platform",
    "path",
)


@dataclass
class Layer3Result:
    """Result of a Layer 3 knowledge graph verification check.

    Attributes:
        passed: Whether all claims in the output are consistent with the graph.
        issues: List of human-readable inconsistency descriptions.
        entities_checked: Number of entities that were verified against the graph.
        revised: Whether the output was revised via re-prompt.
        layer: Verification layer number (always 3).
    """

    passed: bool
    issues: list[str]
    entities_checked: int
    revised: bool
    layer: int = 3


class VerificationLayer3:
    """Layer 3 CRITIC-style knowledge graph verification.

    Extracts entities mentioned in LLM output using EntityResolver, checks
    claimed relationships and properties against the knowledge graph, and
    optionally revises the output via re-prompt with graph context when
    inconsistencies are found.

    Design constraints:
    - Opt-in only: runs when verify=True or complexity >= complexity_threshold
      (default 5).
    - Requires access to GraphStore (for graph queries) and ModelRouter (for
      re-prompt on inconsistency).
    - Entity extraction uses EntityResolver (no LLM calls for extraction).
    - Revision uses a single model call with graph context injected.
    """

    def __init__(
        self,
        store: GraphStore,
        router: ModelRouter,
        complexity_threshold: int = 5,
    ) -> None:
        self._store = store
        self._router = router
        self._complexity_threshold = complexity_threshold

    @property
    def complexity_threshold(self) -> int:
        """Return the complexity threshold for auto-activation."""
        return self._complexity_threshold

    def _should_run(self, task: TaskNode, verify: bool) -> bool:
        """Determine whether Layer 3 should run for this task.

        Returns True when verify=True or task complexity >= threshold.
        """
        if verify:
            return True
        return task.complexity >= self._complexity_threshold

    async def verify(
        self,
        output: str,
        task: TaskNode,
        verify: bool = False,
    ) -> Layer3Result:
        """Verify LLM output against the knowledge graph.

        If gating conditions are not met (verify=False and complexity below
        threshold), returns a passing result without checking.

        Args:
            output: The raw text output from the LLM.
            task: The TaskNode being verified.
            verify: Explicit opt-in flag for Layer 3.

        Returns:
            Layer3Result with pass/fail status, issues, and entity count.
        """
        if not self._should_run(task, verify):
            return Layer3Result(
                passed=True, issues=[], entities_checked=0, revised=False,
            )

        if not output or not output.strip():
            return Layer3Result(
                passed=True, issues=[], entities_checked=0, revised=False,
            )

        # Step 1: Extract entities from output text
        entities = self._extract_entities(output)
        if not entities:
            return Layer3Result(
                passed=True, issues=[], entities_checked=0, revised=False,
            )

        # Step 2: Verify claims against the graph
        issues = self._check_claims(output, entities)

        passed = len(issues) == 0
        return Layer3Result(
            passed=passed,
            issues=issues,
            entities_checked=len(entities),
            revised=False,
        )

    async def verify_and_revise(
        self,
        output: str,
        task: TaskNode,
        verify: bool = False,
    ) -> tuple[str, Layer3Result]:
        """Verify output and revise via re-prompt on inconsistency.

        If the output is consistent with the graph (or gating conditions are
        not met), returns the original output unchanged.

        If inconsistencies are found, builds a re-prompt with the original
        task description, the issues found, and graph context, then calls
        the model to produce a revised output.

        Args:
            output: The raw text output from the LLM.
            task: The TaskNode being verified.
            verify: Explicit opt-in flag for Layer 3.

        Returns:
            Tuple of (final_output, Layer3Result).
        """
        result = await self.verify(output, task, verify=verify)

        if result.passed or result.entities_checked == 0:
            return output, result

        # Build graph context for revision
        entity_ids = [eid for eid, _conf in self._extract_entities(output)]
        graph_context = self._store.get_context(entity_ids)
        context_text = graph_context.format()

        # Build revision prompt with issues and graph context
        issues_text = "\n".join(f"- {issue}" for issue in result.issues)
        revision_prompt = (
            f"{task.description}\n\n"
            f"[KNOWLEDGE GRAPH VERIFICATION: The previous response contained "
            f"factual inconsistencies with the known knowledge graph. "
            f"Please revise the response to be consistent with the following "
            f"verified facts.]\n\n"
            f"Issues found:\n{issues_text}\n\n"
            f"Verified graph context:\n{context_text}\n\n"
            f"Original response to revise:\n{output}"
        )

        messages = [{"role": "user", "content": revision_prompt}]

        logger.info(
            "Layer 3 verification found %d inconsistencies for node %s, "
            "triggering revision",
            len(result.issues),
            task.id,
        )

        try:
            completion = await self._router.route(task, messages)
            revised_output = completion.content
            result.revised = True
            return revised_output, result
        except Exception as exc:
            logger.warning(
                "Layer 3 revision failed for node %s: %s, returning original",
                task.id,
                exc,
            )
            return output, result

    def _extract_entities(
        self, text: str,
    ) -> list[tuple[str, float]]:
        """Extract entity mentions from text using EntityResolver.

        Splits the text into candidate phrases (n-grams of 1-3 words from
        each sentence) and resolves each against the graph. Returns
        deduplicated entity matches above the confidence threshold.

        Args:
            text: The output text to extract entities from.

        Returns:
            List of (entity_id, confidence) tuples, deduplicated by entity_id.
        """
        from graph.resolver import EntityResolver

        resolver = EntityResolver(self._store)

        # Extract candidate phrases: split into sentences, then n-grams
        sentences = re.split(r'[.!?;]\s+', text.strip())
        candidates: list[str] = []
        for sentence in sentences:
            words = sentence.split()
            # Generate 1-gram, 2-gram, and 3-gram candidates
            for n in range(1, min(4, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    # Skip very short or common words as single-word candidates
                    if n == 1 and len(phrase) <= 2:
                        continue
                    candidates.append(phrase)

        # Resolve each candidate and collect best matches
        seen: dict[str, float] = {}
        for candidate in candidates:
            matches = resolver.resolve(candidate, top_k=1)
            for eid, confidence in matches:
                if confidence >= _ENTITY_CONFIDENCE_THRESHOLD:
                    if confidence > seen.get(eid, 0.0):
                        seen[eid] = confidence

        return sorted(seen.items(), key=lambda x: x[1], reverse=True)

    def _check_claims(
        self,
        output: str,
        entities: list[tuple[str, float]],
    ) -> list[str]:
        """Check claims in the output against known graph properties.

        For each extracted entity, loads its properties from the graph and
        checks whether the output text contradicts any verifiable property
        values. Uses a two-pass approach:

        1. For each entity, collect its verifiable graph properties.
        2. For each property, use targeted regex patterns to extract the
           claimed value from sentences mentioning the entity.
        3. Flag only when the output explicitly states a different value
           for a property that the graph defines.

        Args:
            output: The LLM output text.
            entities: List of (entity_id, confidence) from entity extraction.

        Returns:
            List of human-readable inconsistency descriptions.
        """
        issues: list[str] = []
        output_lower = output.lower()

        for entity_id, _confidence in entities:
            node_info = self._store._find_node_table(entity_id)
            if node_info is None:
                continue

            table_name, props = node_info
            entity_name = str(
                props.get("name", props.get("path", entity_id))
            )
            entity_lower = entity_name.lower()

            # Skip entity if not mentioned in the output
            if entity_lower not in output_lower:
                continue

            # Extract sentences mentioning this entity
            entity_sentences = self._find_entity_sentences(
                output_lower, entity_lower,
            )
            if not entity_sentences:
                continue

            # Check each verifiable property
            for prop_key in _VERIFIABLE_PROPERTIES:
                graph_value = props.get(prop_key)
                if not graph_value or not str(graph_value).strip():
                    continue

                graph_value_str = str(graph_value).strip()
                graph_value_lower = graph_value_str.lower()

                contradiction = self._detect_property_contradiction(
                    entity_sentences,
                    entity_lower,
                    prop_key,
                    graph_value_lower,
                )
                if contradiction:
                    issues.append(
                        f"Claimed {prop_key} for {entity_name} contradicts "
                        f"knowledge graph: graph says "
                        f"{prop_key}='{graph_value_str}', but output states: "
                        f"{contradiction}"
                    )

        return issues

    @staticmethod
    def _find_entity_sentences(
        output_lower: str,
        entity_lower: str,
    ) -> list[str]:
        """Find sentences in the output that mention a specific entity.

        Args:
            output_lower: Lowercased full output text.
            entity_lower: Lowercased entity name to search for.

        Returns:
            List of lowercased sentences containing the entity name.
        """
        sentences = re.split(r'[.!?;]\s+', output_lower)
        return [s for s in sentences if entity_lower in s]

    @staticmethod
    def _detect_property_contradiction(
        entity_sentences: list[str],
        entity_lower: str,
        prop_key: str,
        graph_value_lower: str,
    ) -> str | None:
        """Detect if entity sentences contradict a known property value.

        Uses targeted regex patterns per property type to extract claimed
        values from sentences. Only flags a contradiction when the output
        explicitly states a different concrete value for the property.

        The patterns are intentionally narrow to minimize false positives.
        Each pattern captures the claimed value in group 1. The captured
        value is compared against the graph value; if they differ (and the
        captured value is not a filler/stop word), a contradiction is
        reported.

        Args:
            entity_sentences: Lowercased sentences mentioning the entity.
            entity_lower: Lowercased entity name.
            prop_key: Property key (e.g., "language", "role").
            graph_value_lower: Known correct value from the graph (lowercased).

        Returns:
            The contradicting claim text if found, None otherwise.
        """
        # Patterns per property type. Each pattern must capture the claimed
        # value as group 1. Patterns are narrow by design: they only match
        # explicit property assertions, not incidental word usage.
        claim_patterns: dict[str, list[str]] = {
            "language": [
                r"(?:written|coded|programmed|developed|built)\s+in\s+(\w+)",
                r"(?:uses?|using)\s+(\w+)\s+(?:as\s+(?:its?\s+)?)?(?:language|programming)",
            ],
            "framework": [
                r"(?:built|developed|made)\s+(?:with|on|using)\s+(?:the\s+)?(\w+)\s+framework",
                r"(?:uses?|using)\s+(?:the\s+)?(\w+)\s+framework",
            ],
            "role": [
                r"(?:" + re.escape(entity_lower) + r")\s+(?:is|works\s+as)\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
            ],
            "institution": [
                r"(?:" + re.escape(entity_lower) + r")\s+(?:studies|studied|attends?|enrolled)\s+at\s+(\S+)",
                r"(?:" + re.escape(entity_lower) + r")\s+(?:is\s+)?(?:from|at)\s+(\S+)",
            ],
            "status": [
                r"(?:" + re.escape(entity_lower) + r")\s+(?:is\s+)?(?:currently\s+)?(\w+)\s+(?:project|status)",
                r"status\s+(?:is\s+|of\s+\S+\s+is\s+)?(\w+)",
            ],
        }

        patterns = claim_patterns.get(prop_key, [])
        if not patterns:
            return None

        # Stop words: values that are never real property values
        stop_words = frozenset({
            "a", "an", "the", "is", "are", "was", "were", "has", "have",
            "had", "and", "or", "but", "its", "their", "his", "her", "it",
            "this", "that", "not", "also", "which", "who", "whom", "with",
            "from", "for", "by", "to", "in", "on", "at", "of", "be",
            "been", "being", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall",
        })

        for sentence in entity_sentences:
            for pattern in patterns:
                for match in re.finditer(pattern, sentence):
                    claimed_value = match.group(1).strip().lower()

                    # Skip stop words
                    if claimed_value in stop_words:
                        continue

                    # Skip if the claimed value matches the graph value
                    if (
                        claimed_value == graph_value_lower
                        or claimed_value in graph_value_lower
                        or graph_value_lower in claimed_value
                    ):
                        continue

                    # Found a genuine contradiction
                    return match.group(0)

        return None
