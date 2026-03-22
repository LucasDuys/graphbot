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
    from models.router import ModelRouter

from core_gb.types import CompletionResult, ExecutionResult, TaskNode

logger = logging.getLogger(__name__)

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
