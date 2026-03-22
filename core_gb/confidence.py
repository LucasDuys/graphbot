"""Confidence estimation for cascade routing results.

Estimates how likely a completion result is to be correct and useful,
combining token log-probabilities (where available) with heuristic
fallbacks: content length scoring, refusal detection, and structured
output validation.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.router import CascadeConfig

from core_gb.types import CompletionResult, TaskNode

logger = logging.getLogger(__name__)

# Length thresholds
_MIN_CONTENT_LENGTH = 10
_SHORT_CONTENT_LENGTH = 50
_MEDIUM_CONTENT_LENGTH = 200

# Token count thresholds
_MIN_TOKEN_COUNT = 5

# Refusal phrases (lowercased). Presence of any of these in the
# response text signals a model refusal.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i'm not able",
    "i am not able",
    "i'm unable",
    "i am unable",
    "i apologize",
    "i'm sorry, but i can't",
    "as an ai",
    "as a language model",
    "as an ai language model",
)

# Refusal penalty: multiplicative factor applied when a refusal is detected.
_REFUSAL_PENALTY = 0.3


class ConfidenceEstimator:
    """Estimate confidence of an LLM completion result.

    Scoring approach:

    1. Compute a base score from additive quality signals:
       - Content length (weight 0.35)
       - Token count (weight 0.35)
       - Log-probabilities when available (weight 0.30; redistributed
         to content/token dimensions when absent)

    2. Apply multiplicative penalties:
       - Refusal detection: if a refusal phrase is found, the score is
         multiplied by ``_REFUSAL_PENALTY`` (0.3).
       - Structured output validity: if the task declares ``provides``
         keys and the response is not valid JSON or is missing keys,
         the score is multiplied by a penalty factor (0.2 - 1.0).
    """

    def estimate(self, result: CompletionResult, task: TaskNode) -> float:
        """Return a confidence score in [0.0, 1.0].

        Args:
            result: The completion result to evaluate.
            task: The task node that produced the request (used for
                structured output validation via ``task.provides``).
        """
        content = result.content.strip()

        # Empty content is always zero confidence.
        if not content:
            return 0.0

        # --- Additive base score from quality signals ---
        length_score = self._score_content_length(content)
        token_score = self._score_token_count(result.tokens_out)
        logprobs_score = self._score_logprobs(result.logprobs)

        has_logprobs = result.logprobs is not None and len(result.logprobs) > 0

        if has_logprobs:
            base_score = (
                0.35 * length_score
                + 0.35 * token_score
                + 0.30 * logprobs_score
            )
        else:
            # No logprobs: split evenly between length and token count.
            base_score = 0.50 * length_score + 0.50 * token_score

        # --- Multiplicative penalties ---
        refusal_factor = self._refusal_factor(content)
        structured_factor = self._structured_output_factor(content, task)

        score = base_score * refusal_factor * structured_factor

        return max(0.0, min(1.0, score))

    def compute_token_budget(self, complexity: int, config: CascadeConfig) -> int:
        """Compute the max_tokens directive for a given task complexity.

        Formula: ``int(base_tokens * complexity_multipliers[clamped_complexity])``

        Args:
            complexity: Raw complexity level (will be clamped to 1-5).
            config: Cascade configuration containing base_tokens and
                complexity_multipliers.

        Returns:
            The computed token budget as an integer.
        """
        clamped = max(1, min(5, complexity))
        multipliers = config.complexity_multipliers or {
            1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0,
        }
        multiplier = multipliers.get(clamped, 1.0)
        return int(config.base_tokens * multiplier)

    # ------------------------------------------------------------------
    # Additive scoring sub-dimensions
    # ------------------------------------------------------------------

    @staticmethod
    def _score_content_length(content: str) -> float:
        """Score based on response content length.

        Returns a value in [0.0, 1.0].
        """
        content_len = len(content)
        if content_len >= _MEDIUM_CONTENT_LENGTH:
            return 1.0
        if content_len >= _SHORT_CONTENT_LENGTH:
            return 0.6
        if content_len >= _MIN_CONTENT_LENGTH:
            return 0.3
        return 0.1

    @staticmethod
    def _score_token_count(tokens_out: int) -> float:
        """Score based on output token count.

        Returns a value in [0.0, 1.0].
        """
        if tokens_out >= 20:
            return 1.0
        if tokens_out >= _MIN_TOKEN_COUNT:
            return 0.6
        if tokens_out >= 2:
            return 0.2
        return 0.0

    @staticmethod
    def _score_logprobs(logprobs: list[float] | None) -> float:
        """Score based on mean token log-probability.

        Maps the mean logprob to [0.0, 1.0] using the exponential
        (converting log-prob to probability).  A mean logprob of 0.0
        corresponds to 100% probability (score 1.0), while very
        negative values approach 0.0.

        Returns 0.5 (neutral) when logprobs are unavailable.
        """
        if not logprobs:
            return 0.5

        mean_logprob = sum(logprobs) / len(logprobs)
        # exp(logprob) gives the probability; clamp to [0, 1]
        probability = math.exp(mean_logprob)
        return max(0.0, min(1.0, probability))

    # ------------------------------------------------------------------
    # Multiplicative penalty factors
    # ------------------------------------------------------------------

    @staticmethod
    def _refusal_factor(content: str) -> float:
        """Return 1.0 if no refusal, or _REFUSAL_PENALTY if refusal detected."""
        lower = content.lower()
        for phrase in _REFUSAL_PHRASES:
            if phrase in lower:
                return _REFUSAL_PENALTY
        return 1.0

    @staticmethod
    def _structured_output_factor(content: str, task: TaskNode) -> float:
        """Return a multiplicative factor for structured output validity.

        If the task declares ``provides`` keys, checks that:
        1. The content is valid JSON.
        2. The JSON object contains the expected keys.

        When no ``provides`` keys are declared, returns 1.0 (no penalty).
        """
        provides = task.provides
        if not provides:
            return 1.0

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # Expected structured output but got plain text
            return 0.3

        if not isinstance(parsed, dict):
            return 0.4

        present = sum(1 for key in provides if key in parsed)
        if present == len(provides):
            return 1.0
        if present > 0:
            return 0.4 + 0.6 * (present / len(provides))
        return 0.4
