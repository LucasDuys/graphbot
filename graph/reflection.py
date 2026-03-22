"""Failure reflection engine: generates structured reflections from failed task executions.

On task failure, the engine calls a small LLM to produce a structured analysis of
what went wrong, why, and what to try differently. The reflection is stored as a
Memory node in the knowledge graph for future retrieval during decomposition.
"""

from __future__ import annotations

import json
import logging

from core_gb.types import CompletionResult, ExecutionResult, Reflection
from models.base import ModelProvider

logger = logging.getLogger(__name__)

# Model used for reflection -- should be cheap and fast.
_REFLECTION_MODEL = "meta-llama/llama-3.1-8b-instruct"

_SYSTEM_PROMPT = (
    "You are a failure analysis engine. Given a failed task and its errors, "
    "produce a structured JSON reflection with exactly three fields:\n"
    '  "what_failed": a concise description of what went wrong\n'
    '  "why": your best analysis of the root cause\n'
    '  "what_to_try": a concrete suggestion for what to do differently next time\n\n'
    "Respond with ONLY valid JSON, no markdown, no explanation."
)


def _build_user_message(task_description: str, result: ExecutionResult) -> str:
    """Build the user message for the reflection prompt."""
    errors_text = "\n".join(f"- {e}" for e in result.errors) if result.errors else "No error details"
    return (
        f"Task: {task_description}\n"
        f"Status: FAILED\n"
        f"Errors:\n{errors_text}\n"
        f"Tokens used: {result.total_tokens}\n"
        f"Latency: {result.total_latency_ms:.0f}ms\n"
        f"Nodes executed: {result.total_nodes}"
    )


def _parse_reflection(content: str) -> Reflection | None:
    """Parse LLM output into a Reflection, returning None on failure."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Reflection LLM output is not valid JSON: %.100s", content)
        return None

    what_failed = data.get("what_failed")
    why = data.get("why")
    what_to_try = data.get("what_to_try")

    if not all(isinstance(v, str) and v for v in (what_failed, why, what_to_try)):
        logger.warning(
            "Reflection JSON missing required fields: keys=%s",
            list(data.keys()),
        )
        return None

    return Reflection(
        what_failed=what_failed,
        why=why,
        what_to_try=what_to_try,
    )


class ReflectionEngine:
    """Generates structured reflections from failed task executions via LLM.

    Args:
        provider: The LLM provider to use for generating reflections.
        model: Override the default reflection model.
    """

    def __init__(
        self,
        provider: ModelProvider,
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model or _REFLECTION_MODEL

    async def reflect(
        self,
        task_description: str,
        result: ExecutionResult,
    ) -> Reflection | None:
        """Generate a structured reflection for a failed task.

        Returns a Reflection on success, None if the task succeeded or if the
        LLM call fails or produces unparseable output. This method never raises;
        all errors are logged and return None.
        """
        if result.success:
            return None

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(task_description, result)},
        ]

        try:
            completion: CompletionResult = await self._provider.complete(
                messages, self._model
            )
        except Exception:
            logger.exception("Reflection LLM call failed for task %s", result.root_id)
            return None

        return _parse_reflection(completion.content)
