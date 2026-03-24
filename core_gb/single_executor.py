"""SingleCallExecutor -- one enriched LLM call with graph context.

Assembles a single rich prompt from graph context, conversation history,
and pattern hints, then makes exactly one LLM call via ModelRouter.route().
No decomposition, no multi-step, no tools.
"""

from __future__ import annotations

import logging
import time
import uuid

from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    GraphContext,
    Pattern,
    TaskNode,
    TaskStatus,
)
from models.errors import AllProvidersExhaustedError, ProviderError
from models.router import ModelRouter

logger = logging.getLogger(__name__)

# Default number of recent conversation messages to include.
DEFAULT_HISTORY_LIMIT: int = 20


class SingleCallExecutor:
    """Execute a task with one enriched LLM call using full graph context.

    Takes a task description, graph context, conversation history, and pattern
    hints. Assembles a single prompt with structured sections and makes exactly
    one call to the model router. Returns an ExecutionResult compatible with
    the existing pipeline.

    No decomposition, no multi-step reasoning, no tool use -- just one
    enriched call.
    """

    def __init__(
        self,
        router: ModelRouter,
        *,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self._router = router
        self._history_limit = history_limit

    async def execute(
        self,
        task: str,
        graph_context: GraphContext,
        conversation_history: list[dict[str, str]] | None = None,
        pattern_hints: list[Pattern] | None = None,
        complexity: int = 1,
    ) -> ExecutionResult:
        """Execute a single enriched LLM call.

        Args:
            task: The task description / user message.
            graph_context: Assembled graph context with entities, memories,
                and reflections.
            conversation_history: Optional list of prior conversation messages
                (dicts with "role" and "content" keys).
            pattern_hints: Optional list of Pattern objects that matched the
                task. Formatted as hints in the prompt.
            complexity: Task complexity for model routing (1-5).

        Returns:
            An ExecutionResult with the LLM response and metrics.
        """
        start = time.perf_counter()
        root_id = str(uuid.uuid4())

        messages = self._build_messages(
            task=task,
            graph_context=graph_context,
            conversation_history=conversation_history,
            pattern_hints=pattern_hints,
        )

        task_node = TaskNode(
            id=root_id,
            description=task,
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=complexity,
            status=TaskStatus.READY,
        )

        try:
            completion: CompletionResult = await self._router.route(
                task_node, messages,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            return ExecutionResult(
                root_id=root_id,
                output=completion.content,
                success=True,
                total_nodes=1,
                total_tokens=completion.tokens_in + completion.tokens_out,
                total_latency_ms=elapsed_ms,
                total_cost=completion.cost,
                context_tokens=graph_context.total_tokens,
                model_used=completion.model,
                tools_used=0,
                llm_calls=1,
                nodes=(root_id,),
                errors=(),
            )
        except (ProviderError, AllProvidersExhaustedError) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if isinstance(exc, AllProvidersExhaustedError):
                error_strs = tuple(str(e) for e in exc.errors)
            else:
                error_strs = (str(exc),)

            return ExecutionResult(
                root_id=root_id,
                output="",
                success=False,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=elapsed_ms,
                total_cost=0.0,
                context_tokens=graph_context.total_tokens,
                model_used="",
                nodes=(root_id,),
                errors=error_strs,
            )

    def _build_messages(
        self,
        *,
        task: str,
        graph_context: GraphContext,
        conversation_history: list[dict[str, str]] | None = None,
        pattern_hints: list[Pattern] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the structured prompt sections into a message list.

        Message structure:
        1. System message with context, pattern hints
        2. Conversation history (last N messages)
        3. User message with the actual task
        """
        system_parts: list[str] = ["You are a helpful assistant."]

        # Context section: graph entities, relationships, memories
        context_str = graph_context.format()
        if context_str:
            system_parts.append(
                f"\n<context>\n{context_str}\n</context>"
            )

        # Pattern hints section
        if pattern_hints:
            hints = self._format_pattern_hints(pattern_hints)
            if hints:
                system_parts.append(f"\n{hints}")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "".join(system_parts)},
        ]

        # Conversation section: last N messages
        if conversation_history:
            recent = conversation_history[-self._history_limit :]
            messages.extend(recent)

        # User message: the actual task
        messages.append({"role": "user", "content": task})

        return messages

    @staticmethod
    def _format_pattern_hints(patterns: list[Pattern]) -> str:
        """Format pattern hints into a prompt section."""
        if not patterns:
            return ""

        lines: list[str] = ["Similar tasks have been answered like this:"]
        for pattern in patterns:
            detail = f'- "{pattern.trigger}": {pattern.description}'
            if pattern.success_count > 0:
                detail += f" ({pattern.success_count} successes)"
            lines.append(detail)

        return "\n".join(lines)
