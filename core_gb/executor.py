"""Minimal single-task executor: graph context + LLM call."""

from __future__ import annotations

import json
import time
import uuid

from core_gb.types import CompletionResult, ExecutionResult, TaskNode, TaskStatus, Domain
from graph.resolver import EntityResolver
from graph.store import GraphStore
from models.errors import ProviderError
from models.router import ModelRouter


class SimpleExecutor:
    """Minimal single-task executor: graph context + LLM call.

    No decomposition -- single node, single LLM call.
    This is the Phase 1 proof of concept; the full DAG executor comes in Phase 3.
    """

    def __init__(self, store: GraphStore, router: ModelRouter, tool_registry: object | None = None) -> None:
        self._store = store
        self._router = router
        self._resolver = EntityResolver(store)
        self._tool_registry = tool_registry

    async def execute(
        self, task: str, complexity: int = 1, provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute a single task with graph context.

        Flow:
        1. Extract entity mentions from task text (simple word extraction)
        2. Resolve entities via EntityResolver
        3. Assemble context from graph via GraphStore.get_context
        4. Build messages with context injected at beginning (ADR-009)
        5. Create TaskNode, route to model via ModelRouter
        6. Return ExecutionResult with all metrics
        """
        start = time.perf_counter()
        root_id = str(uuid.uuid4())

        # Check if task should use a tool directly (skip if task contains
        # forwarded data from DAG execution -- DAGExecutor handles tool routing)
        if self._tool_registry and "<forwarded_data>" not in task:
            from core_gb.decomposer import infer_domain_from_description
            inferred_domain = infer_domain_from_description(task)
            if inferred_domain and self._tool_registry.has_tool(inferred_domain):
                from core_gb.types import TaskNode as TN, TaskStatus as TS
                tool_node = TN(
                    id=root_id, description=task,
                    is_atomic=True, domain=inferred_domain, status=TS.READY,
                )
                result = await self._tool_registry.execute(tool_node)
                elapsed = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    root_id=root_id,
                    output=result.output,
                    success=result.success,
                    total_nodes=1,
                    total_tokens=0,
                    total_latency_ms=elapsed,
                    total_cost=0.0,
                    model_used=result.model_used,
                    errors=result.errors,
                )

        # Step 1: Extract entity mentions (simple word extraction)
        words = self._extract_mentions(task)

        # Step 2: Resolve entities
        entity_ids: list[str] = []
        for word in words:
            matches = self._resolver.resolve(word, top_k=1)
            for eid, confidence in matches:
                if confidence > 0.5 and eid not in entity_ids:
                    entity_ids.append(eid)

        # Step 3: Assemble context from graph
        context = self._store.get_context(entity_ids)
        context_str = context.format()

        # Step 4: Build messages (ADR-009: context at beginning)
        json_instruction = ""
        if provides_keys:
            keys_desc = ", ".join(f'"{k}"' for k in provides_keys)
            json_instruction = (
                f"\n\nIMPORTANT: Return your answer as a JSON object with these keys: "
                f"{keys_desc}. Each value should be a string with your answer for that part."
            )

        if context_str:
            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": f"<context>\n{context_str}\n</context>\n\nYou are a helpful assistant.{json_instruction}",
                },
                {"role": "user", "content": task},
            ]
        else:
            messages = [
                {"role": "system", "content": f"You are a helpful assistant.{json_instruction}"},
                {"role": "user", "content": task},
            ]

        # Step 5: Create TaskNode and route to model
        task_node = TaskNode(
            id=root_id,
            description=task,
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=complexity,
            status=TaskStatus.READY,
        )

        try:
            route_kwargs: dict[str, object] = {}
            if provides_keys:
                route_kwargs["response_format"] = {"type": "json_object"}

            completion: CompletionResult = await self._router.route(
                task_node, messages, **route_kwargs
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Parse structured output when provides_keys is set
            output = completion.content
            if provides_keys:
                try:
                    parsed = json.loads(completion.content)
                    output = json.dumps(parsed)
                except json.JSONDecodeError:
                    output = json.dumps({provides_keys[0]: completion.content})

            return ExecutionResult(
                root_id=root_id,
                output=output,
                success=True,
                total_nodes=1,
                total_tokens=completion.tokens_in + completion.tokens_out,
                total_latency_ms=elapsed_ms,
                total_cost=completion.cost,
                context_tokens=context.total_tokens,
                model_used=completion.model,
                nodes=(root_id,),
                errors=(),
            )
        except ProviderError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000

            return ExecutionResult(
                root_id=root_id,
                output="",
                success=False,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=elapsed_ms,
                total_cost=0.0,
                context_tokens=context.total_tokens,
                model_used="",
                nodes=(root_id,),
                errors=(str(exc),),
            )

    @staticmethod
    def _extract_mentions(text: str) -> list[str]:
        """Extract candidate entity mentions from task text.

        Simple word extraction: splits on whitespace and strips punctuation.
        Returns unique words with length >= 3 (to avoid noise from short words).
        """
        seen: set[str] = set()
        mentions: list[str] = []
        for word in text.split():
            cleaned = word.strip(".,!?;:\"'()[]{}").strip()
            if len(cleaned) >= 3 and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                mentions.append(cleaned)
        return mentions
