"""Graph update loop: records task outcomes and extracted patterns in the knowledge graph."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from core_gb.patterns import PatternExtractor, PatternMatcher, PatternStore
from core_gb.types import ExecutionResult, Reflection, TaskNode
from graph.store import GraphStore

logger = logging.getLogger(__name__)


class GraphUpdater:
    """Records task outcomes and extracted patterns in the knowledge graph.

    Optionally accepts a ReflectionEngine for generating structured failure
    reflections. When provided, failed tasks trigger an async LLM call to
    produce a reflection that is stored as a Memory node with a REFLECTION_OF
    edge to the failed Task node.
    """

    def __init__(
        self,
        store: GraphStore,
        reflection_engine: object | None = None,
    ) -> None:
        self._store = store
        self._extractor = PatternExtractor()
        self._matcher = PatternMatcher()
        self._pattern_store = PatternStore(store)
        # Typed as object in signature to avoid circular import; actual type
        # is graph.reflection.ReflectionEngine.
        self._reflection_engine = reflection_engine

    def _record_core(
        self, task: str, nodes: list[TaskNode], result: ExecutionResult
    ) -> str | None:
        """Record execution outcome in the graph (sync, shared logic).

        1. Create Task node for the root task
        2. Create ExecutionTree node
        3. Link ExecutionTree -> Task via DERIVED_FROM edge
        4. Increment success or failure counter on any matching pattern
        5. Extract pattern if applicable, store it

        Returns pattern ID if a new pattern was extracted, None otherwise.
        """
        now = datetime.now()

        # 1. Record root task
        task_id = result.root_id
        self._store.create_node("Task", {
            "id": task_id,
            "description": task[:500],
            "domain": "synthesis",
            "complexity": 1,
            "status": "completed" if result.success else "failed",
            "tokens_used": result.total_tokens,
            "latency_ms": result.total_latency_ms,
            "created_at": now,
            "completed_at": now if result.success else None,
        })

        # 2. Record execution tree
        tree_id = f"tree_{task_id}"
        self._store.create_node("ExecutionTree", {
            "id": tree_id,
            "root_task_id": task_id,
            "total_nodes": result.total_nodes,
            "total_tokens": result.total_tokens,
            "total_latency_ms": result.total_latency_ms,
            "created_at": now,
        })

        # 3. Link tree -> task
        self._store.create_edge("DERIVED_FROM", tree_id, task_id)

        # 4. Increment success or failure counter on any matching pattern
        existing_patterns = self._pattern_store.load_all()
        if existing_patterns:
            match_result = self._matcher.match(task, existing_patterns)
            if match_result is not None:
                matched_pattern, _ = match_result
                if result.success:
                    self._pattern_store.increment_usage(matched_pattern.id)
                    logger.info(
                        "Incremented success on pattern %s", matched_pattern.id
                    )
                else:
                    self._pattern_store.increment_failure(matched_pattern.id)
                    logger.info(
                        "Incremented failure on pattern %s", matched_pattern.id
                    )

        # 5. Extract and store pattern
        pattern = self._extractor.extract(task, nodes, result)
        if pattern is not None:
            self._pattern_store.save(pattern)
            logger.info("Extracted pattern: %s", pattern.trigger[:60])
            return pattern.id

        return None

    def _store_reflection(
        self, task_id: str, reflection: Reflection
    ) -> str:
        """Store a Reflection as a Memory node and link it to the failed Task.

        Returns the ID of the created Memory node.
        """
        memory_id = f"refl_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        content = json.dumps({
            "what_failed": reflection.what_failed,
            "why": reflection.why,
            "what_to_try": reflection.what_to_try,
        })

        self._store.create_node("Memory", {
            "id": memory_id,
            "content": content,
            "category": "reflection",
            "confidence": 1.0,
            "source_episode": task_id,
            "valid_from": now,
        })

        self._store.create_edge("REFLECTION_OF", memory_id, task_id)
        logger.info(
            "Stored failure reflection %s for task %s", memory_id, task_id
        )
        return memory_id

    def update(
        self, task: str, nodes: list[TaskNode], result: ExecutionResult
    ) -> str | None:
        """Record execution outcome in the graph (synchronous, no reflection).

        Backward-compatible entry point. Does not invoke the reflection engine.
        Returns pattern ID if a new pattern was extracted, None otherwise.
        """
        return self._record_core(task, nodes, result)

    async def update_async(
        self, task: str, nodes: list[TaskNode], result: ExecutionResult
    ) -> str | None:
        """Record execution outcome in the graph, with async reflection on failure.

        1. Record core graph state (Task, ExecutionTree, edges, patterns)
        2. If result.success is False and a reflection engine is available,
           generate a structured reflection and store it as a Memory node

        Returns pattern ID if a new pattern was extracted, None otherwise.
        """
        pattern_id = self._record_core(task, nodes, result)

        # Post-execution hook: generate reflection on failure
        if not result.success and self._reflection_engine is not None:
            try:
                from graph.reflection import ReflectionEngine

                engine: ReflectionEngine = self._reflection_engine  # type: ignore[assignment]
                reflection = await engine.reflect(
                    task_description=task,
                    result=result,
                )
                if reflection is not None:
                    self._store_reflection(result.root_id, reflection)
            except Exception:
                logger.exception(
                    "Reflection generation failed for task %s; update continues",
                    result.root_id,
                )

        return pattern_id
