"""Graph update loop: records task outcomes and extracted patterns in the knowledge graph."""

from __future__ import annotations

import logging
from datetime import datetime

from core_gb.patterns import PatternExtractor, PatternStore
from core_gb.types import ExecutionResult, TaskNode
from graph.store import GraphStore

logger = logging.getLogger(__name__)


class GraphUpdater:
    """Records task outcomes and extracted patterns in the knowledge graph."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store
        self._extractor = PatternExtractor()
        self._pattern_store = PatternStore(store)

    def update(
        self, task: str, nodes: list[TaskNode], result: ExecutionResult
    ) -> str | None:
        """Record execution outcome in the graph.

        1. Create Task node for the root task
        2. Create ExecutionTree node
        3. Link ExecutionTree -> Task via DERIVED_FROM edge
        4. Extract pattern if applicable, store it

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

        # 4. Extract and store pattern
        pattern = self._extractor.extract(task, nodes, result)
        if pattern is not None:
            self._pattern_store.save(pattern)
            logger.info("Extracted pattern: %s", pattern.trigger[:60])
            return pattern.id

        return None
