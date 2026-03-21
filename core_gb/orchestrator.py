"""Orchestrator -- wires intake, decomposer, and executor into a single flow."""

from __future__ import annotations

import logging

from core_gb.dag_executor import DAGExecutor
from core_gb.decomposer import Decomposer
from core_gb.executor import SimpleExecutor
from core_gb.intake import IntakeParser
from core_gb.types import ExecutionResult, TaskNode
from graph.resolver import EntityResolver
from graph.store import GraphStore
from models.router import ModelRouter

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main entry point: routes messages through intake -> decompose -> execute.

    Complex tasks are decomposed into a DAG of TaskNodes and executed in
    parallel via DAGExecutor (streaming topological dispatch).
    """

    def __init__(self, store: GraphStore, router: ModelRouter) -> None:
        self._store = store
        self._router = router
        self._intake = IntakeParser()
        self._executor = SimpleExecutor(store, router)
        self._dag_executor = DAGExecutor(self._executor)
        self._decomposer = Decomposer(router)
        self._resolver = EntityResolver(store)

    async def process(self, message: str) -> ExecutionResult:
        """Process a user message end-to-end.

        Flow:
        1. IntakeParser.parse(message) -> IntakeResult
        2. If simple (is_simple=True): route directly to SimpleExecutor
        3. If complex:
           a. Resolve entities and assemble graph context
           b. Decompose via Decomposer
           c. Execute DAG in parallel via DAGExecutor
           d. Aggregate results into single ExecutionResult
        """
        intake = self._intake.parse(message)

        if intake.is_simple:
            return await self._executor.execute(message, intake.complexity)

        # Complex path: resolve entities, get context, decompose, execute
        entity_ids: list[str] = []
        for entity in intake.entities:
            matches = self._resolver.resolve(entity, top_k=1)
            for eid, confidence in matches:
                if confidence > 0.5 and eid not in entity_ids:
                    entity_ids.append(eid)

        context = self._store.get_context(entity_ids) if entity_ids else None

        nodes = await self._decomposer.decompose(message, context)

        # Filter to atomic (leaf) nodes only
        leaves = [n for n in nodes if n.is_atomic]

        if not leaves:
            # Edge case: no leaves found, treat entire message as single task
            return await self._executor.execute(message, intake.complexity)

        if len(leaves) == 1:
            # Single leaf: execute directly
            return await self._execute_single_leaf(leaves[0], message)

        return await self._dag_executor.execute(nodes)

    async def _execute_single_leaf(
        self, leaf: TaskNode, original_message: str
    ) -> ExecutionResult:
        """Execute a single leaf node via SimpleExecutor."""
        task_text = leaf.description or original_message
        return await self._executor.execute(task_text, leaf.complexity)

