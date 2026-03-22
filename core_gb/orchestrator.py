"""Orchestrator -- wires intake, decomposer, and executor into a single flow."""

from __future__ import annotations

import json
import logging
import uuid

from pathlib import Path

from core_gb.dag_executor import DAGExecutor
from core_gb.decomposer import Decomposer
from core_gb.executor import SimpleExecutor
from core_gb.intake import IntakeParser, TaskType
from core_gb.patterns import PatternMatcher, PatternStore
from core_gb.types import Domain, ExecutionResult, Pattern, TaskNode, TaskStatus
from graph.resolver import EntityResolver
from graph.store import GraphStore
from graph.updater import GraphUpdater
from models.router import ModelRouter
from tools_gb.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main entry point: routes messages through intake -> decompose -> execute.

    Complex tasks are decomposed into a DAG of TaskNodes and executed in
    parallel via DAGExecutor (streaming topological dispatch).
    Pattern matching provides a cache layer before decomposition.
    """

    def __init__(self, store: GraphStore, router: ModelRouter) -> None:
        self._store = store
        self._router = router
        self._intake = IntakeParser()
        self._tool_registry = ToolRegistry(workspace=str(Path.cwd()), router=router)
        self._executor = SimpleExecutor(store, router, tool_registry=self._tool_registry)
        self._dag_executor = DAGExecutor(self._executor, tool_registry=self._tool_registry)
        self._decomposer = Decomposer(router)
        self._resolver = EntityResolver(store)
        self._pattern_store = PatternStore(store)
        self._pattern_matcher = PatternMatcher()
        self._graph_updater = GraphUpdater(store)

    async def process(self, message: str) -> ExecutionResult:
        """Process a user message end-to-end.

        Flow:
        1. IntakeParser.parse(message) -> IntakeResult
        2. Check pattern cache for a match (skip decomposition if hit)
        3. If simple (is_simple=True): route directly to SimpleExecutor
        4. If complex:
           a. Resolve entities and assemble graph context
           b. Decompose via Decomposer
           c. Execute DAG in parallel via DAGExecutor
           d. Aggregate results into single ExecutionResult
        5. Update knowledge graph with execution outcome
        """
        intake = self._intake.parse(message)

        # Check pattern cache before decomposing
        patterns = self._pattern_store.load_all()
        if patterns:
            match_result = self._pattern_matcher.match(message, patterns)
            if match_result is not None:
                pattern, bindings = match_result
                nodes = self._instantiate_pattern(pattern, bindings)
                if nodes:
                    self._pattern_store.increment_usage(pattern.id)
                    logger.info("Pattern cache hit: %s", pattern.trigger[:60])
                    leaves = [n for n in nodes if n.is_atomic]
                    if len(leaves) > 1:
                        result = await self._dag_executor.execute(nodes)
                    else:
                        result = await self._execute_single_leaf(
                            nodes[0], message
                        )
                    self._graph_updater.update(message, nodes, result)
                    return result

        if intake.is_simple:
            node = TaskNode(
                id=str(uuid.uuid4()),
                description=message,
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=intake.complexity,
                status=TaskStatus.READY,
            )
            result = await self._executor.execute(message, intake.complexity)
            self._graph_updater.update(message, [node], result)
            return result

        # INTEGRATED tasks: single LLM call with full context, skip decomposition
        if intake.task_type == TaskType.INTEGRATED:
            result = await self._executor.execute(
                message, max(intake.complexity, 3)
            )
            node = TaskNode(
                id=str(uuid.uuid4()),
                description=message,
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=max(intake.complexity, 3),
                status=TaskStatus.READY,
            )
            self._graph_updater.update(message, [node], result)
            return result

        # Complex path: resolve entities, get context, decompose, execute
        # Applies to DATA_PARALLEL, SEQUENTIAL, and non-simple ATOMIC tasks
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
            result = await self._executor.execute(message, intake.complexity)
            self._graph_updater.update(message, nodes, result)
            return result

        if len(leaves) == 1:
            # Single leaf: execute directly
            result = await self._execute_single_leaf(leaves[0], message)
            self._graph_updater.update(message, nodes, result)
            return result

        self._dag_executor.aggregation_template = self._decomposer.last_template
        result = await self._dag_executor.execute(nodes)
        self._graph_updater.update(message, nodes, result)
        return result

    async def _execute_single_leaf(
        self, leaf: TaskNode, original_message: str
    ) -> ExecutionResult:
        """Execute a single leaf node via SimpleExecutor."""
        task_text = leaf.description or original_message
        return await self._executor.execute(task_text, leaf.complexity)

    def _instantiate_pattern(
        self, pattern: Pattern, bindings: dict[str, str]
    ) -> list[TaskNode] | None:
        """Instantiate a pattern template with variable bindings."""
        if not pattern.tree_template:
            return None
        try:
            template_nodes = json.loads(pattern.tree_template)
            nodes: list[TaskNode] = []
            for tnode in template_nodes:
                desc = tnode["description"]
                for slot, value in bindings.items():
                    desc = desc.replace("{" + slot + "}", value)

                node = TaskNode(
                    id=str(uuid.uuid4()),
                    description=desc,
                    is_atomic=tnode.get("is_atomic", False),
                    domain=Domain(tnode.get("domain", "synthesis")),
                    complexity=tnode.get("complexity", 1),
                    provides=tnode.get("provides", []),
                    consumes=tnode.get("consumes", []),
                    status=TaskStatus.READY if tnode.get("is_atomic") else TaskStatus.CREATED,
                )
                nodes.append(node)
            return nodes if nodes else None
        except Exception as exc:
            logger.warning("Pattern instantiation failed: %s", exc)
            return None
