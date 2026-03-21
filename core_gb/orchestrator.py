"""Orchestrator -- wires intake, decomposer, and executor into a single flow."""

from __future__ import annotations

import logging
import time
import uuid
from graphlib import TopologicalSorter
from typing import Any

from core_gb.decomposer import Decomposer
from core_gb.executor import SimpleExecutor
from core_gb.intake import IntakeParser
from core_gb.types import ExecutionResult, TaskNode, TaskStatus
from graph.resolver import EntityResolver
from graph.store import GraphStore
from models.router import ModelRouter

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main entry point: routes messages through intake -> decompose -> execute.

    For Phase 2, complex tasks execute leaves sequentially in topological order.
    Phase 3 will add parallel DAG execution.
    """

    def __init__(self, store: GraphStore, router: ModelRouter) -> None:
        self._store = store
        self._router = router
        self._intake = IntakeParser()
        self._executor = SimpleExecutor(store, router)
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
           c. Execute leaves in topological order (sequential for Phase 2)
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

        return await self._execute_dag(nodes, leaves, message)

    async def _execute_single_leaf(
        self, leaf: TaskNode, original_message: str
    ) -> ExecutionResult:
        """Execute a single leaf node via SimpleExecutor."""
        task_text = leaf.description or original_message
        return await self._executor.execute(task_text, leaf.complexity)

    async def _execute_dag(
        self,
        all_nodes: list[TaskNode],
        leaves: list[TaskNode],
        original_message: str,
    ) -> ExecutionResult:
        """Execute leaf nodes in topological order, forwarding data between them."""
        start = time.perf_counter()
        root_id = str(uuid.uuid4())

        # Build lookup by node ID
        node_by_id: dict[str, TaskNode] = {n.id: n for n in all_nodes}
        leaf_set: set[str] = {n.id for n in leaves}

        # Build dependency graph for topological sort (leaves only).
        # A leaf depends on other leaves that it requires (directly or transitively
        # through non-leaf parents).
        dep_graph: dict[str, set[str]] = {n.id: set() for n in leaves}
        for leaf in leaves:
            for req_id in leaf.requires:
                if req_id in leaf_set:
                    dep_graph[leaf.id].add(req_id)
                elif req_id in node_by_id:
                    # Requirement is a non-leaf: find leaf descendants it contains
                    self._collect_leaf_deps(req_id, node_by_id, leaf_set, dep_graph[leaf.id])

        sorter: TopologicalSorter[str] = TopologicalSorter(dep_graph)
        try:
            execution_order = list(sorter.static_order())
        except Exception:
            # Fallback: execute in original list order
            logger.warning("Topological sort failed, falling back to list order")
            execution_order = [n.id for n in leaves]

        # Execute leaves sequentially in topological order
        results: list[ExecutionResult] = []
        completed_outputs: dict[str, ExecutionResult] = {}
        # Track which data keys map to which output text
        data_registry: dict[str, str] = {}

        for node_id in execution_order:
            node = node_by_id[node_id]

            # Forward data from completed dependencies
            self._forward_data(node, data_registry)

            # Build task text, injecting forwarded input_data as context
            task_text = node.description
            result = await self._execute_leaf_with_context(node, task_text)
            results.append(result)
            completed_outputs[node_id] = result

            # Register provided data keys
            if result.success:
                output_text = result.output
                for key in node.provides:
                    data_registry[key] = node.output_data.get(key, output_text)
                # Also store under first provides key
                if node.provides:
                    node.output_data[node.provides[0]] = output_text

        return self._aggregate_results(root_id, results, start)

    def _collect_leaf_deps(
        self,
        node_id: str,
        node_by_id: dict[str, TaskNode],
        leaf_set: set[str],
        out: set[str],
    ) -> None:
        """Recursively find leaf nodes within a non-leaf subtree."""
        node = node_by_id.get(node_id)
        if node is None:
            return
        if node_id in leaf_set:
            out.add(node_id)
            return
        for child_id in node.children:
            self._collect_leaf_deps(child_id, node_by_id, leaf_set, out)

    def _forward_data(self, node: TaskNode, data_registry: dict[str, str]) -> None:
        """Set input_data on node from completed providers via data_registry."""
        for key in node.consumes:
            if key in data_registry:
                node.input_data[key] = data_registry[key]

    async def _execute_leaf_with_context(
        self, node: TaskNode, task_text: str
    ) -> ExecutionResult:
        """Execute a leaf node, injecting any forwarded input_data as context."""
        if node.input_data:
            # Build a context string from forwarded data
            context_parts = []
            for key, value in node.input_data.items():
                context_parts.append(f"[{key}]: {value}")
            context_str = "\n".join(context_parts)

            # Prepend context to the task description for the executor
            augmented_task = (
                f"<forwarded_data>\n{context_str}\n</forwarded_data>\n\n{task_text}"
            )
            return await self._executor.execute(augmented_task, node.complexity)

        return await self._executor.execute(task_text, node.complexity)

    def _aggregate_results(
        self,
        root_id: str,
        results: list[ExecutionResult],
        start: float,
    ) -> ExecutionResult:
        """Combine results from multiple leaf executions into a single result."""
        if not results:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                root_id=root_id,
                output="",
                success=False,
                total_nodes=0,
                total_tokens=0,
                total_latency_ms=elapsed_ms,
                total_cost=0.0,
                errors=("No leaf nodes executed",),
            )

        outputs: list[str] = []
        total_tokens = 0
        total_cost = 0.0
        all_nodes: list[str] = []
        all_errors: list[str] = []
        all_success = True
        max_latency = 0.0

        for r in results:
            if r.output:
                outputs.append(r.output)
            total_tokens += r.total_tokens
            total_cost += r.total_cost
            if r.total_latency_ms > max_latency:
                max_latency = r.total_latency_ms
            all_nodes.extend(r.nodes)
            all_errors.extend(r.errors)
            if not r.success:
                all_success = False

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExecutionResult(
            root_id=root_id,
            output="\n".join(outputs),
            success=all_success,
            total_nodes=len(results),
            total_tokens=total_tokens,
            total_latency_ms=elapsed_ms,
            total_cost=total_cost,
            model_used=results[-1].model_used if results else "",
            nodes=tuple(all_nodes),
            errors=tuple(all_errors),
        )
