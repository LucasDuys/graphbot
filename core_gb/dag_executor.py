"""Parallel DAG executor using streaming topological dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from graphlib import TopologicalSorter

from core_gb.aggregator import Aggregator
from core_gb.types import ExecutionResult, TaskNode

logger = logging.getLogger(__name__)


class DAGExecutor:
    """Parallel DAG executor using streaming topological dispatch.

    When a node completes, immediately unblocks and starts dependents.
    Independent nodes execute concurrently via asyncio, bounded by a semaphore.
    """

    def __init__(
        self,
        executor: object,
        max_concurrency: int = 10,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._executor = executor
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._aggregator = Aggregator()
        self._tool_registry = tool_registry
        self.aggregation_template: dict | None = None

    async def execute(self, nodes: list[TaskNode]) -> ExecutionResult:
        """Execute a DAG of TaskNodes with parallel streaming dispatch.

        Flow:
        1. Filter to atomic (leaf) nodes only.
        2. Build dependency graph from node.requires fields.
        3. Use TopologicalSorter in streaming mode (prepare/get_ready/done).
        4. Launch ready nodes concurrently, bounded by semaphore.
        5. As each completes, call sorter.done() to unblock dependents.
        6. Forward data from completed nodes to their dependents.
        7. Aggregate all results into a single ExecutionResult.
        """
        start = time.perf_counter()
        root_id = str(uuid.uuid4())

        # Filter to leaf nodes only
        leaves = [n for n in nodes if n.is_atomic]

        if not leaves:
            return ExecutionResult(
                root_id=root_id,
                output="",
                success=True,
                total_nodes=0,
                total_tokens=0,
                total_latency_ms=0.0,
                total_cost=0.0,
            )

        node_by_id: dict[str, TaskNode] = {n.id: n for n in nodes}
        leaf_set: set[str] = {n.id for n in leaves}

        # Build dependency graph: only leaf-to-leaf dependencies
        dep_graph: dict[str, set[str]] = {n.id: set() for n in leaves}
        for leaf in leaves:
            for req_id in leaf.requires:
                if req_id in leaf_set:
                    dep_graph[leaf.id].add(req_id)
                elif req_id in node_by_id:
                    self._collect_leaf_deps(req_id, node_by_id, leaf_set, dep_graph[leaf.id])

        # Data registry: maps provides keys to output text
        data_registry: dict[str, str] = {}
        results: dict[str, ExecutionResult] = {}

        sorter: TopologicalSorter[str] = TopologicalSorter(dep_graph)
        sorter.prepare()

        pending: set[asyncio.Task[tuple[str, ExecutionResult]]] = set()

        while sorter.is_active():
            ready = sorter.get_ready()
            for node_id in ready:
                node = node_by_id[node_id]
                task = asyncio.create_task(
                    self._execute_node(node, data_registry.copy())
                )
                pending.add(task)

            if pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    node_id, result = completed_task.result()
                    results[node_id] = result
                    sorter.done(node_id)

                    # Register provided data keys
                    if result.success:
                        node = node_by_id[node_id]
                        output_text = result.output
                        for key in node.provides:
                            data_registry[key] = node.output_data.get(key, output_text)
                        if node.provides:
                            node.output_data[node.provides[0]] = output_text

        return self._aggregate_results(root_id, leaves, results, node_by_id, start)

    async def _execute_node(
        self,
        node: TaskNode,
        data_snapshot: dict[str, str],
    ) -> tuple[str, ExecutionResult]:
        """Execute a single node, injecting forwarded data into the task description."""
        async with self._semaphore:
            task_text = node.description

            # Forward consumed data into the task description
            consumed_data: dict[str, str] = {}
            for key in node.consumes:
                if key in data_snapshot:
                    consumed_data[key] = data_snapshot[key]

            if consumed_data:
                context_parts = [f"[{k}]: {v}" for k, v in consumed_data.items()]
                context_str = "\n".join(context_parts)
                task_text = f"<forwarded_data>\n{context_str}\n</forwarded_data>\n\n{task_text}"

            _provides = list(node.provides) if node.provides else None

            try:
                # Use tool registry for domains with registered tools
                if self._tool_registry and node.is_atomic and self._tool_registry.has_tool(node.domain):
                    result = await self._tool_registry.execute(node)
                else:
                    result = await self._executor.execute(task_text, node.complexity, provides_keys=_provides)
            except Exception as exc:
                logger.error("Node %s failed with exception: %s", node.id, exc)
                result = ExecutionResult(
                    root_id=node.id,
                    output="",
                    success=False,
                    total_nodes=1,
                    total_tokens=0,
                    total_latency_ms=0.0,
                    total_cost=0.0,
                    errors=(str(exc),),
                )

            return node.id, result

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

    def _aggregate_results(
        self,
        root_id: str,
        leaves: list[TaskNode],
        results: dict[str, ExecutionResult],
        node_by_id: dict[str, TaskNode],
        start: float,
    ) -> ExecutionResult:
        """Combine results from all leaf executions into a single ExecutionResult."""
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

        total_tokens = 0
        total_cost = 0.0
        all_nodes: list[str] = []
        all_errors: list[str] = []
        all_success = True

        # Iterate in original leaf order for deterministic output
        for leaf in leaves:
            r = results.get(leaf.id)
            if r is None:
                continue
            total_tokens += r.total_tokens
            total_cost += r.total_cost
            all_nodes.extend(r.nodes)
            all_errors.extend(r.errors)
            if not r.success:
                all_success = False

        # Build leaf_outputs dict for the deterministic aggregator
        leaf_outputs: dict[str, str] = {}
        for leaf in leaves:
            r = results.get(leaf.id)
            if r is None or not r.output:
                continue
            node = node_by_id[leaf.id]
            if node.provides:
                try:
                    parsed = json.loads(r.output)
                    if isinstance(parsed, dict):
                        for key in node.provides:
                            if key in parsed:
                                leaf_outputs[key] = str(parsed[key])
                            else:
                                leaf_outputs[key] = r.output
                    else:
                        for key in node.provides:
                            leaf_outputs[key] = r.output
                except json.JSONDecodeError:
                    for key in node.provides:
                        leaf_outputs[key] = r.output
            else:
                leaf_outputs[f"output_{leaf.id}"] = r.output

        aggregated = self._aggregator.aggregate(
            self.aggregation_template, leaf_outputs
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExecutionResult(
            root_id=root_id,
            output=aggregated,
            success=all_success,
            total_nodes=len(results),
            total_tokens=total_tokens,
            total_latency_ms=elapsed_ms,
            total_cost=total_cost,
            nodes=tuple(all_nodes),
            errors=tuple(all_errors),
        )
