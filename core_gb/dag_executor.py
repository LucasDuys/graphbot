"""Parallel DAG executor using streaming topological dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from core_gb.decomposer import Decomposer
    from models.router import ModelRouter
    from tools_gb.registry import ToolRegistry

from core_gb.wave_event import WaveCompleteEvent

import re

from core_gb.aggregator import Aggregator
from core_gb.autonomy import AutonomyLevel, RiskScorer
from core_gb.sanitizer import OutputSanitizer
from core_gb.types import ConditionalNode, ExecutionResult, LoopNode, TaskNode, TaskStatus
from core_gb.verification import (
    VerificationConfig,
    VerificationLayer1,
    VerificationLayer2,
    VerificationResult,
)

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
        verification_config: VerificationConfig | None = None,
        router: ModelRouter | None = None,
        on_wave_complete: list[Callable[[WaveCompleteEvent], None]] | None = None,
        max_expansion_depth: int = 2,
        risk_scorer: RiskScorer | None = None,
        autonomy_level: AutonomyLevel | None = None,
    ) -> None:
        self._executor = executor
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._on_wave_complete: list[Callable[[WaveCompleteEvent], None]] = (
            on_wave_complete if on_wave_complete is not None else []
        )
        self._aggregator = Aggregator()
        self._sanitizer = OutputSanitizer()
        self._verification_config = verification_config or VerificationConfig()
        self._verifier = VerificationLayer1()
        # Layer 2 requires a ModelRouter for 3-way sampling. If no router
        # is provided, L2 verification is unavailable and will be skipped
        # even when config thresholds are met.
        self._layer2_verifier: VerificationLayer2 | None = (
            VerificationLayer2(
                router=router,
                complexity_threshold=self._verification_config.layer2_threshold,
            )
            if router is not None
            else None
        )
        self._tool_registry = tool_registry
        self._decomposer: Decomposer | None = None
        self._max_expansion_depth: int = max_expansion_depth
        self.aggregation_template: dict | None = None
        self._on_replan: (
            Callable[[WaveCompleteEvent], Awaitable[list[TaskNode] | None]] | None
        ) = None
        # Per-action autonomy enforcement. When both a RiskScorer and an
        # AutonomyLevel are provided, each node is scored before execution
        # and blocked if its risk exceeds the autonomy ceiling.
        self._risk_scorer: RiskScorer | None = risk_scorer
        self._autonomy_level: AutonomyLevel | None = autonomy_level

    def set_replan_callback(
        self,
        callback: Callable[[WaveCompleteEvent], Awaitable[list[TaskNode] | None]],
    ) -> None:
        """Register an async callback for intermediate result re-planning.

        After each wave completes, this callback is invoked with the
        WaveCompleteEvent. If it returns a non-None list of TaskNode
        instances, those replace all remaining unexecuted nodes in the
        DAG. Completed results are preserved.

        Args:
            callback: Async function that receives a WaveCompleteEvent and
                returns either None (no re-planning) or a list of TaskNode
                instances to replace the remaining execution plan.
        """
        self._on_replan = callback

    async def execute(
        self,
        nodes: list[TaskNode],
        _expansion_depth: dict[str, int] | None = None,
        _expansion_count: int = 0,
    ) -> ExecutionResult:
        """Execute a DAG of TaskNodes with parallel streaming dispatch.

        Flow:
        1. Filter to atomic (leaf) nodes only, handling LoopNode and
           ConditionalNode control flow nodes specially.
        2. Build dependency graph from node.requires fields.
        3. Use TopologicalSorter in streaming mode (prepare/get_ready/done).
        4. Launch ready nodes concurrently, bounded by semaphore.
        5. As each completes, call sorter.done() to unblock dependents.
        6. Forward data from completed nodes to their dependents.
        7. Resolve conditional branches: evaluate conditions and skip branches.
        8. Handle LoopNode: when a loop placeholder becomes ready, dispatch
           body nodes via execute_loop(), register output, mark done.
        9. Propagate skipping transitively to dependents of skipped nodes.
        10. On leaf failure: attempt re-decomposition via expand_node() if a
            decomposer is available and max_expansion_depth not exceeded.
        11. Aggregate all results into a single ExecutionResult.

        Args:
            nodes: The list of TaskNode instances forming the DAG.
            _expansion_depth: Internal tracker mapping node IDs to their
                current expansion depth. Passed through recursive calls
                after re-decomposition. Callers should not set this.
            _expansion_count: Internal counter of how many expansions have
                occurred so far. Passed through recursive calls. Callers
                should not set this.
        """
        start = time.perf_counter()
        root_id = str(uuid.uuid4())

        expansion_depth: dict[str, int] = (
            dict(_expansion_depth) if _expansion_depth is not None else {}
        )
        expansion_count: int = _expansion_count

        # Lazy expansion: replace expandable nodes with sub-DAGs from decomposer
        nodes = await self._expand_nodes(nodes)

        # Resolve conditional nodes: rewire branch dependencies and build
        # the conditional registry for deferred evaluation during dispatch
        nodes, conditional_registry = self._prepare_conditionals(nodes)
        skipped_ids: set[str] = set()

        # --- Identify LoopNodes and extract their body nodes ---
        loop_registry: dict[str, LoopNode] = {}
        loop_body_ids: set[str] = set()
        for node in nodes:
            if isinstance(node, LoopNode):
                loop_registry[node.id] = node
                loop_body_ids.update(node.body_nodes)

        node_by_id: dict[str, TaskNode] = {n.id: n for n in nodes}

        # Resolve loop body nodes: map loop_id -> list of body TaskNode objects
        loop_body_nodes: dict[str, list[TaskNode]] = {}
        for loop_id, loop_node in loop_registry.items():
            body: list[TaskNode] = []
            for bn_id in loop_node.body_nodes:
                bn = node_by_id.get(bn_id)
                if bn is not None:
                    body.append(bn)
            loop_body_nodes[loop_id] = body

        # Build the dispatchable set: atomic leaves that are NOT loop body
        # nodes (those are dispatched inside execute_loop), plus loop
        # placeholders that participate in the topological sort.
        leaves = [
            n for n in nodes
            if n.is_atomic and n.id not in loop_body_ids
        ]

        # Add loop nodes as dispatchable entries (they act as single units
        # in the topological sort, dispatched via execute_loop when ready)
        dispatch_nodes: list[TaskNode] = list(leaves)
        for loop_id, loop_node in loop_registry.items():
            dispatch_nodes.append(loop_node)

        if not dispatch_nodes:
            return ExecutionResult(
                root_id=root_id,
                output="",
                success=True,
                total_nodes=0,
                total_tokens=0,
                total_latency_ms=0.0,
                total_cost=0.0,
                expansion_count=expansion_count,
            )

        dispatch_set: set[str] = {n.id for n in dispatch_nodes}

        # Build dependency graph over dispatchable entries
        dep_graph: dict[str, set[str]] = {n.id: set() for n in dispatch_nodes}
        for dnode in dispatch_nodes:
            for req_id in dnode.requires:
                if req_id in dispatch_set:
                    dep_graph[dnode.id].add(req_id)
                elif req_id in node_by_id:
                    self._collect_leaf_deps(
                        req_id, node_by_id, dispatch_set, dep_graph[dnode.id],
                    )

        # Data registry: maps provides keys to output text
        data_registry: dict[str, str] = {}
        results: dict[str, ExecutionResult] = {}

        sorter: TopologicalSorter[str] = TopologicalSorter(dep_graph)
        sorter.prepare()

        wave_index: int = 0
        all_dispatch_ids: set[str] = set(dispatch_set)

        while sorter.is_active():
            ready = sorter.get_ready()
            wave_tasks: set[asyncio.Task[tuple[str, ExecutionResult]]] = set()
            wave_node_ids: list[str] = []

            for node_id in ready:
                # Skip nodes that were marked SKIPPED by conditional routing
                if node_id in skipped_ids:
                    sorter.done(node_id)
                    # Transitively skip dependents of skipped nodes
                    newly_skipped = self._propagate_skips(
                        node_id, dep_graph, skipped_ids, node_by_id,
                    )
                    skipped_ids.update(newly_skipped)
                    continue

                # --- LoopNode dispatch: run via execute_loop ---
                if node_id in loop_registry:
                    loop_node = loop_registry[node_id]
                    body = loop_body_nodes.get(node_id, [])
                    wave_node_ids.append(node_id)
                    task = asyncio.create_task(
                        self._execute_loop_in_dispatch(
                            loop_node, body, data_registry.copy(),
                        )
                    )
                    wave_tasks.add(task)
                    continue

                wave_node_ids.append(node_id)
                node = node_by_id[node_id]
                task = asyncio.create_task(
                    self._execute_node(node, data_registry.copy())
                )
                wave_tasks.add(task)

            if not wave_tasks:
                continue

            # Wait for all tasks in this wave to complete
            done_tasks, _ = await asyncio.wait(
                wave_tasks, return_when=asyncio.ALL_COMPLETED
            )

            wave_completed_ids: list[str] = []
            expandable_failures: list[tuple[str, ExecutionResult]] = []

            for completed_task in done_tasks:
                node_id, result = completed_task.result()
                results[node_id] = result
                wave_completed_ids.append(node_id)
                sorter.done(node_id)

                # Register provided data keys (sanitize before forwarding)
                if result.success:
                    node = node_by_id[node_id]
                    output_text = self._sanitizer.sanitize(result.output)
                    for key in node.provides:
                        raw_value = node.output_data.get(key, output_text)
                        data_registry[key] = self._sanitizer.sanitize(raw_value)
                    if node.provides:
                        node.output_data[node.provides[0]] = output_text
                else:
                    # Track failed node for potential re-decomposition
                    expandable_failures.append((node_id, result))

                # Evaluate any conditional nodes whose predecessor just completed
                newly_skipped = self._evaluate_conditionals(
                    node_id, conditional_registry, data_registry, node_by_id,
                )
                skipped_ids.update(newly_skipped)

                # Transitively propagate skips to dependents of newly skipped nodes
                for skip_id in list(newly_skipped):
                    transitive = self._propagate_skips(
                        skip_id, dep_graph, skipped_ids, node_by_id,
                    )
                    skipped_ids.update(transitive)

            # --- Re-decomposition on failure ---
            # For each failed leaf, attempt to expand it into a sub-DAG
            # via the decomposer. If successful, execute the sub-DAG
            # inline and replace the failed result.
            for failed_id, failed_result in expandable_failures:
                node_depth = expansion_depth.get(failed_id, 0)
                if node_depth >= self._max_expansion_depth:
                    logger.info(
                        "Node %s failed but max expansion depth (%d) "
                        "reached, not expanding further",
                        failed_id,
                        self._max_expansion_depth,
                    )
                    continue

                failure_context = (
                    "; ".join(failed_result.errors) or "Unknown failure"
                )
                failed_node = node_by_id[failed_id]
                sub_dag = await self.expand_node(
                    failed_node, failure_context,
                )
                if not sub_dag:
                    continue

                expansion_count += 1

                # Propagate expansion depth to sub-DAG nodes
                new_depth = node_depth + 1
                for sub_node in sub_dag:
                    expansion_depth[sub_node.id] = new_depth

                # Rewire entry nodes of the sub-DAG to inherit the
                # failed node's upstream dependencies
                sub_ids = {n.id for n in sub_dag}
                sub_leaves_list = [n for n in sub_dag if n.is_atomic]
                for sn in sub_leaves_list:
                    internal_deps = {
                        r for r in sn.requires if r in sub_ids
                    }
                    if not internal_deps:
                        sn.requires = list(failed_node.requires) + [
                            r for r in sn.requires if r not in sub_ids
                        ]

                # Execute sub-DAG inline as a standalone sub-execution
                logger.info(
                    "Executing sub-DAG for failed node %s "
                    "(expansion_count=%d, depth=%d)",
                    failed_id,
                    expansion_count,
                    new_depth,
                )
                sub_result = await self.execute(
                    sub_dag,
                    _expansion_depth=expansion_depth,
                    _expansion_count=expansion_count,
                )
                # Update expansion_count from the sub-execution
                expansion_count = sub_result.expansion_count

                if sub_result.success:
                    # Replace the failed result with the sub-DAG result
                    results[failed_id] = ExecutionResult(
                        root_id=failed_result.root_id,
                        output=sub_result.output,
                        success=True,
                        total_nodes=sub_result.total_nodes,
                        total_tokens=(
                            failed_result.total_tokens
                            + sub_result.total_tokens
                        ),
                        total_latency_ms=(
                            failed_result.total_latency_ms
                            + sub_result.total_latency_ms
                        ),
                        total_cost=(
                            failed_result.total_cost + sub_result.total_cost
                        ),
                        errors=(),
                        expansion_count=expansion_count,
                    )
                    # Register the sub-DAG output data so downstream
                    # nodes can consume it via the provides keys
                    output_text = self._sanitizer.sanitize(
                        sub_result.output,
                    )
                    for key in failed_node.provides:
                        data_registry[key] = output_text
                    if failed_node.provides:
                        failed_node.output_data[
                            failed_node.provides[0]
                        ] = output_text
                else:
                    # Sub-DAG also failed; keep original failure but
                    # record the expansion attempt
                    results[failed_id] = ExecutionResult(
                        root_id=failed_result.root_id,
                        output=failed_result.output,
                        success=False,
                        total_nodes=(
                            failed_result.total_nodes
                            + sub_result.total_nodes
                        ),
                        total_tokens=(
                            failed_result.total_tokens
                            + sub_result.total_tokens
                        ),
                        total_latency_ms=(
                            failed_result.total_latency_ms
                            + sub_result.total_latency_ms
                        ),
                        total_cost=(
                            failed_result.total_cost + sub_result.total_cost
                        ),
                        errors=(
                            failed_result.errors + sub_result.errors
                        ),
                        expansion_count=expansion_count,
                    )

            # Emit wave-complete event
            completed_so_far: set[str] = set(results.keys())
            remaining = sorted(
                nid for nid in all_dispatch_ids
                if nid not in completed_so_far and nid not in skipped_ids
            )
            accumulated: dict[str, str] = {
                nid: r.output for nid, r in results.items()
            }
            event = WaveCompleteEvent(
                wave_index=wave_index,
                completed_nodes=sorted(wave_completed_ids),
                accumulated_results=accumulated,
                remaining_nodes=remaining,
            )
            for callback in self._on_wave_complete:
                try:
                    callback(event)
                except Exception as exc:
                    logger.warning(
                        "on_wave_complete callback raised: %s", exc,
                    )

            # --- Async re-planning callback ---
            # If an on_replan callback is registered and there are remaining
            # nodes, invoke it. When it returns replacement nodes, drain the
            # remaining entries from the sorter, execute the replacement
            # sub-DAG, and merge results.
            replan_nodes: list[TaskNode] | None = None
            if self._on_replan is not None and remaining:
                try:
                    replan_nodes = await self._on_replan(event)
                except Exception as exc:
                    logger.warning(
                        "on_replan callback raised: %s", exc,
                    )

            if replan_nodes is not None and len(replan_nodes) > 0:
                logger.info(
                    "Re-planning after wave %d: replacing %d remaining "
                    "nodes with %d new nodes",
                    wave_index,
                    len(remaining),
                    len(replan_nodes),
                )
                # Drain all remaining nodes from the sorter so it can
                # finish cleanly. Mark them done without executing.
                while sorter.is_active():
                    drain_ready = sorter.get_ready()
                    for drain_id in drain_ready:
                        sorter.done(drain_id)

                # Execute the replacement sub-DAG, forwarding current
                # data_registry context into new nodes.
                replan_result = await self.execute(
                    replan_nodes,
                    _expansion_depth=expansion_depth,
                    _expansion_count=expansion_count,
                )

                # Merge replan results with completed results. Build a
                # combined dispatch list for aggregation that includes
                # both already-completed nodes and the replan output.
                replan_dispatch = [
                    n for n in replan_nodes if n.is_atomic
                ]
                all_completed_nodes = [
                    n for n in dispatch_nodes
                    if n.id in results
                    and n.id not in skipped_ids
                    and n.id not in loop_registry
                ]
                combined_leaves = all_completed_nodes + replan_dispatch

                # Create a synthetic result entry for the replan sub-DAG
                # keyed by a unique ID so the aggregator includes it.
                replan_id = f"_replan_{wave_index}"
                results[replan_id] = replan_result
                replan_node = TaskNode(
                    id=replan_id,
                    description="Re-planned execution",
                    is_atomic=True,
                    status=TaskStatus.COMPLETED,
                )
                node_by_id[replan_id] = replan_node
                combined_leaves.append(replan_node)

                result = self._aggregate_results(
                    root_id,
                    combined_leaves,
                    results,
                    node_by_id,
                    start,
                )
                return result

            wave_index += 1

        # Filter skipped nodes out of the dispatch list for aggregation
        active_leaves = [
            n for n in dispatch_nodes
            if n.id not in skipped_ids and n.id not in loop_registry
        ]
        result = self._aggregate_results(
            root_id, active_leaves, results, node_by_id, start,
        )
        # Attach the cumulative expansion count
        if expansion_count > 0 or result.expansion_count != expansion_count:
            result = ExecutionResult(
                root_id=result.root_id,
                output=result.output,
                success=result.success,
                total_nodes=result.total_nodes,
                total_tokens=result.total_tokens,
                total_latency_ms=result.total_latency_ms,
                total_cost=result.total_cost,
                context_tokens=result.context_tokens,
                model_used=result.model_used,
                tools_used=result.tools_used,
                llm_calls=result.llm_calls,
                nodes=result.nodes,
                errors=result.errors,
                verification_results=result.verification_results,
                expansion_count=expansion_count,
            )
        return result

    def _prepare_conditionals(
        self,
        nodes: list[TaskNode],
    ) -> tuple[list[TaskNode], dict[str, ConditionalNode]]:
        """Extract ConditionalNode instances and rewire branch dependencies.

        For each ConditionalNode found:
        - Record it in a registry keyed by the predecessor node IDs it consumes.
        - Rewire branch nodes (then_branch and else_branch) so they depend on
          the conditional node's prerequisites instead of the conditional node
          itself. This lets the topological sorter place branch nodes after the
          predecessor, while the dispatch loop decides which branch to activate.

        Returns:
            A tuple of (updated nodes list, conditional registry). The registry
            maps predecessor node IDs to the ConditionalNode instances that
            should be evaluated when that predecessor completes.
        """
        conditional_registry: dict[str, ConditionalNode] = {}
        cond_ids: set[str] = set()

        for node in nodes:
            if isinstance(node, ConditionalNode):
                cond_ids.add(node.id)
                # Map each prerequisite to this conditional for later evaluation
                for req_id in node.requires:
                    conditional_registry[req_id] = node

        if not cond_ids:
            return nodes, conditional_registry

        # Rewire branch nodes: replace dependency on the ConditionalNode with
        # dependencies on the conditional's own prerequisites
        for node in nodes:
            if node.id in cond_ids:
                continue
            new_requires: list[str] = []
            for req_id in node.requires:
                if req_id in cond_ids:
                    # Find the ConditionalNode this branch node depends on
                    cond_node = next(
                        n for n in nodes
                        if isinstance(n, ConditionalNode) and n.id == req_id
                    )
                    # Inherit the conditional's prerequisites
                    new_requires.extend(cond_node.requires)
                else:
                    new_requires.append(req_id)
            node.requires = new_requires

        return nodes, conditional_registry

    @staticmethod
    def _evaluate_condition(condition: str, text: str) -> bool:
        """Evaluate a simple condition string against text.

        Supported condition formats:
        - "contains '<substring>'" -- True if substring is found in text
          (case-insensitive).
        - Any other string -- treated as a plain substring check
          (case-insensitive).

        Args:
            condition: The condition expression to evaluate.
            text: The text (predecessor output) to evaluate against.

        Returns:
            True if the condition matches, False otherwise.
        """
        # Parse "contains '<value>'" pattern
        match = re.match(r"contains\s+'([^']*)'", condition, re.IGNORECASE)
        if match:
            substring = match.group(1)
            return substring.lower() in text.lower()

        # Fallback: plain substring check
        return condition.lower() in text.lower()

    def _evaluate_conditionals(
        self,
        completed_node_id: str,
        conditional_registry: dict[str, ConditionalNode],
        data_registry: dict[str, str],
        node_by_id: dict[str, TaskNode],
    ) -> set[str]:
        """Check if a completed node triggers any conditional evaluations.

        When a predecessor node completes, look up whether any ConditionalNode
        was waiting on it. If so, gather the predecessor output from the data
        registry, evaluate the condition, and mark the skipped branch nodes
        with TaskStatus.SKIPPED.

        Args:
            completed_node_id: ID of the node that just completed.
            conditional_registry: Map of predecessor IDs to ConditionalNodes.
            data_registry: Current data registry with provides-key outputs.
            node_by_id: Lookup dict for all nodes.

        Returns:
            Set of node IDs that should be skipped.
        """
        if completed_node_id not in conditional_registry:
            return set()

        cond_node = conditional_registry[completed_node_id]
        skipped: set[str] = set()

        # Gather predecessor output from data registry (consumed keys)
        predecessor_output_parts: list[str] = []
        for key in cond_node.consumes:
            if key in data_registry:
                predecessor_output_parts.append(data_registry[key])

        # Fall back to the completed node's raw output if no consumed keys matched
        if not predecessor_output_parts:
            completed_node = node_by_id.get(completed_node_id)
            if completed_node and completed_node.output_data:
                first_key = next(iter(completed_node.output_data))
                predecessor_output_parts.append(
                    str(completed_node.output_data[first_key])
                )

        predecessor_text = " ".join(predecessor_output_parts)
        condition_result = self._evaluate_condition(cond_node.condition, predecessor_text)

        if condition_result:
            # True: execute then_branch, skip else_branch
            skip_ids = set(cond_node.else_branch)
            logger.info(
                "Conditional %s evaluated TRUE (condition='%s'), "
                "routing to then_branch=%s, skipping else_branch=%s",
                cond_node.id,
                cond_node.condition,
                cond_node.then_branch,
                cond_node.else_branch,
            )
        else:
            # False: execute else_branch, skip then_branch
            skip_ids = set(cond_node.then_branch)
            logger.info(
                "Conditional %s evaluated FALSE (condition='%s'), "
                "routing to else_branch=%s, skipping then_branch=%s",
                cond_node.id,
                cond_node.condition,
                cond_node.else_branch,
                cond_node.then_branch,
            )

        for skip_id in skip_ids:
            target = node_by_id.get(skip_id)
            if target is not None:
                target.status = TaskStatus.SKIPPED
                skipped.add(skip_id)

        return skipped

    async def _execute_loop_in_dispatch(
        self,
        loop_node: LoopNode,
        body_nodes: list[TaskNode],
        data_snapshot: dict[str, str],
    ) -> tuple[str, ExecutionResult]:
        """Execute a LoopNode as a single dispatchable unit within topological dispatch.

        Injects upstream data from the data snapshot into the loop body nodes'
        consumes fields, then delegates to execute_loop() for iteration. Returns
        the loop node ID and the aggregated ExecutionResult so the dispatch loop
        can register provided data and mark the loop done.

        Args:
            loop_node: The LoopNode to execute.
            body_nodes: The TaskNode instances forming the loop body.
            data_snapshot: Current data registry snapshot for context injection.

        Returns:
            Tuple of (loop_node.id, ExecutionResult).
        """
        import copy

        # Inject upstream consumed data into body node descriptions
        consumed_data: dict[str, str] = {}
        for key in loop_node.consumes:
            if key in data_snapshot:
                consumed_data[key] = data_snapshot[key]

        prepared_body: list[TaskNode] = []
        for bn in body_nodes:
            bn_copy = copy.copy(bn)
            if consumed_data:
                context_parts = [f"[{k}]: {v}" for k, v in consumed_data.items()]
                context_str = "\n".join(context_parts)
                bn_copy.description = (
                    f"<forwarded_data>\n{context_str}\n</forwarded_data>\n\n"
                    f"{bn.description}"
                )
            prepared_body.append(bn_copy)

        result = await self.execute_loop(loop_node, prepared_body)
        return loop_node.id, result

    @staticmethod
    def _propagate_skips(
        skipped_node_id: str,
        dep_graph: dict[str, set[str]],
        already_skipped: set[str],
        node_by_id: dict[str, TaskNode],
    ) -> set[str]:
        """Transitively skip nodes whose dependencies are all skipped.

        When a node is skipped (e.g. via conditional routing), any downstream
        node that depends exclusively on skipped nodes should also be skipped.
        This handles nested conditionals where an inner conditional's branches
        depend on a node that was skipped by the outer conditional.

        Args:
            skipped_node_id: The node that was just skipped.
            dep_graph: The dependency graph (node_id -> set of dependency IDs).
            already_skipped: The current set of skipped node IDs.
            node_by_id: Lookup dict for all nodes.

        Returns:
            Set of newly skipped node IDs (not including already_skipped).
        """
        newly_skipped: set[str] = set()
        # Build a combined set of all skipped IDs (existing + the new one)
        all_skipped = already_skipped | {skipped_node_id}

        # Check all nodes in the dep graph for transitive skipping
        changed = True
        while changed:
            changed = False
            for node_id, deps in dep_graph.items():
                if node_id in all_skipped:
                    continue
                if not deps:
                    continue
                # If ALL dependencies of this node are skipped, skip it too
                if deps.issubset(all_skipped):
                    target = node_by_id.get(node_id)
                    if target is not None:
                        target.status = TaskStatus.SKIPPED
                    all_skipped.add(node_id)
                    newly_skipped.add(node_id)
                    changed = True

        return newly_skipped

    async def _expand_nodes(self, nodes: list[TaskNode]) -> list[TaskNode]:
        """Replace expandable nodes with sub-DAGs from the decomposer.

        For each atomic node with expandable=True:
        1. Invoke the decomposer on the node's task description.
        2. Rewire the sub-DAG entry nodes to inherit the parent's upstream
           dependencies (requires).
        3. Rewire downstream nodes that depended on the parent to instead
           depend on the sub-DAG's terminal nodes.
        4. Remove the original expandable node and insert sub-DAG nodes.

        If no decomposer is available, expandable nodes are left as-is and
        executed directly (graceful fallback).
        """
        if self._decomposer is None:
            return nodes

        expandable = [n for n in nodes if n.is_atomic and n.expandable]
        if not expandable:
            return nodes

        # Work with a mutable copy
        result_nodes: list[TaskNode] = list(nodes)

        for exp_node in expandable:
            try:
                sub_dag = await self._decomposer.decompose(exp_node.description)
            except Exception as exc:
                logger.warning(
                    "Decomposition failed for expandable node %s, "
                    "executing directly: %s",
                    exp_node.id,
                    exc,
                )
                continue

            if not sub_dag:
                logger.warning(
                    "Decomposer returned empty sub-DAG for node %s, "
                    "executing directly",
                    exp_node.id,
                )
                continue

            sub_leaves = [n for n in sub_dag if n.is_atomic]
            if not sub_leaves:
                logger.warning(
                    "Sub-DAG for node %s has no atomic leaves, "
                    "executing directly",
                    exp_node.id,
                )
                continue

            sub_ids = {n.id for n in sub_dag}

            # Identify entry nodes: sub-DAG nodes with no requires within
            # the sub-DAG (they need the parent's upstream deps)
            entry_ids: set[str] = set()
            for sn in sub_leaves:
                internal_deps = {r for r in sn.requires if r in sub_ids}
                if not internal_deps:
                    entry_ids.add(sn.id)

            # Identify terminal nodes: sub-DAG leaves that no other sub-DAG
            # node depends on (downstream nodes will depend on these)
            depended_on: set[str] = set()
            for sn in sub_dag:
                depended_on.update(r for r in sn.requires if r in sub_ids)
            terminal_ids: set[str] = {
                sn.id for sn in sub_leaves if sn.id not in depended_on
            }

            # Rewire entry nodes: inherit parent's upstream dependencies
            for sn in sub_dag:
                if sn.id in entry_ids:
                    sn.requires = list(exp_node.requires) + [
                        r for r in sn.requires if r not in sub_ids
                    ]

            # Rewire downstream nodes: replace dependency on parent with
            # dependencies on terminal sub-DAG nodes
            for node in result_nodes:
                if exp_node.id in node.requires:
                    node.requires = [
                        r for r in node.requires if r != exp_node.id
                    ] + list(terminal_ids)

            # Remove expandable node and insert sub-DAG nodes
            result_nodes = [n for n in result_nodes if n.id != exp_node.id]
            result_nodes.extend(sub_dag)

            logger.info(
                "Expanded node %s into %d sub-nodes (entries=%s, terminals=%s)",
                exp_node.id,
                len(sub_dag),
                sorted(entry_ids),
                sorted(terminal_ids),
            )

        return result_nodes

    async def expand_node(
        self,
        node: TaskNode,
        failure_context: str,
    ) -> list[TaskNode] | None:
        """Re-decompose a failed leaf node into a sub-DAG.

        Invokes the decomposer with the original task description and the
        failure context to produce a refined sub-DAG that replaces the
        failed leaf in the mutable execution plan.

        Args:
            node: The failed TaskNode to re-decompose.
            failure_context: A description of the failure (error messages,
                stack traces, etc.) to guide the decomposer.

        Returns:
            A list of TaskNode instances forming the sub-DAG, or None if
            expansion is not possible (no decomposer, empty result, or
            decomposer raised an exception).
        """
        if self._decomposer is None:
            return None

        # Build augmented task description that includes the failure context
        # so the decomposer can produce a refined plan avoiding the original
        # failure mode.
        augmented_task = (
            f"[Re-decompose after failure] "
            f"Failure: {failure_context} | "
            f"Original task: {node.description}"
        )

        try:
            sub_dag = await self._decomposer.decompose(
                augmented_task,
                failure_context=failure_context,
            )
        except Exception as exc:
            logger.warning(
                "Re-decomposition failed for node %s: %s",
                node.id,
                exc,
            )
            return None

        if not sub_dag:
            logger.warning(
                "Re-decomposition returned empty sub-DAG for node %s",
                node.id,
            )
            return None

        sub_leaves = [n for n in sub_dag if n.is_atomic]
        if not sub_leaves:
            logger.warning(
                "Re-decomposition sub-DAG for node %s has no atomic leaves",
                node.id,
            )
            return None

        logger.info(
            "Re-decomposed failed node %s into %d sub-nodes",
            node.id,
            len(sub_dag),
        )
        return sub_dag

    async def _execute_node(
        self,
        node: TaskNode,
        data_snapshot: dict[str, str],
    ) -> tuple[str, ExecutionResult]:
        """Execute a single node, injecting forwarded data into the task description.

        Before execution, checks per-action autonomy policy. If a RiskScorer
        and AutonomyLevel are configured, the node's risk is scored and
        compared against the autonomy ceiling. Nodes exceeding the ceiling
        are blocked with a failed ExecutionResult.
        """
        # Per-action autonomy enforcement: score risk and block if not allowed
        if self._risk_scorer is not None and self._autonomy_level is not None:
            if not self._risk_scorer.is_allowed(node, self._autonomy_level):
                risk = self._risk_scorer.score_node(node)
                logger.info(
                    "Node '%s' blocked by autonomy policy: risk=%s, "
                    "autonomy_level=%s",
                    node.id,
                    risk.value,
                    self._autonomy_level.value,
                )
                return node.id, ExecutionResult(
                    root_id=node.id,
                    output="Action blocked by autonomy policy",
                    success=False,
                    total_nodes=1,
                    total_tokens=0,
                    total_latency_ms=0.0,
                    total_cost=0.0,
                    errors=(
                        f"Node '{node.id}' blocked: risk={risk.value}, "
                        f"autonomy_level={self._autonomy_level.value}",
                    ),
                )

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
                    if not result.success:
                        # Retry once after 1s
                        await asyncio.sleep(1)
                        result = await self._tool_registry.execute(node)
                        if not result.success:
                            # Fallback to LLM
                            logger.warning(
                                "Tool failed twice for node %s, falling back to LLM",
                                node.id,
                            )
                            result = await self._executor.execute(
                                task_text, node.complexity, provides_keys=_provides,
                            )
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

            # --- Verification pipeline (config-driven) ---
            if result.success:
                result = await self._apply_verification(node, result, task_text)

            return node.id, result

    async def _apply_verification(
        self,
        node: TaskNode,
        result: ExecutionResult,
        task_text: str,
    ) -> ExecutionResult:
        """Apply verification layers to a successful node output based on config.

        Layer application rules (cumulative):
        - Layer 1: Runs if verification_config.layer1_enabled is True.
        - Layer 2: Runs if node.complexity >= verification_config.layer2_threshold
          and a Layer 2 verifier is available (requires ModelRouter).
        - Layer 3: Runs if node.complexity >= verification_config.layer3_threshold
          AND verification_config.layer3_opt_in is True (not yet implemented).

        Each verification pass appends its VerificationResult to the collected
        list, which is attached to the returned ExecutionResult.

        Args:
            node: The TaskNode being verified.
            result: The ExecutionResult from node execution.
            task_text: The task description text (for building L2 messages).

        Returns:
            Potentially updated ExecutionResult after verification, with
            verification_results populated.
        """
        cfg = self._verification_config
        collected_vrs: list[VerificationResult] = []

        # Layer 1: rule-based format/type checks with single retry
        if cfg.layer1_enabled:
            verified_output, vr = await self._verifier.verify_and_retry(
                output=result.output,
                node=node,
                executor=self._executor,
                expects_json=False,
            )
            collected_vrs.append(vr)

            if vr.passed and vr.retry_count == 0:
                logger.info(
                    "Layer 1 verification passed for node %s",
                    node.id,
                )
            elif vr.passed and vr.retry_count > 0:
                logger.warning(
                    "Layer 1 verification failed for node %s, "
                    "retry succeeded (retry_count=%d)",
                    node.id,
                    vr.retry_count,
                )
            else:
                logger.warning(
                    "Layer 1 verification failed for node %s: %s (retry_count=%d)",
                    node.id,
                    "; ".join(vr.issues),
                    vr.retry_count,
                )

            if verified_output != result.output:
                result = ExecutionResult(
                    root_id=result.root_id,
                    output=verified_output,
                    success=result.success,
                    total_nodes=result.total_nodes,
                    total_tokens=result.total_tokens,
                    total_latency_ms=result.total_latency_ms,
                    total_cost=result.total_cost,
                    context_tokens=result.context_tokens,
                    model_used=result.model_used,
                    tools_used=result.tools_used,
                    llm_calls=result.llm_calls,
                    nodes=result.nodes,
                    errors=result.errors,
                    verification_results=tuple(collected_vrs),
                )

        # Layer 2: self-consistency via 3-way sampling
        if (
            node.complexity >= cfg.layer2_threshold
            and self._layer2_verifier is not None
        ):
            messages = [{"role": "user", "content": task_text}]
            logger.info(
                "Running Layer 2 verification for node %s (complexity=%d)",
                node.id,
                node.complexity,
            )
            sampling_result = await self._layer2_verifier.verify(
                node, messages,
            )

            l2_passed = not sampling_result.low_confidence
            l2_vr = VerificationResult(
                passed=l2_passed,
                issues=(
                    [f"Low confidence (score={sampling_result.agreement_score:.2f})"]
                    if not l2_passed
                    else []
                ),
                layer=2,
                retry_count=0,
            )
            collected_vrs.append(l2_vr)

            if l2_passed:
                logger.info(
                    "Layer 2 verification passed for node %s (score=%.2f)",
                    node.id,
                    sampling_result.agreement_score,
                )
            else:
                logger.warning(
                    "Layer 2 verification failed for node %s (score=%.2f)",
                    node.id,
                    sampling_result.agreement_score,
                )

            # Update result with L2 output and aggregated cost/tokens
            result = ExecutionResult(
                root_id=result.root_id,
                output=sampling_result.content,
                success=result.success,
                total_nodes=result.total_nodes,
                total_tokens=result.total_tokens + sampling_result.tokens_in + sampling_result.tokens_out,
                total_latency_ms=result.total_latency_ms + sampling_result.latency_ms,
                total_cost=result.total_cost + sampling_result.cost,
                context_tokens=result.context_tokens,
                model_used=result.model_used,
                tools_used=result.tools_used,
                llm_calls=result.llm_calls,
                nodes=result.nodes,
                errors=result.errors,
                verification_results=tuple(collected_vrs),
            )

        # Layer 3: CRITIC-style knowledge graph verification (placeholder)
        if (
            node.complexity >= cfg.layer3_threshold
            and cfg.layer3_opt_in
        ):
            logger.info(
                "Layer 3 verification eligible for node %s (complexity=%d) "
                "but not yet implemented",
                node.id,
                node.complexity,
            )

        # Attach collected verification results to the final result
        if collected_vrs and result.verification_results != tuple(collected_vrs):
            result = ExecutionResult(
                root_id=result.root_id,
                output=result.output,
                success=result.success,
                total_nodes=result.total_nodes,
                total_tokens=result.total_tokens,
                total_latency_ms=result.total_latency_ms,
                total_cost=result.total_cost,
                context_tokens=result.context_tokens,
                model_used=result.model_used,
                tools_used=result.tools_used,
                llm_calls=result.llm_calls,
                nodes=result.nodes,
                errors=result.errors,
                verification_results=tuple(collected_vrs),
            )

        return result

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
        all_verification_results: list[VerificationResult] = []
        all_success = True
        tools_count = 0
        llm_count = 0

        # Iterate in original leaf order for deterministic output
        for leaf in leaves:
            r = results.get(leaf.id)
            if r is None:
                continue
            total_tokens += r.total_tokens
            total_cost += r.total_cost
            all_nodes.extend(r.nodes)
            all_errors.extend(r.errors)
            all_verification_results.extend(r.verification_results)
            if not r.success:
                all_success = False
            if r.model_used and r.model_used.startswith("tool:"):
                tools_count += 1
            elif r.total_tokens > 0:
                llm_count += 1

        # Build leaf_outputs dict for the deterministic aggregator
        # Sanitize all outputs before aggregation to prevent injection in final result
        leaf_outputs: dict[str, str] = {}
        for leaf in leaves:
            r = results.get(leaf.id)
            if r is None or not r.output:
                continue
            sanitized_output = self._sanitizer.sanitize(r.output)
            node = node_by_id[leaf.id]
            if node.provides:
                try:
                    parsed = json.loads(sanitized_output)
                    if isinstance(parsed, dict):
                        for key in node.provides:
                            if key in parsed:
                                leaf_outputs[key] = self._sanitizer.sanitize(
                                    str(parsed[key])
                                )
                            else:
                                leaf_outputs[key] = sanitized_output
                    else:
                        for key in node.provides:
                            leaf_outputs[key] = sanitized_output
                except json.JSONDecodeError:
                    for key in node.provides:
                        leaf_outputs[key] = sanitized_output
            else:
                leaf_outputs[f"output_{leaf.id}"] = sanitized_output

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
            tools_used=tools_count,
            llm_calls=llm_count,
            nodes=tuple(all_nodes),
            errors=tuple(all_errors),
            verification_results=tuple(all_verification_results),
        )

    async def execute_loop(
        self,
        loop_node: LoopNode,
        body_nodes: list[TaskNode],
    ) -> ExecutionResult:
        """Execute a LoopNode by iterating its body nodes with retry-with-context.

        Each iteration:
        1. Runs the body nodes via the existing DAG executor.
        2. Checks the exit condition against the iteration output.
        3. If the condition is met, returns immediately with the output.
        4. If not, injects the previous iteration's output as context into
           the body nodes for the next iteration.
        5. Hard-stops at max_iterations regardless of the exit condition.

        Args:
            loop_node: The LoopNode describing iteration parameters.
            body_nodes: The list of TaskNode instances that form the loop body.

        Returns:
            ExecutionResult from the final iteration (or the iteration that
            satisfied the exit condition).
        """
        start = time.perf_counter()
        root_id = loop_node.id

        if loop_node.max_iterations <= 0:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Loop %s has max_iterations=%d, skipping execution",
                root_id,
                loop_node.max_iterations,
            )
            return ExecutionResult(
                root_id=root_id,
                output="",
                success=True,
                total_nodes=0,
                total_tokens=0,
                total_latency_ms=elapsed_ms,
                total_cost=0.0,
            )

        total_tokens = 0
        total_cost = 0.0
        all_errors: list[str] = []
        previous_output: str = ""
        last_result: ExecutionResult | None = None

        for iteration in range(1, loop_node.max_iterations + 1):
            logger.info(
                "Loop %s: starting iteration %d/%d",
                root_id,
                iteration,
                loop_node.max_iterations,
            )

            # Inject previous iteration context into body node descriptions
            iteration_body = self._prepare_iteration_body(
                body_nodes, previous_output, iteration,
            )

            # Execute the body nodes as a sub-DAG
            iter_result = await self.execute(iteration_body)

            total_tokens += iter_result.total_tokens
            total_cost += iter_result.total_cost
            all_errors.extend(iter_result.errors)
            last_result = iter_result

            # If any body node failed, stop the loop
            if not iter_result.success:
                logger.warning(
                    "Loop %s: body failed on iteration %d",
                    root_id,
                    iteration,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    root_id=root_id,
                    output=iter_result.output,
                    success=False,
                    total_nodes=iteration,
                    total_tokens=total_tokens,
                    total_latency_ms=elapsed_ms,
                    total_cost=total_cost,
                    errors=tuple(all_errors),
                )

            previous_output = iter_result.output

            # Check exit condition
            if self._check_exit_condition(
                loop_node.exit_condition, iter_result.output,
            ):
                logger.info(
                    "Loop %s: exit condition met on iteration %d",
                    root_id,
                    iteration,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    root_id=root_id,
                    output=iter_result.output,
                    success=True,
                    total_nodes=iteration,
                    total_tokens=total_tokens,
                    total_latency_ms=elapsed_ms,
                    total_cost=total_cost,
                    errors=tuple(all_errors),
                )

        # Exhausted all iterations without meeting exit condition
        logger.info(
            "Loop %s: reached max_iterations (%d) without exit condition",
            root_id,
            loop_node.max_iterations,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        final_output = last_result.output if last_result else ""
        return ExecutionResult(
            root_id=root_id,
            output=final_output,
            success=True,
            total_nodes=loop_node.max_iterations,
            total_tokens=total_tokens,
            total_latency_ms=elapsed_ms,
            total_cost=total_cost,
            errors=tuple(all_errors),
        )

    def _prepare_iteration_body(
        self,
        body_nodes: list[TaskNode],
        previous_output: str,
        iteration: int,
    ) -> list[TaskNode]:
        """Create fresh copies of body nodes with previous iteration context injected.

        On iteration 1, nodes are returned as-is. On subsequent iterations,
        the previous iteration's output is prepended to each body node's
        description so the executor can leverage prior context.

        Args:
            body_nodes: Original body node definitions.
            previous_output: Output text from the previous iteration (empty on
                first iteration).
            iteration: 1-based iteration number.

        Returns:
            List of TaskNode copies ready for execution in this iteration.
        """
        import copy

        copies: list[TaskNode] = []
        for node in body_nodes:
            node_copy = copy.copy(node)
            if iteration > 1 and previous_output:
                node_copy.description = (
                    f"<previous_iteration output=\"{iteration - 1}\">\n"
                    f"{previous_output}\n"
                    f"</previous_iteration>\n\n"
                    f"{node.description}"
                )
            copies.append(node_copy)
        return copies

    @staticmethod
    def _check_exit_condition(condition: str, output: str) -> bool:
        """Evaluate an exit condition string against iteration output.

        Supported condition formats:
        - ``""`` (empty): Never triggers early exit.
        - ``"contains:<substring>"``: True if output contains the substring
            (case-sensitive).

        Args:
            condition: The exit condition expression.
            output: The iteration output to check.

        Returns:
            True if the condition is satisfied, False otherwise.
        """
        if not condition:
            return False

        if ":" not in condition:
            logger.warning(
                "Malformed exit condition (no colon separator): %r",
                condition,
            )
            return False

        check_type, _, value = condition.partition(":")

        if check_type == "contains":
            return value in output

        logger.warning("Unknown exit condition type: %r", check_type)
        return False
