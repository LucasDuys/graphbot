"""Parallel DAG executor using streaming topological dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.router import ModelRouter
    from tools_gb.registry import ToolRegistry

from core_gb.aggregator import Aggregator
from core_gb.sanitizer import OutputSanitizer
from core_gb.types import ExecutionResult, TaskNode
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
    ) -> None:
        self._executor = executor
        self._semaphore = asyncio.Semaphore(max_concurrency)
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

                    # Register provided data keys (sanitize before forwarding)
                    if result.success:
                        node = node_by_id[node_id]
                        output_text = self._sanitizer.sanitize(result.output)
                        for key in node.provides:
                            raw_value = node.output_data.get(key, output_text)
                            data_registry[key] = self._sanitizer.sanitize(raw_value)
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
