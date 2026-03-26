"""Orchestrator -- wires intake, decomposer, and executor into a single flow."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from core_gb.context_enrichment import ContextEnricher, EnrichedContext
from core_gb.conversation import ConversationMemory
from core_gb.dag_executor import DAGExecutor
from core_gb.decomposer import Decomposer
from core_gb.executor import SimpleExecutor
from core_gb.intake import IntakeParser, IntakeResult, TaskType
from core_gb.patterns import PatternMatcher, PatternStore
from core_gb.constitution import ConstitutionalChecker
from core_gb.safety import IntentClassifier
from core_gb.single_executor import SingleCallExecutor
from core_gb.tool_factory import ToolFactory
from core_gb.types import Domain, ExecutionResult, GraphContext, Pattern, TaskNode, TaskStatus
from core_gb.verification import VerificationConfig
from core_gb.wave_event import WaveCompleteEvent
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

    # Domains that require tool access and therefore need decomposition.
    _TOOL_DOMAINS: frozenset[Domain] = frozenset({
        Domain.FILE,
        Domain.WEB,
        Domain.CODE,
        Domain.BROWSER,
    })

    def __init__(
        self,
        store: GraphStore,
        router: ModelRouter,
        verification_config: VerificationConfig | None = None,
        enable_replan: bool = False,
        force_decompose: bool = False,
    ) -> None:
        self._store = store
        self._router = router
        self._verification_config = verification_config or VerificationConfig()
        self._enable_replan = enable_replan
        self._force_decompose = force_decompose
        self._intake = IntakeParser()
        self._tool_registry = ToolRegistry(workspace=str(Path.cwd()), router=router)
        self._tool_factory = ToolFactory(router=router, store=store)
        self._tool_registry.set_tool_factory(self._tool_factory)

        # Load previously persisted tools from the knowledge graph.
        loaded = self._tool_factory.load_from_graph()
        if loaded > 0:
            logger.info("Loaded %d persisted tool(s) from knowledge graph", loaded)

        self._executor = SimpleExecutor(store, router, tool_registry=self._tool_registry)
        self._dag_executor = DAGExecutor(
            self._executor,
            tool_registry=self._tool_registry,
            verification_config=self._verification_config,
            router=router,
        )
        self._decomposer = Decomposer(router)
        self._resolver = EntityResolver(store)
        self._pattern_store = PatternStore(store)
        self._pattern_matcher = PatternMatcher()
        self._graph_updater = GraphUpdater(store)
        self._intent_classifier = IntentClassifier()
        self._constitutional_checker = ConstitutionalChecker()
        self._conversation_memory = ConversationMemory(store)
        self._context_enricher = ContextEnricher(store)
        self._single_executor = SingleCallExecutor(router)

        if self._enable_replan:
            self._dag_executor.set_replan_callback(self._replan_callback)

    async def process(
        self, message: str, chat_id: str | None = None
    ) -> ExecutionResult:
        """Process a user message end-to-end.

        Flow:
        0. Pre-decomposition safety: scan raw message for harmful intent (zero LLM cost)
        1. IntakeParser.parse(message) -> IntakeResult
        1b. FAST PATH: trivial queries (greetings, acks) return immediately (<50ms)
        2. Check pattern cache for a match (skip decomposition if hit)
           -- skipped for complexity=1 (no patterns worth matching)
        3. If simple (is_simple=True): route directly to SimpleExecutor
           -- skip entity resolution for complexity=1
        4. If complex:
           a. Resolve entities and assemble graph context
           b. Decompose via Decomposer
           c. Execute DAG in parallel via DAGExecutor
           d. Aggregate results into single ExecutionResult
        5. Update knowledge graph with execution outcome

        Args:
            message: The user's message text.
            chat_id: Optional conversation identifier for multi-turn memory.
                When provided, previous messages for this chat_id are included
                in the LLM context, and the current exchange is persisted.

        Each pipeline stage logs its latency at DEBUG level for profiling.
        """
        pipeline_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Conversation memory: retrieve history and store incoming message
        # ------------------------------------------------------------------
        conversation_history: list[dict[str, str]] = []
        if chat_id is not None:
            conversation_history = self._conversation_memory.format_as_messages(chat_id)
            self._conversation_memory.add_message(chat_id, "user", message)

        # ------------------------------------------------------------------
        # Pre-decomposition safety: block obviously harmful messages before
        # any LLM calls, pattern matching, or decomposition. This is a
        # zero-cost regex scan on the raw user input.
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        pre_safety = self._intent_classifier.classify_text(message)
        if pre_safety.blocked:
            return self._blocked_result(message, pre_safety.reason)

        pre_constitutional = self._constitutional_checker.check_text(message)
        if not pre_constitutional.passed:
            reasons = "; ".join(
                f"{name}: {reason}"
                for name, reason in pre_constitutional.violations
            )
            return self._blocked_result(message, reasons)
        self._log_stage_latency("safety_check", t0)

        # ------------------------------------------------------------------
        # Stage 1: Intake parse
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        intake = self._intake.parse(message)
        self._log_stage_latency("intake_parse", t0)

        # ------------------------------------------------------------------
        # Stage 1b: FAST PATH -- trivial queries (greetings, acks)
        # Return canned response immediately, skip entire pipeline.
        # No LLM call, no pattern check, no entity resolution.
        # ------------------------------------------------------------------
        if intake.is_trivial is True:
            t0 = time.perf_counter()
            trivial_text = self._intake.trivial_response(intake)
            if trivial_text is not None:
                self._log_stage_latency("trivial_fast_path", t0)
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                logger.debug(
                    "Pipeline total latency: %.1fms (trivial fast path)",
                    total_ms,
                )
                trivial_result = ExecutionResult(
                    root_id="trivial",
                    output=trivial_text,
                    success=True,
                    total_nodes=0,
                    total_tokens=0,
                    total_latency_ms=total_ms,
                    llm_calls=0,
                )
                self._store_assistant_response(chat_id, trivial_result)
                return trivial_result

        # ------------------------------------------------------------------
        # Stage 2: Pattern cache check (skip for trivial/simple queries)
        # Only load patterns when complexity > 1 -- simple queries never
        # match patterns worth caching.
        # ------------------------------------------------------------------
        if intake.complexity > 1:
            t0 = time.perf_counter()
            patterns = self._pattern_store.load_all()
            self._log_stage_latency("pattern_load", t0)
            if patterns:
                t0 = time.perf_counter()
                match_result = self._pattern_matcher.match(message, patterns)
                self._log_stage_latency("pattern_match", t0)
                if match_result is not None:
                    pattern, bindings = match_result
                    nodes = self._instantiate_pattern(pattern, bindings)
                    if nodes:
                        # Safety check on pattern-instantiated DAG
                        verdict = self._intent_classifier.classify_dag(nodes)
                        if verdict.blocked:
                            return self._blocked_result(message, verdict.reason)

                        # Constitutional check on pattern-instantiated DAG
                        const_verdict = self._constitutional_checker.check_plan(nodes)
                        if not const_verdict.passed:
                            reasons = "; ".join(
                                f"{name}: {reason}"
                                for name, reason in const_verdict.violations
                            )
                            return self._blocked_result(message, reasons)

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
                        self._log_pipeline_total(pipeline_start, "pattern_hit")
                        self._store_assistant_response(chat_id, result)
                        return result

        # ------------------------------------------------------------------
        # Stage 3: Smart routing -- single-call vs decomposition
        # ------------------------------------------------------------------
        should_decompose, decompose_reason = self._should_decompose(intake)

        if not should_decompose:
            # Single-call path: enrich context from knowledge graph, then
            # make exactly one LLM call via SingleCallExecutor.
            logger.info(
                "Routing to single-call (complexity=%d, domain=%s)",
                intake.complexity,
                intake.domain.value,
            )
            t0 = time.perf_counter()
            enriched = self._context_enricher.enrich(
                message, chat_id=chat_id,
            )
            self._log_stage_latency("context_enrich", t0)

            graph_ctx = self._enriched_to_graph_context(enriched)

            t0 = time.perf_counter()
            result = await self._single_executor.execute(
                task=message,
                graph_context=graph_ctx,
                conversation_history=conversation_history or None,
                pattern_hints=list(enriched.patterns) or None,
                complexity=intake.complexity,
                domain=intake.domain,
            )
            self._log_stage_latency("single_call_execute", t0)

            node = TaskNode(
                id=result.root_id,
                description=message,
                is_atomic=True,
                domain=intake.domain,
                complexity=intake.complexity,
                status=TaskStatus.READY,
            )
            self._graph_updater.update(message, [node], result)
            self._log_pipeline_total(pipeline_start, "single_call")
            self._store_assistant_response(chat_id, result)
            return result

        # Decomposition path: task is complex enough or requires tools.
        logger.info(
            "Routing to decomposition (reason: %s)", decompose_reason,
        )

        # ------------------------------------------------------------------
        # Stage 4: Complex path -- resolve entities, get context, decompose
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        entity_ids: list[str] = []
        for entity in intake.entities:
            matches = self._resolver.resolve(entity, top_k=1)
            for eid, confidence in matches:
                if confidence > 0.5 and eid not in entity_ids:
                    entity_ids.append(eid)
        self._log_stage_latency("entity_resolution", t0)

        t0 = time.perf_counter()
        context = self._store.get_context(entity_ids) if entity_ids else None
        self._log_stage_latency("context_assembly", t0)

        t0 = time.perf_counter()
        nodes = await self._decomposer.decompose(message, context)
        self._log_stage_latency("decompose", t0)

        # Safety check on decomposed DAG before execution
        verdict = self._intent_classifier.classify_dag(nodes)
        if verdict.blocked:
            return self._blocked_result(message, verdict.reason)

        # Constitutional check on decomposed DAG before execution
        const_verdict = self._constitutional_checker.check_plan(nodes)
        if not const_verdict.passed:
            reasons = "; ".join(
                f"{name}: {reason}"
                for name, reason in const_verdict.violations
            )
            return self._blocked_result(message, reasons)

        # Filter to atomic (leaf) nodes only
        leaves = [n for n in nodes if n.is_atomic]

        if not leaves:
            # Edge case: no leaves found, treat entire message as single task
            result = await self._executor.execute(
                message, intake.complexity,
                conversation_history=conversation_history or None,
            )
            self._graph_updater.update(message, nodes, result)
            self._log_pipeline_total(pipeline_start, "complex_no_leaves")
            self._store_assistant_response(chat_id, result)
            return result

        if len(leaves) == 1:
            # Single leaf: execute directly
            t0 = time.perf_counter()
            result = await self._execute_single_leaf(
                leaves[0], message,
                conversation_history=conversation_history or None,
            )
            self._log_stage_latency("single_leaf_execute", t0)
            self._graph_updater.update(message, nodes, result)
            self._log_pipeline_total(pipeline_start, "complex_single_leaf")
            self._store_assistant_response(chat_id, result)
            return result

        t0 = time.perf_counter()
        self._dag_executor.aggregation_template = self._decomposer.last_template
        self._dag_executor.original_question = message
        result = await self._dag_executor.execute(nodes)
        self._log_stage_latency("dag_execute", t0)
        self._graph_updater.update(message, nodes, result)
        self._log_pipeline_total(pipeline_start, "complex_dag")
        self._store_assistant_response(chat_id, result)
        return result

    @staticmethod
    def _log_stage_latency(stage: str, start: float) -> None:
        """Log the latency of a pipeline stage at DEBUG level."""
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("Pipeline stage [%s] latency: %.1fms", stage, elapsed_ms)

    @staticmethod
    def _log_pipeline_total(start: float, path: str) -> None:
        """Log the total pipeline latency at DEBUG level."""
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "Pipeline total latency: %.1fms (path: %s)", elapsed_ms, path,
        )

    def _store_assistant_response(
        self, chat_id: str | None, result: ExecutionResult
    ) -> None:
        """Persist the assistant's response in conversation memory.

        No-op when chat_id is None (no conversation tracking).
        """
        if chat_id is not None and result.output:
            self._conversation_memory.add_message(chat_id, "assistant", result.output)

    def _should_decompose(self, intake: IntakeResult) -> tuple[bool, str]:
        """Decide whether a task requires decomposition or can use single-call.

        Returns:
            A (should_decompose, reason) tuple. If should_decompose is False,
            reason is empty. If True, reason describes why decomposition was
            chosen.
        """
        if self._force_decompose:
            return True, "force_decompose=True"

        if intake.complexity >= 4:
            return True, f"complexity={intake.complexity} >= 4"

        if intake.domain in self._TOOL_DOMAINS:
            return True, f"domain={intake.domain.value} requires tools"

        return False, ""

    @staticmethod
    def _enriched_to_graph_context(enriched: EnrichedContext) -> GraphContext:
        """Convert an EnrichedContext to a GraphContext for SingleCallExecutor."""
        return GraphContext(
            relevant_entities=enriched.entities,
            active_memories=enriched.memories,
            matching_patterns=enriched.patterns,
            reflections=enriched.reflections,
            total_tokens=enriched.total_tokens,
            token_count=enriched.total_tokens,
        )

    @staticmethod
    def _blocked_result(message: str, reason: str) -> ExecutionResult:
        """Return an error ExecutionResult when the DAG is blocked by safety checks."""
        logger.warning("DAG execution blocked: %s", reason)
        return ExecutionResult(
            root_id="blocked",
            output=f"Request blocked by safety classifier: {reason}",
            success=False,
            errors=(reason,),
        )

    async def _execute_single_leaf(
        self,
        leaf: TaskNode,
        original_message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ExecutionResult:
        """Execute a single leaf node via SimpleExecutor."""
        task_text = leaf.description or original_message
        return await self._executor.execute(
            task_text, leaf.complexity,
            conversation_history=conversation_history,
        )

    async def _replan_callback(
        self, event: WaveCompleteEvent
    ) -> list[TaskNode] | None:
        """Re-plan remaining nodes using accumulated results as context.

        Called by DAGExecutor after each wave when enable_replan is True.
        Passes accumulated results to the Decomposer to produce a revised
        plan for the remaining work. Returns None if re-planning is not
        applicable (e.g. no remaining nodes, or first wave has not
        produced meaningful context yet).

        Args:
            event: The WaveCompleteEvent with completed and remaining info.

        Returns:
            A list of replacement TaskNode instances, or None to skip
            re-planning for this wave.
        """
        if not event.remaining_nodes:
            return None

        # Build a context summary from accumulated results
        result_summaries: list[str] = []
        for node_id, output in event.accumulated_results.items():
            result_summaries.append(f"[{node_id}]: {output}")

        context_str = "\n".join(result_summaries)
        remaining_desc = ", ".join(event.remaining_nodes)

        replan_prompt = (
            f"[Re-plan with intermediate results]\n"
            f"Completed results so far:\n{context_str}\n\n"
            f"Remaining tasks to re-plan: {remaining_desc}\n"
            f"Produce a revised plan for the remaining work, "
            f"taking the completed results into account."
        )

        try:
            revised_nodes = await self._decomposer.decompose(replan_prompt)
            if revised_nodes:
                logger.info(
                    "Re-planned %d remaining nodes into %d new nodes "
                    "after wave %d",
                    len(event.remaining_nodes),
                    len(revised_nodes),
                    event.wave_index,
                )
                return revised_nodes
        except Exception as exc:
            logger.warning("Re-planning failed: %s", exc)

        return None

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
