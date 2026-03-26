"""Graph context enrichment for single-call mode.

Assembles the richest possible context from the knowledge graph for a given
task description. Pulls relevant entities (via EntityResolver), conversation
history (via ConversationMemory), failure reflections (via get_relevant_reflections),
and similar patterns (via PatternMatcher).

Uses activation-ranked retrieval (ACT-R model) so that the most relevant
context appears first. Optionally uses PPR-based retrieval for deep multi-hop
context when use_ppr is True.

Each section of the returned EnrichedContext carries a token estimate for
downstream budget enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core_gb.conversation import ConversationMemory
from core_gb.patterns import PatternMatcher, PatternStore
from core_gb.types import Pattern
from graph.activation import ActivationModel
from graph.context import assemble_context, get_relevant_reflections
from graph.resolver import EntityResolver
from graph.store import GraphStore
from graph.subgraph import SubgraphRetriever, SubgraphResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichedContext:
    """Enriched context assembled from multiple knowledge graph sources.

    Each section carries its own token estimate so callers can enforce
    per-section or total budget constraints.

    Attributes:
        entities: Resolved entity dicts from the knowledge graph.
        memories: Active memory strings from the knowledge graph.
        reflections: Failure reflection dicts from past task executions.
        patterns: Matched Pattern objects from the pattern cache.
        conversation_turns: Recent conversation history as role/content dicts.
        entity_tokens: Estimated token cost of the entities section.
        memory_tokens: Estimated token cost of the memories section.
        reflection_tokens: Estimated token cost of the reflections section.
        pattern_tokens: Estimated token cost of the patterns section.
        conversation_tokens: Estimated token cost of the conversation section.
    """

    entities: tuple[dict[str, str], ...] = ()
    memories: tuple[str, ...] = ()
    reflections: tuple[dict[str, str], ...] = ()
    patterns: tuple[Pattern, ...] = ()
    conversation_turns: tuple[dict[str, str], ...] = ()
    relationship_descriptions: tuple[str, ...] = ()
    community_summaries: tuple[str, ...] = ()
    entity_tokens: int = 0
    memory_tokens: int = 0
    reflection_tokens: int = 0
    pattern_tokens: int = 0
    conversation_tokens: int = 0
    relationship_tokens: int = 0
    community_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of all section token estimates."""
        return (
            self.entity_tokens
            + self.memory_tokens
            + self.reflection_tokens
            + self.pattern_tokens
            + self.conversation_tokens
            + self.relationship_tokens
            + self.community_tokens
        )


class ContextEnricher:
    """Assembles rich context from the knowledge graph for a single task.

    Combines multiple retrieval strategies:
    1. Entity resolution -- finds graph entities mentioned in the task
    2. Graph context assembly -- 2-hop or PPR-based traversal from resolved entities
    3. Conversation history -- recent turns from the current chat
    4. Failure reflections -- past failures on similar tasks
    5. Pattern matching -- reusable execution templates

    All results are ranked by ACT-R activation (most relevant first) and
    trimmed to fit the token budget.

    Args:
        store: Initialized GraphStore instance.
        max_conversation_turns: Maximum number of conversation turns to include.
            Defaults to 10.
    """

    def __init__(
        self,
        store: GraphStore,
        max_conversation_turns: int = 10,
    ) -> None:
        self._store = store
        self._resolver = EntityResolver(store)
        self._conversation = ConversationMemory(store, max_messages=max_conversation_turns)
        self._pattern_store = PatternStore(store)
        self._pattern_matcher = PatternMatcher()
        self._activation_model = ActivationModel()
        self._subgraph_retriever = SubgraphRetriever(store)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using the same heuristic as GraphStore."""
        return max(1, int(len(text.split()) * 1.3))

    def enrich(
        self,
        task_description: str,
        *,
        chat_id: str | None = None,
        use_ppr: bool = False,
        max_tokens: int = 4000,
    ) -> EnrichedContext:
        """Assemble enriched context for a task description.

        Pulls context from all available sources in the knowledge graph,
        ranks by activation, and trims to fit the token budget.

        Args:
            task_description: The task to assemble context for.
            chat_id: Optional conversation ID for including chat history.
            use_ppr: If True, use PPR-based retrieval for deep multi-hop
                context instead of standard 2-hop traversal.
            max_tokens: Total token budget across all sections. Defaults to 4000.

        Returns:
            An EnrichedContext with all retrieved sections and token estimates.
        """
        if not task_description.strip():
            return EnrichedContext()

        if max_tokens <= 0:
            return EnrichedContext()

        remaining_budget = max_tokens

        # 1. Resolve entities from the task description
        entities, memories, entity_tokens, memory_tokens = self._retrieve_entities(
            task_description,
            use_ppr=use_ppr,
            max_tokens=remaining_budget,
        )
        remaining_budget -= (entity_tokens + memory_tokens)

        # 1b. Retrieve subgraph context (relationships + communities)
        resolved = self._resolver.resolve(task_description)
        entity_ids = [eid for eid, _score in resolved]
        relationships, communities, rel_tokens, comm_tokens = self._retrieve_subgraph(
            entity_ids,
            max_tokens=max(0, remaining_budget),
        )
        remaining_budget -= (rel_tokens + comm_tokens)

        # 2. Retrieve conversation history
        conversation_turns, conversation_tokens = self._retrieve_conversation(
            chat_id,
            max_tokens=max(0, remaining_budget),
        )
        remaining_budget -= conversation_tokens

        # 3. Retrieve failure reflections
        reflections, reflection_tokens = self._retrieve_reflections(
            task_description,
            max_tokens=max(0, remaining_budget),
        )
        remaining_budget -= reflection_tokens

        # 4. Retrieve matching patterns
        patterns, pattern_tokens = self._retrieve_patterns(
            task_description,
            max_tokens=max(0, remaining_budget),
        )

        return EnrichedContext(
            entities=tuple(entities),
            memories=tuple(memories),
            reflections=tuple(reflections),
            patterns=tuple(patterns),
            conversation_turns=tuple(conversation_turns),
            relationship_descriptions=tuple(relationships),
            community_summaries=tuple(communities),
            entity_tokens=entity_tokens,
            memory_tokens=memory_tokens,
            reflection_tokens=reflection_tokens,
            pattern_tokens=pattern_tokens,
            conversation_tokens=conversation_tokens,
            relationship_tokens=rel_tokens,
            community_tokens=comm_tokens,
        )

    def _retrieve_subgraph(
        self,
        entity_ids: list[str],
        *,
        max_tokens: int,
    ) -> tuple[list[str], list[str], int, int]:
        """Retrieve subgraph context: relationship descriptions and community summaries.

        Uses SubgraphRetriever to perform multi-hop traversal, then extracts
        relationship descriptions and community summaries with token budgeting.

        Returns:
            (relationship_descriptions, community_summaries, rel_tokens, comm_tokens)
        """
        if not entity_ids or max_tokens <= 0:
            return [], [], 0, 0

        subgraph = self._subgraph_retriever.retrieve_subgraph(entity_ids)

        # Budget relationship descriptions first (more specific context).
        rel_descriptions: list[str] = []
        rel_tokens = 0
        seen_descriptions: set[str] = set()
        for edge in subgraph.edges:
            if edge.description and edge.description not in seen_descriptions:
                seen_descriptions.add(edge.description)
                cost = self._estimate_tokens(edge.description)
                if rel_tokens + cost > max_tokens:
                    break
                rel_tokens += cost
                rel_descriptions.append(edge.description)

        # Budget community summaries with remaining tokens.
        comm_budget = max_tokens - rel_tokens
        comm_summaries: list[str] = []
        comm_tokens = 0
        for summary in subgraph.community_summaries:
            cost = self._estimate_tokens(summary)
            if comm_tokens + cost > comm_budget:
                break
            comm_tokens += cost
            comm_summaries.append(summary)

        return rel_descriptions, comm_summaries, rel_tokens, comm_tokens

    def _retrieve_entities(
        self,
        task_description: str,
        *,
        use_ppr: bool,
        max_tokens: int,
    ) -> tuple[list[dict[str, str]], list[str], int, int]:
        """Resolve entities and assemble graph context.

        Returns:
            (entities, memories, entity_tokens, memory_tokens)
        """
        if max_tokens <= 0:
            return [], [], 0, 0

        # Resolve entity mentions in the task description
        resolved = self._resolver.resolve(task_description)
        entity_ids = [eid for eid, _score in resolved]

        if not entity_ids:
            return [], [], 0, 0

        # Assemble context using 2-hop or PPR
        graph_ctx = assemble_context(
            self._store,
            entity_ids,
            use_ppr=use_ppr,
            max_tokens=max_tokens,
        )

        # Rank entities by activation score (highest first)
        ranked_entities = self._rank_entities_by_activation(
            list(graph_ctx.relevant_entities),
        )

        # Compute per-section token costs
        entity_tokens = 0
        budgeted_entities: list[dict[str, str]] = []
        for entity in ranked_entities:
            text = f"{entity.get('type', '')} {entity.get('name', '')} {entity.get('details', '')}"
            cost = self._estimate_tokens(text)
            if entity_tokens + cost > max_tokens:
                break
            entity_tokens += cost
            budgeted_entities.append(entity)

        memory_tokens = 0
        budgeted_memories: list[str] = []
        memory_budget = max_tokens - entity_tokens
        for memory in graph_ctx.active_memories:
            cost = self._estimate_tokens(memory)
            if memory_tokens + cost > memory_budget:
                break
            memory_tokens += cost
            budgeted_memories.append(memory)

        return budgeted_entities, budgeted_memories, entity_tokens, memory_tokens

    def _rank_entities_by_activation(
        self,
        entities: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Rank entity dicts by ACT-R activation score.

        Looks up each entity in the graph to get access metadata, then
        scores and sorts by activation descending.
        """
        if not entities:
            return []

        scored: list[tuple[dict[str, str], float]] = []
        for entity in entities:
            name = entity.get("name", "")
            # Try to find the node in the graph to get activation metadata
            score = self._lookup_activation_score(name)
            scored.append((entity, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _score in scored]

    def _lookup_activation_score(self, name: str) -> float:
        """Look up the ACT-R activation score for a named entity.

        Searches for the entity by name, extracts access metadata, and
        computes the activation score. Returns 0.0 if not found.
        """
        resolved = self._resolver.resolve(name, top_k=1)
        if not resolved:
            return 0.0

        entity_id = resolved[0][0]
        found = self._store._find_node_table(entity_id)
        if found is None:
            return 0.0

        _table, props = found
        return self._store._node_activation_score(self._activation_model, props)

    def _retrieve_conversation(
        self,
        chat_id: str | None,
        *,
        max_tokens: int,
    ) -> tuple[list[dict[str, str]], int]:
        """Retrieve conversation history for the given chat_id.

        Returns:
            (conversation_turns, token_cost)
        """
        if chat_id is None or max_tokens <= 0:
            return [], 0

        history = self._conversation.get_history(chat_id)
        if not history:
            return [], 0

        # Apply token budget to conversation turns (most recent first is
        # already handled by ConversationMemory; here we trim if over budget)
        budgeted: list[dict[str, str]] = []
        total_tokens = 0
        for turn in history:
            text = f"{turn.get('role', '')}: {turn.get('content', '')}"
            cost = self._estimate_tokens(text)
            if total_tokens + cost > max_tokens:
                break
            total_tokens += cost
            budgeted.append(turn)

        return budgeted, total_tokens

    def _retrieve_reflections(
        self,
        task_description: str,
        *,
        max_tokens: int,
    ) -> tuple[list[dict[str, str]], int]:
        """Retrieve failure reflections relevant to the task description.

        Returns:
            (reflections, token_cost)
        """
        if max_tokens <= 0:
            return [], 0

        reflections = get_relevant_reflections(self._store, task_description)
        if not reflections:
            return [], 0

        budgeted: list[dict[str, str]] = []
        total_tokens = 0
        for refl in reflections:
            text = (
                f"{refl.get('task_description', '')} "
                f"{refl.get('what_failed', '')} "
                f"{refl.get('why', '')} "
                f"{refl.get('what_to_try', '')}"
            )
            cost = self._estimate_tokens(text)
            if total_tokens + cost > max_tokens:
                break
            total_tokens += cost
            budgeted.append(refl)

        return budgeted, total_tokens

    def _retrieve_patterns(
        self,
        task_description: str,
        *,
        max_tokens: int,
    ) -> tuple[list[Pattern], int]:
        """Retrieve matching patterns from the pattern cache.

        Returns:
            (patterns, token_cost)
        """
        if max_tokens <= 0:
            return [], 0

        all_patterns = self._pattern_store.load_all()
        if not all_patterns:
            return [], 0

        match_result = self._pattern_matcher.match(task_description, all_patterns)
        if match_result is None:
            return [], 0

        matched_pattern, _bindings = match_result

        # Estimate token cost for the pattern
        text = f"{matched_pattern.trigger} {matched_pattern.description}"
        cost = self._estimate_tokens(text)

        if cost > max_tokens:
            return [], 0

        return [matched_pattern], cost
