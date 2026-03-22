"""Context assembly with optional PPR-based retrieval.

Provides two context assembly strategies:
1. Standard 2-hop traversal via GraphStore.get_context (default)
2. PPR-based retrieval that spreads activation from seed entities through the
   graph, reaching relevant nodes far beyond the 2-hop boundary

Also provides reflection retrieval: queries the knowledge graph for Memory nodes
with category="reflection" linked (via REFLECTION_OF edges) to Task nodes with
similar descriptions, using Levenshtein similarity to rank relevance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import Levenshtein

from core_gb.types import GraphContext
from graph.retrieval import PPRRetriever
from graph.store import GraphStore

logger = logging.getLogger(__name__)

# Default PPR top-K: how many nodes PPR returns before token budgeting.
_DEFAULT_PPR_TOP_K: int = 20

# Default similarity threshold: reflections below this Levenshtein ratio are
# excluded. 0.4 is intentionally permissive -- even loosely related past
# failures can be valuable context for decomposition.
_DEFAULT_SIMILARITY_THRESHOLD: float = 0.4

# Default maximum number of reflections to return.
_DEFAULT_MAX_RESULTS: int = 5


def assemble_context(
    store: GraphStore,
    entity_ids: list[str],
    *,
    use_ppr: bool = False,
    max_tokens: int = 2500,
    ppr_top_k: int = _DEFAULT_PPR_TOP_K,
) -> GraphContext:
    """Assemble context from the knowledge graph for the given entity IDs.

    When ``use_ppr`` is False (default), delegates directly to the standard
    2-hop traversal in ``GraphStore.get_context``. This preserves full backward
    compatibility.

    When ``use_ppr`` is True, seeds a Personalized PageRank computation from the
    entity IDs, retrieves the top-K ranked nodes, and constructs a GraphContext
    from their properties. Falls back to 2-hop if PPR returns empty results.

    Args:
        store: The GraphStore to query.
        entity_ids: Entity IDs to seed context from (e.g., from entity resolution).
        use_ppr: If True, use PPR-based retrieval instead of 2-hop traversal.
            Defaults to False for backward compatibility.
        max_tokens: Token budget for the assembled context. Defaults to 2500.
        ppr_top_k: Maximum number of nodes PPR retrieves before token budgeting.
            Only used when use_ppr=True. Defaults to 20.

    Returns:
        A GraphContext with entities, memories, user summary, and token count.
    """
    if not entity_ids:
        return GraphContext()

    if not use_ppr:
        return store.get_context(entity_ids, max_tokens=max_tokens)

    # PPR mode: seed from entity IDs, retrieve top-K by PPR score
    retriever = PPRRetriever(store=store)
    ppr_results = retriever.retrieve(seed_ids=entity_ids, top_k=ppr_top_k)

    # Fallback: if PPR returns nothing, use 2-hop
    if not ppr_results:
        logger.debug(
            "PPR returned empty results for seeds %s; falling back to 2-hop",
            entity_ids,
        )
        return store.get_context(entity_ids, max_tokens=max_tokens)

    # Build GraphContext from PPR-ranked node IDs
    ranked_ids = [node_id for node_id, _score in ppr_results]
    return _build_context_from_node_ids(store, ranked_ids, max_tokens=max_tokens)


def _build_context_from_node_ids(
    store: GraphStore,
    ranked_ids: list[str],
    *,
    max_tokens: int = 2500,
) -> GraphContext:
    """Build a GraphContext from a pre-ranked list of node IDs.

    Looks up each node in the graph, classifies it as a User (for user_summary),
    Memory (for active_memories), or general entity (for relevant_entities),
    and applies token budgeting in the same order PPR ranked them.

    Args:
        store: The GraphStore to look up nodes.
        ranked_ids: Node IDs sorted by relevance (highest first, e.g., PPR score).
        max_tokens: Token budget for the assembled context.

    Returns:
        A fully populated GraphContext.
    """
    user_summary: str = ""
    entity_candidates: list[dict[str, str]] = []
    memory_candidates: list[str] = []

    now = datetime.now(timezone.utc)

    for node_id in ranked_ids:
        found = store._find_node_table(node_id)
        if found is None:
            continue
        table_name, props = found

        if table_name == "Memory":
            # Filter: only active memories (valid_until is NULL or in the future)
            valid_until = props.get("valid_until")
            if valid_until is not None:
                if isinstance(valid_until, datetime):
                    if valid_until.tzinfo is None:
                        valid_until = valid_until.replace(tzinfo=timezone.utc)
                    if valid_until < now:
                        continue
                elif isinstance(valid_until, str) and valid_until.strip():
                    try:
                        vt = datetime.fromisoformat(valid_until)
                        if vt.tzinfo is None:
                            vt = vt.replace(tzinfo=timezone.utc)
                        if vt < now:
                            continue
                    except ValueError:
                        pass
            content = str(props.get("content", ""))
            if content:
                memory_candidates.append(content)
        elif table_name == "User":
            name = str(props.get("name", ""))
            role = str(props.get("role", ""))
            institution = str(props.get("institution", ""))
            interests = str(props.get("interests", ""))
            parts = [p for p in [name, role, institution, interests] if p]
            if not user_summary and parts:
                user_summary = " | ".join(parts)
            entity_dict: dict[str, str] = {
                "type": table_name,
                "name": name,
                "details": " | ".join([role, institution, interests]),
            }
            entity_candidates.append(entity_dict)
        else:
            name = str(props.get("name", props.get("id", "")))
            detail_parts: list[str] = []
            for key in (
                "path", "language", "framework", "status", "type", "url",
                "relationship", "platform", "description",
            ):
                val = props.get(key)
                if val:
                    detail_parts.append(f"{key}={val}")
            entity_dict = {
                "type": table_name,
                "name": name,
                "details": ", ".join(detail_parts),
            }
            entity_candidates.append(entity_dict)

    # Apply token budget: entities first (in PPR rank order), then memories.
    token_budget = max_tokens
    used_tokens = store._estimate_tokens(user_summary) if user_summary else 0

    final_entities: list[dict[str, str]] = []
    for entity_dict in entity_candidates:
        text = (
            f"{entity_dict.get('type', '')} "
            f"{entity_dict.get('name', '')} "
            f"{entity_dict.get('details', '')}"
        )
        cost = store._estimate_tokens(text)
        if used_tokens + cost > token_budget:
            break
        used_tokens += cost
        final_entities.append(entity_dict)

    final_memories: list[str] = []
    for content in memory_candidates:
        cost = store._estimate_tokens(content)
        if used_tokens + cost > token_budget:
            break
        used_tokens += cost
        final_memories.append(content)

    return GraphContext(
        user_summary=user_summary,
        relevant_entities=tuple(final_entities),
        active_memories=tuple(final_memories),
        matching_patterns=(),
        total_tokens=used_tokens,
        token_count=used_tokens,
    )


def get_relevant_reflections(
    store: GraphStore,
    task_description: str,
    *,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    max_results: int = _DEFAULT_MAX_RESULTS,
) -> list[dict[str, str]]:
    """Retrieve reflections relevant to a task description from the knowledge graph.

    Queries for Memory nodes with category="reflection" that are linked via
    REFLECTION_OF edges to Task nodes. Ranks by Levenshtein similarity between
    the query task_description and the linked Task's description.

    Args:
        store: The GraphStore to query.
        task_description: Description of the current task to find relevant
            past failures for.
        similarity_threshold: Minimum Levenshtein ratio (0.0-1.0) for a
            reflection to be considered relevant. Defaults to 0.4.
        max_results: Maximum number of reflections to return. Defaults to 5.

    Returns:
        List of dicts, each containing:
            - task_description: the original failed task's description
            - what_failed: what went wrong
            - why: root cause analysis
            - what_to_try: suggestion for next attempt
        Sorted by similarity descending (most relevant first).
    """
    if not task_description.strip():
        return []

    # Query all reflection Memory nodes with their linked Task descriptions.
    # REFLECTION_OF: Memory -> Task
    rows = store.query(
        "MATCH (m:Memory)-[:REFLECTION_OF]->(t:Task) "
        "WHERE m.category = $category "
        "RETURN m.id, m.content, t.description",
        {"category": "reflection"},
    )

    if not rows:
        return []

    # Score each reflection by Levenshtein similarity to the query task.
    scored: list[tuple[float, dict[str, str]]] = []
    query_lower = task_description.lower().strip()

    for row in rows:
        task_desc = str(row.get("t.description", ""))
        content_raw = str(row.get("m.content", ""))

        # Compute similarity between query and the failed task description.
        similarity = Levenshtein.ratio(
            query_lower, task_desc.lower().strip()
        )

        if similarity < similarity_threshold:
            continue

        # Parse the reflection JSON content.
        try:
            content = json.loads(content_raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Skipping reflection %s: content is not valid JSON",
                row.get("m.id", "unknown"),
            )
            continue

        what_failed = content.get("what_failed", "")
        why = content.get("why", "")
        what_to_try = content.get("what_to_try", "")

        if not any([what_failed, why, what_to_try]):
            continue

        scored.append((similarity, {
            "task_description": task_desc,
            "what_failed": what_failed,
            "why": why,
            "what_to_try": what_to_try,
        }))

    # Sort by similarity descending, return top max_results.
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:max_results]]
