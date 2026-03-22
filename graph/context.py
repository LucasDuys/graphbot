"""Reflection retrieval for context assembly.

Queries the knowledge graph for Memory nodes with category="reflection" that are
linked (via REFLECTION_OF edges) to Task nodes with similar descriptions. Uses
Levenshtein similarity to rank relevance, returning structured reflection dicts
for injection into the decomposer prompt.
"""

from __future__ import annotations

import json
import logging

import Levenshtein

from graph.store import GraphStore

logger = logging.getLogger(__name__)

# Default similarity threshold: reflections below this Levenshtein ratio are
# excluded. 0.4 is intentionally permissive -- even loosely related past
# failures can be valuable context for decomposition.
_DEFAULT_SIMILARITY_THRESHOLD: float = 0.4

# Default maximum number of reflections to return.
_DEFAULT_MAX_RESULTS: int = 5


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
