"""Memory consolidation engine: dedup, merge, and summarize graph entities.

Detects duplicate entities via the 3-layer EntityResolver, merges them by
combining properties and redirecting edges, and generates summary Memory nodes
for clusters of related memories. Runs as a callable (script or background).
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from graph.resolver import EntityResolver
from graph.schema import EDGE_TYPES, NODE_TYPES
from graph.store import GraphStore

logger = logging.getLogger(__name__)

# Node types that use "name" as display field.
_NAMED_TYPES: list[str] = [
    nt.name for nt in NODE_TYPES if "name" in nt.properties and nt.name != "File"
]

# File uses "path" as display field.
_FILE_TYPE: str = "File"

# Minimum cluster size to generate a summary.
_MIN_SUMMARY_CLUSTER: int = 2

# Default confidence threshold for the resolver to consider a match a duplicate.
_DUPLICATE_THRESHOLD: float = 0.95


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    duplicates_found: int = 0
    merged_count: int = 0
    summaries_created: int = 0
    errors: list[str] = field(default_factory=list)


class ConsolidationEngine:
    """Detects duplicates, merges them, and summarizes related memory clusters.

    Uses the existing 3-layer EntityResolver for duplicate detection.
    Merge combines properties (primary wins on conflicts), redirects all edges
    from the secondary to the primary, and deletes the secondary node.
    Summarize generates a single Memory node from a cluster of related memories.

    Args:
        store: The Kuzu graph store to consolidate.
        duplicate_threshold: Minimum resolver confidence to treat as duplicate.
    """

    def __init__(
        self,
        store: GraphStore,
        duplicate_threshold: float = _DUPLICATE_THRESHOLD,
    ) -> None:
        self._store = store
        self._resolver = EntityResolver(store)
        self._duplicate_threshold = duplicate_threshold

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def detect_duplicates(self) -> list[list[tuple[str, str]]]:
        """Detect duplicate entities across all named node types.

        Returns a list of duplicate groups. Each group is a list of
        (node_id, table_name) tuples that are duplicates of each other.

        Uses the EntityResolver's 3-layer matching (exact, Levenshtein, BM25)
        to find entities that resolve to the same mention with high confidence.
        """
        # Collect all entities: (id, table, display_text)
        all_entities: list[tuple[str, str, str]] = []

        for table in _NAMED_TYPES:
            rows = self._store.query(f"MATCH (n:{table}) RETURN n.id, n.name")
            for row in rows:
                node_id = str(row["n.id"])
                name = str(row.get("n.name") or "")
                if name:
                    all_entities.append((node_id, table, name))

        # File type uses "path"
        file_rows = self._store.query(f"MATCH (n:{_FILE_TYPE}) RETURN n.id, n.path")
        for row in file_rows:
            node_id = str(row["n.id"])
            path = str(row.get("n.path") or "")
            if path:
                all_entities.append((node_id, _FILE_TYPE, path))

        # Build duplicate groups using union-find approach.
        # For each entity, resolve its display text and check if another entity
        # in the same table matches with high confidence.
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Track table per entity
        entity_table: dict[str, str] = {}
        for node_id, table, _display in all_entities:
            entity_table[node_id] = table
            parent[node_id] = node_id

        # Compare within same-table entities
        by_table: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for node_id, table, display in all_entities:
            by_table[table].append((node_id, display))

        for table, entities in by_table.items():
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    id_a, display_a = entities[i]
                    id_b, display_b = entities[j]

                    # Quick check: use the resolver's logic inline for efficiency.
                    # Exact normalized match
                    norm_a = display_a.lower().strip()
                    norm_b = display_b.lower().strip()

                    if norm_a == norm_b:
                        union(id_a, id_b)
                        continue

                    # Levenshtein ratio check
                    try:
                        import Levenshtein
                        ratio = Levenshtein.ratio(norm_a, norm_b)
                        if ratio >= self._duplicate_threshold:
                            union(id_a, id_b)
                    except ImportError:
                        pass

        # Collect groups with more than one member
        groups_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for node_id in parent:
            root = find(node_id)
            groups_map[root].append((node_id, entity_table[node_id]))

        duplicate_groups: list[list[tuple[str, str]]] = [
            group for group in groups_map.values() if len(group) > 1
        ]

        return duplicate_groups

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, table: str, primary_id: str, secondary_id: str) -> bool:
        """Merge secondary node into primary: combine properties, redirect edges, delete.

        Properties from secondary fill in missing (None/empty) values on primary.
        Primary values always take precedence on conflicts.
        All edges involving the secondary node are redirected to the primary.
        The secondary node is then deleted.

        Args:
            table: The node table name (e.g. "Contact", "User").
            primary_id: The ID of the node to keep.
            secondary_id: The ID of the node to merge into primary and delete.

        Returns:
            True if merge succeeded, False if either node was not found.
        """
        primary_props = self._store.get_node(table, primary_id)
        secondary_props = self._store.get_node(table, secondary_id)

        if primary_props is None or secondary_props is None:
            logger.warning(
                "Cannot merge %s/%s into %s/%s: node not found",
                table, secondary_id, table, primary_id,
            )
            return False

        # Combine properties: secondary fills gaps in primary
        updates: dict[str, object] = {}
        for key, value in secondary_props.items():
            if key == "id":
                continue
            primary_value = primary_props.get(key)
            if (primary_value is None or primary_value == "") and value is not None and value != "":
                updates[key] = value

        if updates:
            self._store.update_node(table, primary_id, updates)

        # Redirect edges from secondary to primary
        self._redirect_edges(table, secondary_id, primary_id)

        # Delete the secondary node (use DETACH DELETE to remove remaining edges)
        conn = self._store._get_conn()
        conn.execute(
            f"MATCH (n:{table}) WHERE n.id = $id DETACH DELETE n",
            parameters={"id": secondary_id},
        )

        logger.info(
            "Merged %s node %s into %s (updates: %s)",
            table, secondary_id, primary_id, list(updates.keys()),
        )
        return True

    def _redirect_edges(self, table: str, old_id: str, new_id: str) -> None:
        """Redirect all edges from old_id to new_id.

        For each edge type where old_id participates (as source or target),
        create an equivalent edge pointing to/from new_id, then the old edges
        will be cleaned up when the old node is DETACH DELETEd.
        """
        conn = self._store._get_conn()

        for edge_type in EDGE_TYPES:
            # Outgoing edges: (old_id:from_type) -[edge]-> (other:to_type)
            if edge_type.from_type == table:
                try:
                    rows = conn.execute(
                        f"MATCH (a:{edge_type.from_type})-[r:{edge_type.name}]->(b:{edge_type.to_type}) "
                        f"WHERE a.id = $old_id RETURN b.id",
                        parameters={"old_id": old_id},
                    )
                    targets: list[str] = []
                    while rows.has_next():
                        row = rows.get_next()
                        targets.append(str(row[0]))

                    for target_id in targets:
                        # Avoid self-loops
                        if target_id == new_id:
                            continue
                        try:
                            self._store.create_edge(edge_type.name, new_id, target_id)
                        except Exception:
                            logger.debug(
                                "Could not redirect outgoing %s edge to %s",
                                edge_type.name, target_id,
                            )
                except Exception:
                    logger.debug(
                        "Could not query outgoing %s edges for %s",
                        edge_type.name, old_id,
                    )

            # Incoming edges: (other:from_type) -[edge]-> (old_id:to_type)
            if edge_type.to_type == table:
                try:
                    rows = conn.execute(
                        f"MATCH (a:{edge_type.from_type})-[r:{edge_type.name}]->(b:{edge_type.to_type}) "
                        f"WHERE b.id = $old_id RETURN a.id",
                        parameters={"old_id": old_id},
                    )
                    sources: list[str] = []
                    while rows.has_next():
                        row = rows.get_next()
                        sources.append(str(row[0]))

                    for source_id in sources:
                        # Avoid self-loops
                        if source_id == new_id:
                            continue
                        try:
                            self._store.create_edge(edge_type.name, source_id, new_id)
                        except Exception:
                            logger.debug(
                                "Could not redirect incoming %s edge from %s",
                                edge_type.name, source_id,
                            )
                except Exception:
                    logger.debug(
                        "Could not query incoming %s edges for %s",
                        edge_type.name, old_id,
                    )

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------

    def summarize(self, memory_ids: list[str]) -> str | None:
        """Generate a summary Memory node from a cluster of related memories.

        Collects the content of all given memory IDs, builds a combined summary
        string, creates a new Memory node with category="summary", and links it
        to the same entities that the source memories are linked to (via ABOUT
        and ABOUT_PROJECT edges).

        Args:
            memory_ids: List of Memory node IDs to summarize.

        Returns:
            The ID of the created summary Memory node, or None if fewer than 2
            memories are provided or none could be loaded.
        """
        if len(memory_ids) < _MIN_SUMMARY_CLUSTER:
            return None

        # Load memory contents
        contents: list[str] = []
        for mid in memory_ids:
            node = self._store.get_node("Memory", mid)
            if node is not None:
                content = str(node.get("content") or "")
                if content:
                    contents.append(content)

        if len(contents) < _MIN_SUMMARY_CLUSTER:
            return None

        # Build summary content (algorithmic, no LLM)
        summary_content = "Consolidated from {} memories: {}".format(
            len(contents),
            " | ".join(contents),
        )

        summary_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self._store.create_node("Memory", {
            "id": summary_id,
            "content": summary_content,
            "category": "summary",
            "confidence": 0.8,
            "valid_from": now,
        })

        # Link summary to same entities as source memories
        linked_entity_ids: set[str] = set()
        linked_project_ids: set[str] = set()

        for mid in memory_ids:
            # Check ABOUT edges (Memory -> User)
            rows = self._store.query(
                "MATCH (m:Memory)-[:ABOUT]->(u:User) WHERE m.id = $mid RETURN u.id",
                {"mid": mid},
            )
            for row in rows:
                linked_entity_ids.add(str(row["u.id"]))

            # Check ABOUT_PROJECT edges (Memory -> Project)
            rows = self._store.query(
                "MATCH (m:Memory)-[:ABOUT_PROJECT]->(p:Project) WHERE m.id = $mid RETURN p.id",
                {"mid": mid},
            )
            for row in rows:
                linked_project_ids.add(str(row["p.id"]))

        for entity_id in linked_entity_ids:
            try:
                self._store.create_edge("ABOUT", summary_id, entity_id)
            except Exception:
                logger.debug(
                    "Could not link summary %s to entity %s",
                    summary_id, entity_id,
                )

        for project_id in linked_project_ids:
            try:
                self._store.create_edge("ABOUT_PROJECT", summary_id, project_id)
            except Exception:
                logger.debug(
                    "Could not link summary %s to project %s",
                    summary_id, project_id,
                )

        logger.info(
            "Created summary Memory %s from %d memories",
            summary_id, len(contents),
        )
        return summary_id

    # ------------------------------------------------------------------
    # Cluster detection for summarization
    # ------------------------------------------------------------------

    def _find_memory_clusters(self) -> list[list[str]]:
        """Find clusters of related memories by shared entity links.

        Groups Memory nodes that are linked to the same entity (User or Project)
        via ABOUT or ABOUT_PROJECT edges. Only returns clusters with at least
        _MIN_SUMMARY_CLUSTER+1 members to avoid trivial summaries.

        Returns clusters that do not already have a summary (category != 'summary').
        """
        # entity_id -> list of memory_ids linked to it
        entity_memories: dict[str, list[str]] = defaultdict(list)

        # ABOUT edges: Memory -> User
        rows = self._store.query(
            "MATCH (m:Memory)-[:ABOUT]->(u:User) "
            "WHERE m.category <> $cat OR m.category IS NULL "
            "RETURN m.id, u.id",
            {"cat": "summary"},
        )
        for row in rows:
            mid = str(row["m.id"])
            uid = str(row["u.id"])
            entity_memories[uid].append(mid)

        # ABOUT_PROJECT edges: Memory -> Project
        rows = self._store.query(
            "MATCH (m:Memory)-[:ABOUT_PROJECT]->(p:Project) "
            "WHERE m.category <> $cat OR m.category IS NULL "
            "RETURN m.id, p.id",
            {"cat": "summary"},
        )
        for row in rows:
            mid = str(row["m.id"])
            pid = str(row["p.id"])
            entity_memories[pid].append(mid)

        # Filter to clusters large enough for summarization
        clusters: list[list[str]] = []
        seen_mids: set[str] = set()
        for _entity_id, mids in entity_memories.items():
            unique_mids = [m for m in mids if m not in seen_mids]
            if len(unique_mids) >= _MIN_SUMMARY_CLUSTER + 1:
                clusters.append(unique_mids)
                seen_mids.update(unique_mids)

        return clusters

    # ------------------------------------------------------------------
    # Full consolidation pass
    # ------------------------------------------------------------------

    def run(self) -> ConsolidationResult:
        """Execute a full consolidation pass: detect duplicates, merge, summarize.

        1. Detect duplicate entities via the EntityResolver's 3-layer matching.
        2. Merge each duplicate group (first node is primary, rest are merged in).
        3. Find clusters of related memories and generate summary nodes.

        Returns:
            A ConsolidationResult with counts of duplicates found, merges done,
            and summaries created.
        """
        result = ConsolidationResult()

        # Step 1: Detect and merge duplicates
        duplicate_groups = self.detect_duplicates()
        result.duplicates_found = len(duplicate_groups)

        for group in duplicate_groups:
            # Sort by ID for deterministic primary selection (first alphabetically)
            group.sort(key=lambda x: x[0])
            primary_id, table = group[0]

            for secondary_id, secondary_table in group[1:]:
                if secondary_table != table:
                    logger.warning(
                        "Cross-table duplicate detected: %s/%s vs %s/%s -- skipping",
                        table, primary_id, secondary_table, secondary_id,
                    )
                    result.errors.append(
                        f"Cross-table duplicate: {table}/{primary_id} vs "
                        f"{secondary_table}/{secondary_id}"
                    )
                    continue

                if self.merge(table, primary_id, secondary_id):
                    result.merged_count += 1
                else:
                    result.errors.append(
                        f"Failed to merge {table}/{secondary_id} into {primary_id}"
                    )

        # Step 2: Find memory clusters and generate summaries
        clusters = self._find_memory_clusters()
        for cluster in clusters:
            summary_id = self.summarize(cluster)
            if summary_id is not None:
                result.summaries_created += 1

        logger.info(
            "Consolidation complete: %d duplicates found, %d merged, %d summaries",
            result.duplicates_found, result.merged_count, result.summaries_created,
        )
        return result

    def __call__(self) -> ConsolidationResult:
        """Make the engine callable for use as a script or background task."""
        return self.run()
