"""Kuzu graph database management for the GraphBot knowledge graph."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import kuzu

from core_gb.types import GraphContext
from graph.schema import EDGE_TYPES, NODE_TYPES, get_create_edge_cypher, get_create_node_cypher

logger = logging.getLogger(__name__)

# Build a lookup from edge name to EdgeType for create_edge().
_EDGE_TYPE_MAP: dict[str, tuple[str, str, dict[str, str]]] = {
    et.name: (et.from_type, et.to_type, et.properties) for et in EDGE_TYPES
}


class GraphStore:
    """Manages a Kuzu graph database: connection lifecycle and schema creation.

    Args:
        db_path: Filesystem path for on-disk storage. None creates an in-memory database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            self._db = kuzu.Database()
        else:
            self._db = kuzu.Database(db_path)
        self._conn: kuzu.Connection | None = kuzu.Connection(self._db)

    def initialize(self) -> None:
        """Create all node and edge tables defined in the schema.

        Uses IF NOT EXISTS so this method is idempotent -- safe to call multiple times.
        """
        conn = self._get_conn()

        for node_type in NODE_TYPES:
            cypher = get_create_node_cypher(node_type)
            logger.debug("Creating node table: %s", node_type.name)
            conn.execute(cypher)

        for edge_type in EDGE_TYPES:
            cypher = get_create_edge_cypher(edge_type)
            logger.debug("Creating edge table: %s", edge_type.name)
            conn.execute(cypher)

        logger.info(
            "Schema initialized: %d node tables, %d edge tables",
            len(NODE_TYPES),
            len(EDGE_TYPES),
        )

    def close(self) -> None:
        """Release the Kuzu connection. Safe to call multiple times."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _get_conn(self) -> kuzu.Connection:
        """Return the active connection or raise if closed."""
        if self._conn is None:
            raise RuntimeError("GraphStore connection is closed")
        return self._conn

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_node(self, table: str, properties: dict[str, object]) -> str:
        """Create a node in the given table and return its id.

        If ``properties`` does not contain an ``id`` key, a UUID4 is generated
        automatically.  All property values are passed as Kuzu parameters (no
        string interpolation of user data).
        """
        props = dict(properties)
        if "id" not in props:
            props["id"] = str(uuid.uuid4())

        node_id: str = str(props["id"])
        param_assignments = ", ".join(f"{k}: ${k}" for k in props)
        cypher = f"CREATE (n:{table} {{{param_assignments}}})"

        conn = self._get_conn()
        conn.execute(cypher, parameters=props)
        return node_id

    def get_node(self, table: str, node_id: str) -> dict[str, object] | None:
        """Return properties of a node as a dict, or None if not found."""
        cypher = f"MATCH (n:{table}) WHERE n.id = $id RETURN n.*"
        conn = self._get_conn()
        result = conn.execute(cypher, parameters={"id": node_id})

        if not result.has_next():
            return None

        row = result.get_next()
        columns = result.get_column_names()
        node: dict[str, object] = {}
        for col_name, value in zip(columns, row):
            # Column names come back as "n.prop"; strip the "n." prefix.
            key = col_name.split(".", 1)[1] if "." in col_name else col_name
            node[key] = value
        return node

    def update_node(self, table: str, node_id: str, properties: dict[str, object]) -> bool:
        """Update properties on an existing node. Returns True if node was found."""
        # First check existence
        if self.get_node(table, node_id) is None:
            return False

        set_clauses = ", ".join(f"n.{k} = $set_{k}" for k in properties)
        cypher = f"MATCH (n:{table}) WHERE n.id = $id SET {set_clauses}"

        params: dict[str, object] = {"id": node_id}
        for k, v in properties.items():
            params[f"set_{k}"] = v

        conn = self._get_conn()
        conn.execute(cypher, parameters=params)
        return True

    def delete_node(self, table: str, node_id: str) -> bool:
        """Delete a node by id. Returns True if node existed."""
        if self.get_node(table, node_id) is None:
            return False

        cypher = f"MATCH (n:{table}) WHERE n.id = $id DELETE n"
        conn = self._get_conn()
        conn.execute(cypher, parameters={"id": node_id})
        return True

    def create_edge(
        self,
        table: str,
        from_id: str,
        to_id: str,
        properties: dict[str, object] | None = None,
    ) -> bool:
        """Create a relationship between two nodes.

        The ``from_type`` and ``to_type`` are resolved from ``EDGE_TYPES`` in
        ``graph.schema``.  Returns True on success.
        """
        if table not in _EDGE_TYPE_MAP:
            raise ValueError(f"Unknown edge type: {table}")

        from_type, to_type, _ = _EDGE_TYPE_MAP[table]
        props = properties or {}

        params: dict[str, object] = {"from_id": from_id, "to_id": to_id}

        prop_fragment = ""
        if props:
            prop_assignments = ", ".join(f"{k}: $edge_{k}" for k in props)
            prop_fragment = f" {{{prop_assignments}}}"
            for k, v in props.items():
                params[f"edge_{k}"] = v

        cypher = (
            f"MATCH (a:{from_type}), (b:{to_type}) "
            f"WHERE a.id = $from_id AND b.id = $to_id "
            f"CREATE (a)-[:{table}{prop_fragment}]->(b)"
        )

        conn = self._get_conn()
        conn.execute(cypher, parameters=params)
        return True

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    def _find_node_table(self, node_id: str) -> tuple[str, dict[str, object]] | None:
        """Search all node tables for a node with the given id.

        Returns (table_name, properties) or None.
        """
        conn = self._get_conn()
        for node_type in NODE_TYPES:
            cypher = f"MATCH (n:{node_type.name}) WHERE n.id = $id RETURN n.*"
            result = conn.execute(cypher, parameters={"id": node_id})
            if result.has_next():
                row = result.get_next()
                columns = result.get_column_names()
                props: dict[str, object] = {}
                for col_name, value in zip(columns, row):
                    key = col_name.split(".", 1)[1] if "." in col_name else col_name
                    props[key] = value
                return (node_type.name, props)
        return None

    def _get_connected_1hop(self, node_id: str, table_name: str) -> list[tuple[str, dict[str, object]]]:
        """Get all nodes connected to the given node by 1 hop (any direction)."""
        conn = self._get_conn()
        results: list[tuple[str, dict[str, object]]] = []

        for edge_type in EDGE_TYPES:
            # Outgoing: start is from_type
            if edge_type.from_type == table_name:
                cypher = (
                    f"MATCH (a:{edge_type.from_type})-[:{edge_type.name}]->(b:{edge_type.to_type}) "
                    f"WHERE a.id = $id RETURN b.*"
                )
                try:
                    result = conn.execute(cypher, parameters={"id": node_id})
                    while result.has_next():
                        row = result.get_next()
                        columns = result.get_column_names()
                        props: dict[str, object] = {}
                        for col_name, value in zip(columns, row):
                            key = col_name.split(".", 1)[1] if "." in col_name else col_name
                            props[key] = value
                        results.append((edge_type.to_type, props))
                except Exception:
                    pass

            # Incoming: start is to_type
            if edge_type.to_type == table_name:
                cypher = (
                    f"MATCH (a:{edge_type.from_type})-[:{edge_type.name}]->(b:{edge_type.to_type}) "
                    f"WHERE b.id = $id RETURN a.*"
                )
                try:
                    result = conn.execute(cypher, parameters={"id": node_id})
                    while result.has_next():
                        row = result.get_next()
                        columns = result.get_column_names()
                        props: dict[str, object] = {}
                        for col_name, value in zip(columns, row):
                            key = col_name.split(".", 1)[1] if "." in col_name else col_name
                            props[key] = value
                        results.append((edge_type.from_type, props))
                except Exception:
                    pass

        return results

    def get_context(self, entity_ids: list[str], max_tokens: int = 2500) -> GraphContext:
        """Assemble context from the knowledge graph for given entities.

        Performs 2-hop traversal from given entities, collects:
        - Entity properties (name, type, details)
        - Connected memories (active ones where valid_until IS NULL)
        - Related entities via edges

        Respects max_tokens budget using heuristic: tokens ~= word_count * 1.3
        Truncates least-relevant results when over budget.
        """
        if not entity_ids:
            return GraphContext()

        user_summary = ""
        entities: list[dict[str, str]] = []
        memories: list[str] = []
        seen_ids: set[str] = set()

        # Collect all nodes within 2 hops
        hop1_nodes: list[tuple[str, dict[str, object], int]] = []  # (table, props, hop_distance)

        for eid in entity_ids:
            found = self._find_node_table(eid)
            if found is None:
                continue
            table_name, props = found
            node_id = str(props.get("id", eid))
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            hop1_nodes.append((table_name, props, 0))

            # 1-hop neighbors
            for neighbor_table, neighbor_props in self._get_connected_1hop(node_id, table_name):
                nid = str(neighbor_props.get("id", ""))
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    hop1_nodes.append((neighbor_table, neighbor_props, 1))

                    # 2-hop neighbors
                    for hop2_table, hop2_props in self._get_connected_1hop(nid, neighbor_table):
                        h2id = str(hop2_props.get("id", ""))
                        if h2id and h2id not in seen_ids:
                            seen_ids.add(h2id)
                            hop1_nodes.append((hop2_table, hop2_props, 2))

        # Separate into entities and memories, build user_summary
        entity_candidates: list[tuple[dict[str, str], int]] = []  # (entity_dict, hop_distance)
        memory_candidates: list[tuple[str, int]] = []  # (content, hop_distance)

        now = datetime.now(timezone.utc)

        for table_name, props, hop in hop1_nodes:
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
                    memory_candidates.append((content, hop))
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
                entity_candidates.append((entity_dict, hop))
            else:
                name = str(props.get("name", props.get("id", "")))
                detail_parts: list[str] = []
                for key in ("path", "language", "framework", "status", "type", "url",
                            "relationship", "platform", "description"):
                    val = props.get(key)
                    if val:
                        detail_parts.append(f"{key}={val}")
                entity_dict = {
                    "type": table_name,
                    "name": name,
                    "details": ", ".join(detail_parts),
                }
                entity_candidates.append((entity_dict, hop))

        # Sort: closer hops first. Within same hop, entities before memories.
        entity_candidates.sort(key=lambda x: x[1])
        memory_candidates.sort(key=lambda x: x[1])

        # Apply token budget: entities first, then memories.
        # Truncate least-relevant (highest hop distance) first.
        token_budget = max_tokens
        used_tokens = self._estimate_tokens(user_summary) if user_summary else 0

        final_entities: list[dict[str, str]] = []
        for entity_dict, _hop in entity_candidates:
            text = f"{entity_dict.get('type', '')} {entity_dict.get('name', '')} {entity_dict.get('details', '')}"
            cost = self._estimate_tokens(text)
            if used_tokens + cost > token_budget:
                break
            used_tokens += cost
            final_entities.append(entity_dict)

        final_memories: list[str] = []
        for content, _hop in memory_candidates:
            cost = self._estimate_tokens(content)
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

    def query(self, cypher: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
        """Execute raw Cypher and return results as a list of row dicts."""
        conn = self._get_conn()
        if params:
            result = conn.execute(cypher, parameters=params)
        else:
            result = conn.execute(cypher)

        columns = result.get_column_names()
        rows: list[dict[str, object]] = []
        while result.has_next():
            row = result.get_next()
            rows.append(dict(zip(columns, row)))
        return rows
