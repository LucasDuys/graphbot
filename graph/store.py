"""Kuzu graph database management for the GraphBot knowledge graph."""

from __future__ import annotations

import logging
import uuid

import kuzu

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
