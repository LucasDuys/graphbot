"""Kuzu graph database management for the GraphBot knowledge graph."""

from __future__ import annotations

import logging

import kuzu

from graph.schema import EDGE_TYPES, NODE_TYPES, get_create_edge_cypher, get_create_node_cypher

logger = logging.getLogger(__name__)


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
