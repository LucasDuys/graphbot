"""Tests for GraphStore -- Kuzu graph database management."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph.schema import EDGE_TYPES, NODE_TYPES
from graph.store import GraphStore


class TestGraphStoreInit:
    """GraphStore initialization and lifecycle."""

    def test_create_in_memory(self) -> None:
        """GraphStore with db_path=None creates an in-memory database."""
        store = GraphStore(db_path=None)
        assert store._db is not None
        assert store._conn is not None
        store.close()

    def test_create_on_disk(self, tmp_graph_dir: Path) -> None:
        """GraphStore with a path creates an on-disk database."""
        db_path = str(tmp_graph_dir / "test.db")
        store = GraphStore(db_path=db_path)
        assert store._db is not None
        assert store._conn is not None
        store.close()

    def test_close_releases_connection(self) -> None:
        """close() cleanly releases the Kuzu connection without error."""
        store = GraphStore()
        store.close()
        # Calling close again should not raise
        store.close()


class TestGraphStoreInitialize:
    """Schema creation via initialize()."""

    def test_creates_all_node_tables(self) -> None:
        """initialize() creates all 10 node tables from schema.py."""
        store = GraphStore()
        store.initialize()

        tables = _get_table_names(store)
        for node_type in NODE_TYPES:
            assert node_type.name in tables, f"Missing node table: {node_type.name}"

        store.close()

    def test_creates_all_edge_tables(self) -> None:
        """initialize() creates all 12 edge tables from schema.py."""
        store = GraphStore()
        store.initialize()

        tables = _get_table_names(store)
        for edge_type in EDGE_TYPES:
            assert edge_type.name in tables, f"Missing edge table: {edge_type.name}"

        store.close()

    def test_total_table_count(self) -> None:
        """initialize() creates exactly 10 node + 12 edge = 22 tables."""
        store = GraphStore()
        store.initialize()

        tables = _get_table_names(store)
        assert len(tables) == len(NODE_TYPES) + len(EDGE_TYPES)

        store.close()

    def test_idempotent(self) -> None:
        """Calling initialize() twice does not raise an error."""
        store = GraphStore()
        store.initialize()
        store.initialize()  # second call must not fail

        tables = _get_table_names(store)
        assert len(tables) == len(NODE_TYPES) + len(EDGE_TYPES)

        store.close()


def _get_table_names(store: GraphStore) -> set[str]:
    """Helper: query Kuzu for all table names."""
    result = store._conn.execute("CALL show_tables() RETURN *")
    names: set[str] = set()
    while result.has_next():
        row = result.get_next()
        # Row format: [id, name, type, database, comment]
        names.add(row[1])
    return names
