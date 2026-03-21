"""Tests for GraphStore CRUD operations."""

from __future__ import annotations

import uuid

import pytest

from graph.store import GraphStore


@pytest.fixture
def store() -> GraphStore:
    """Provide an initialized in-memory GraphStore."""
    s = GraphStore(db_path=None)
    s.initialize()
    yield s  # type: ignore[misc]
    s.close()


class TestCreateNode:
    """Tests for create_node()."""

    def test_create_node_returns_id(self, store: GraphStore) -> None:
        """create_node returns a string ID."""
        node_id = store.create_node("User", {"name": "Alice", "role": "student"})
        assert isinstance(node_id, str)
        assert len(node_id) > 0

    def test_create_node_auto_generates_uuid(self, store: GraphStore) -> None:
        """create_node generates a UUID when id is not provided."""
        node_id = store.create_node("User", {"name": "Bob"})
        # Should be a valid UUID
        uuid.UUID(node_id)

    def test_create_node_uses_provided_id(self, store: GraphStore) -> None:
        """create_node uses the id from properties if provided."""
        node_id = store.create_node("User", {"id": "custom-id-123", "name": "Carol"})
        assert node_id == "custom-id-123"

    def test_create_node_entity_type(self, store: GraphStore) -> None:
        """create_node works with Memory table type."""
        node_id = store.create_node("Memory", {
            "content": "User likes Python",
            "category": "preference",
        })
        assert isinstance(node_id, str)

    def test_create_node_task_type(self, store: GraphStore) -> None:
        """create_node works with Task table type."""
        node_id = store.create_node("Task", {
            "description": "Compute something",
            "domain": "CODE",
            "status": "READY",
        })
        assert isinstance(node_id, str)


class TestGetNode:
    """Tests for get_node()."""

    def test_get_existing_node(self, store: GraphStore) -> None:
        """get_node returns properties for an existing node."""
        node_id = store.create_node("User", {"name": "Alice", "role": "student"})
        result = store.get_node("User", node_id)
        assert result is not None
        assert result["name"] == "Alice"
        assert result["role"] == "student"
        assert result["id"] == node_id

    def test_get_nonexistent_node_returns_none(self, store: GraphStore) -> None:
        """get_node returns None for a non-existent node."""
        result = store.get_node("User", "does-not-exist")
        assert result is None

    def test_get_node_all_properties(self, store: GraphStore) -> None:
        """get_node returns all stored properties."""
        props = {"name": "TestProject", "path": "/dev/test", "language": "Python"}
        node_id = store.create_node("Project", props)
        result = store.get_node("Project", node_id)
        assert result is not None
        assert result["name"] == "TestProject"
        assert result["path"] == "/dev/test"
        assert result["language"] == "Python"


class TestUpdateNode:
    """Tests for update_node()."""

    def test_update_existing_node(self, store: GraphStore) -> None:
        """update_node modifies properties and returns True."""
        node_id = store.create_node("User", {"name": "Alice", "role": "student"})
        updated = store.update_node("User", node_id, {"role": "engineer"})
        assert updated is True

        result = store.get_node("User", node_id)
        assert result is not None
        assert result["role"] == "engineer"
        assert result["name"] == "Alice"  # unchanged

    def test_update_nonexistent_node(self, store: GraphStore) -> None:
        """update_node returns False for a non-existent node."""
        updated = store.update_node("User", "does-not-exist", {"name": "Ghost"})
        assert updated is False

    def test_update_multiple_properties(self, store: GraphStore) -> None:
        """update_node can change multiple properties at once."""
        node_id = store.create_node("Project", {"name": "Old", "language": "Java"})
        store.update_node("Project", node_id, {"name": "New", "language": "Python"})

        result = store.get_node("Project", node_id)
        assert result is not None
        assert result["name"] == "New"
        assert result["language"] == "Python"


class TestDeleteNode:
    """Tests for delete_node()."""

    def test_delete_existing_node(self, store: GraphStore) -> None:
        """delete_node removes the node and returns True."""
        node_id = store.create_node("User", {"name": "Alice"})
        deleted = store.delete_node("User", node_id)
        assert deleted is True

        result = store.get_node("User", node_id)
        assert result is None

    def test_delete_nonexistent_node(self, store: GraphStore) -> None:
        """delete_node returns False for a non-existent node."""
        deleted = store.delete_node("User", "does-not-exist")
        assert deleted is False


class TestCreateEdge:
    """Tests for create_edge()."""

    def test_create_edge_between_nodes(self, store: GraphStore) -> None:
        """create_edge links two nodes and returns True."""
        user_id = store.create_node("User", {"name": "Alice"})
        project_id = store.create_node("Project", {"name": "GraphBot"})

        success = store.create_edge("OWNS", user_id, project_id)
        assert success is True

    def test_create_edge_with_properties(self, store: GraphStore) -> None:
        """create_edge can attach properties to the relationship."""
        t1 = store.create_node("Task", {"description": "Task A"})
        t2 = store.create_node("Task", {"description": "Task B"})

        success = store.create_edge("DEPENDS_ON", t1, t2, {"data_key": "result_a"})
        assert success is True

    def test_create_edge_verifiable_via_query(self, store: GraphStore) -> None:
        """Created edge is verifiable via a raw Cypher query."""
        user_id = store.create_node("User", {"name": "Alice"})
        svc_id = store.create_node("Service", {"name": "GitHub", "type": "VCS"})

        store.create_edge("USES", user_id, svc_id)

        rows = store.query(
            "MATCH (u:User)-[:USES]->(s:Service) "
            "WHERE u.id = $uid RETURN s.name",
            {"uid": user_id},
        )
        assert len(rows) == 1
        assert rows[0]["s.name"] == "GitHub"


class TestQuery:
    """Tests for query()."""

    def test_query_returns_list_of_dicts(self, store: GraphStore) -> None:
        """query() returns results as list of dicts."""
        store.create_node("User", {"id": "u1", "name": "Alice"})
        store.create_node("User", {"id": "u2", "name": "Bob"})

        rows = store.query("MATCH (u:User) RETURN u.name ORDER BY u.name")
        assert isinstance(rows, list)
        assert len(rows) == 2
        assert rows[0]["u.name"] == "Alice"
        assert rows[1]["u.name"] == "Bob"

    def test_query_with_params(self, store: GraphStore) -> None:
        """query() supports parameterized queries."""
        store.create_node("User", {"id": "u1", "name": "Alice"})

        rows = store.query(
            "MATCH (u:User) WHERE u.id = $id RETURN u.name",
            {"id": "u1"},
        )
        assert len(rows) == 1
        assert rows[0]["u.name"] == "Alice"

    def test_query_empty_result(self, store: GraphStore) -> None:
        """query() returns empty list when no matches."""
        rows = store.query("MATCH (u:User) RETURN u.name")
        assert rows == []


class TestCrudCycle:
    """Full CRUD cycle tests across multiple node types."""

    @pytest.mark.parametrize("table,props", [
        ("User", {"name": "Test User", "role": "tester"}),
        ("Memory", {"content": "A fact", "category": "general"}),
        ("Task", {"description": "Do thing", "domain": "CODE", "status": "READY"}),
    ])
    def test_full_crud_cycle(
        self, store: GraphStore, table: str, props: dict[str, str]
    ) -> None:
        """Create -> Read -> Update -> Delete cycle works for each type."""
        # Create
        node_id = store.create_node(table, props)
        assert isinstance(node_id, str)

        # Read
        node = store.get_node(table, node_id)
        assert node is not None
        for key, val in props.items():
            assert node[key] == val

        # Update
        first_key = next(iter(props))
        updated = store.update_node(table, node_id, {first_key: "UPDATED"})
        assert updated is True
        node = store.get_node(table, node_id)
        assert node is not None
        assert node[first_key] == "UPDATED"

        # Delete
        deleted = store.delete_node(table, node_id)
        assert deleted is True
        assert store.get_node(table, node_id) is None
