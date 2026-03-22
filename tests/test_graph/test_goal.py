"""Tests for Goal node type: schema, CRUD operations, and DECOMPOSES_TO edge."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.schema import EDGE_TYPES, NODE_TYPES, GoalStatus
from graph.store import GraphStore


@pytest.fixture
def store() -> GraphStore:
    """Provide an initialized in-memory GraphStore."""
    s = GraphStore(db_path=None)
    s.initialize()
    yield s  # type: ignore[misc]
    s.close()


class TestGoalSchema:
    """Verify Goal node type is registered in the schema."""

    def test_goal_node_type_exists(self) -> None:
        """Goal must be present in NODE_TYPES."""
        goal_types = [nt for nt in NODE_TYPES if nt.name == "Goal"]
        assert len(goal_types) == 1, "Expected exactly one Goal node type"

    def test_goal_has_required_properties(self) -> None:
        """Goal schema must include all required properties."""
        goal_type = next(nt for nt in NODE_TYPES if nt.name == "Goal")
        required = {
            "id": "STRING",
            "description": "STRING",
            "status": "STRING",
            "priority": "INT64",
            "deadline": "TIMESTAMP",
            "progress": "DOUBLE",
            "created_at": "TIMESTAMP",
        }
        for prop_name, prop_type in required.items():
            assert prop_name in goal_type.properties, (
                f"Missing property: {prop_name}"
            )
            assert goal_type.properties[prop_name] == prop_type, (
                f"Property {prop_name} should be {prop_type}, "
                f"got {goal_type.properties[prop_name]}"
            )

    def test_goal_has_activation_properties(self) -> None:
        """Goal schema must include standard activation tracking fields."""
        goal_type = next(nt for nt in NODE_TYPES if nt.name == "Goal")
        assert goal_type.properties.get("access_count") == "INT64"
        assert goal_type.properties.get("last_accessed") == "TIMESTAMP"

    def test_decomposes_to_edge_exists(self) -> None:
        """DECOMPOSES_TO edge must be present in EDGE_TYPES."""
        decomp_edges = [et for et in EDGE_TYPES if et.name == "DECOMPOSES_TO"]
        assert len(decomp_edges) == 1, "Expected exactly one DECOMPOSES_TO edge type"

    def test_decomposes_to_edge_connects_goal_to_task(self) -> None:
        """DECOMPOSES_TO must go from Goal to Task."""
        decomp_edge = next(et for et in EDGE_TYPES if et.name == "DECOMPOSES_TO")
        assert decomp_edge.from_type == "Goal"
        assert decomp_edge.to_type == "Task"


class TestGoalStatusEnum:
    """Verify GoalStatus enum values."""

    def test_goal_status_values(self) -> None:
        """GoalStatus must have exactly: active, paused, completed, failed."""
        expected = {"active", "paused", "completed", "failed"}
        actual = {s.value for s in GoalStatus}
        assert actual == expected

    def test_goal_status_is_str_enum(self) -> None:
        """GoalStatus members must be usable as strings."""
        assert str(GoalStatus.ACTIVE) == "GoalStatus.active" or GoalStatus.ACTIVE == "active"
        assert GoalStatus.ACTIVE.value == "active"


class TestGoalCreate:
    """Tests for creating Goal nodes."""

    def test_create_goal_returns_id(self, store: GraphStore) -> None:
        """Creating a Goal node returns a string ID."""
        goal_id = store.create_node("Goal", {
            "description": "Ship MVP by end of month",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        assert isinstance(goal_id, str)
        assert len(goal_id) > 0

    def test_create_goal_with_all_properties(self, store: GraphStore) -> None:
        """Creating a Goal with all properties stores them correctly."""
        now = datetime.now(timezone.utc)
        goal_id = store.create_node("Goal", {
            "description": "Complete Phase 2",
            "status": GoalStatus.ACTIVE.value,
            "priority": 2,
            "deadline": now,
            "progress": 0.25,
            "created_at": now,
        })
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["description"] == "Complete Phase 2"
        assert node["status"] == "active"
        assert node["priority"] == 2
        assert node["progress"] == 0.25

    def test_create_goal_with_custom_id(self, store: GraphStore) -> None:
        """Creating a Goal with a custom ID uses that ID."""
        goal_id = store.create_node("Goal", {
            "id": "goal-custom-001",
            "description": "Custom ID goal",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        assert goal_id == "goal-custom-001"


class TestGoalRead:
    """Tests for reading Goal nodes."""

    def test_get_existing_goal(self, store: GraphStore) -> None:
        """get_node returns the Goal with correct properties."""
        goal_id = store.create_node("Goal", {
            "description": "Read me back",
            "status": GoalStatus.PAUSED.value,
            "priority": 3,
            "progress": 0.5,
        })
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["id"] == goal_id
        assert node["description"] == "Read me back"
        assert node["status"] == "paused"
        assert node["priority"] == 3
        assert node["progress"] == 0.5

    def test_get_nonexistent_goal(self, store: GraphStore) -> None:
        """get_node returns None for a non-existent Goal."""
        result = store.get_node("Goal", "no-such-goal")
        assert result is None


class TestGoalUpdate:
    """Tests for updating Goal nodes."""

    def test_update_goal_status(self, store: GraphStore) -> None:
        """Updating a Goal's status persists the change."""
        goal_id = store.create_node("Goal", {
            "description": "Status transition test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        updated = store.update_node("Goal", goal_id, {
            "status": GoalStatus.COMPLETED.value,
        })
        assert updated is True

        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["status"] == "completed"

    def test_update_goal_progress(self, store: GraphStore) -> None:
        """Updating a Goal's progress persists the change."""
        goal_id = store.create_node("Goal", {
            "description": "Progress test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        updated = store.update_node("Goal", goal_id, {"progress": 0.75})
        assert updated is True

        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["progress"] == 0.75

    def test_update_nonexistent_goal(self, store: GraphStore) -> None:
        """Updating a non-existent Goal returns False."""
        result = store.update_node("Goal", "ghost-goal", {"status": "failed"})
        assert result is False

    def test_update_multiple_goal_properties(self, store: GraphStore) -> None:
        """Updating multiple properties at once works correctly."""
        goal_id = store.create_node("Goal", {
            "description": "Multi-update test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 3,
            "progress": 0.1,
        })
        store.update_node("Goal", goal_id, {
            "status": GoalStatus.PAUSED.value,
            "priority": 1,
            "progress": 0.5,
        })
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["status"] == "paused"
        assert node["priority"] == 1
        assert node["progress"] == 0.5


class TestGoalDelete:
    """Tests for deleting Goal nodes."""

    def test_delete_existing_goal(self, store: GraphStore) -> None:
        """Deleting an existing Goal removes it and returns True."""
        goal_id = store.create_node("Goal", {
            "description": "Delete me",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        deleted = store.delete_node("Goal", goal_id)
        assert deleted is True
        assert store.get_node("Goal", goal_id) is None

    def test_delete_nonexistent_goal(self, store: GraphStore) -> None:
        """Deleting a non-existent Goal returns False."""
        deleted = store.delete_node("Goal", "no-such-goal")
        assert deleted is False


class TestGoalQueryActiveGoals:
    """Tests for querying active goals via raw Cypher."""

    def test_query_active_goals(self, store: GraphStore) -> None:
        """Querying for active goals returns only goals with active status."""
        store.create_node("Goal", {
            "id": "g1",
            "description": "Active goal 1",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        store.create_node("Goal", {
            "id": "g2",
            "description": "Completed goal",
            "status": GoalStatus.COMPLETED.value,
            "priority": 2,
            "progress": 1.0,
        })
        store.create_node("Goal", {
            "id": "g3",
            "description": "Active goal 2",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.3,
        })

        rows = store.query(
            "MATCH (g:Goal) WHERE g.status = $status RETURN g.id, g.description "
            "ORDER BY g.id",
            {"status": "active"},
        )
        assert len(rows) == 2
        assert rows[0]["g.id"] == "g1"
        assert rows[1]["g.id"] == "g3"

    def test_query_active_goals_empty(self, store: GraphStore) -> None:
        """Querying active goals when none exist returns empty list."""
        store.create_node("Goal", {
            "id": "g1",
            "description": "Paused goal",
            "status": GoalStatus.PAUSED.value,
            "priority": 1,
            "progress": 0.0,
        })
        rows = store.query(
            "MATCH (g:Goal) WHERE g.status = $status RETURN g.id",
            {"status": "active"},
        )
        assert rows == []


class TestGoalDecomposesToEdge:
    """Tests for DECOMPOSES_TO edge between Goal and Task."""

    def test_create_decomposes_to_edge(self, store: GraphStore) -> None:
        """DECOMPOSES_TO edge links a Goal to a Task."""
        goal_id = store.create_node("Goal", {
            "description": "Ship v1.0",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        task_id = store.create_node("Task", {
            "description": "Write tests",
            "domain": "CODE",
            "status": "READY",
        })

        success = store.create_edge("DECOMPOSES_TO", goal_id, task_id)
        assert success is True

    def test_decomposes_to_edge_queryable(self, store: GraphStore) -> None:
        """DECOMPOSES_TO edge is queryable via Cypher."""
        goal_id = store.create_node("Goal", {
            "id": "goal-main",
            "description": "Main goal",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        t1_id = store.create_node("Task", {
            "id": "task-a",
            "description": "Sub-task A",
            "domain": "CODE",
            "status": "READY",
        })
        t2_id = store.create_node("Task", {
            "id": "task-b",
            "description": "Sub-task B",
            "domain": "CODE",
            "status": "READY",
        })

        store.create_edge("DECOMPOSES_TO", goal_id, t1_id)
        store.create_edge("DECOMPOSES_TO", goal_id, t2_id)

        rows = store.query(
            "MATCH (g:Goal)-[:DECOMPOSES_TO]->(t:Task) "
            "WHERE g.id = $gid RETURN t.id ORDER BY t.id",
            {"gid": "goal-main"},
        )
        assert len(rows) == 2
        assert rows[0]["t.id"] == "task-a"
        assert rows[1]["t.id"] == "task-b"


class TestGoalFullCrudCycle:
    """Full CRUD cycle test for Goal nodes."""

    def test_full_crud_cycle(self, store: GraphStore) -> None:
        """Create -> Read -> Update -> Delete cycle works for Goal."""
        # Create
        goal_id = store.create_node("Goal", {
            "description": "CRUD cycle test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 2,
            "progress": 0.0,
        })
        assert isinstance(goal_id, str)

        # Read
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["description"] == "CRUD cycle test"
        assert node["status"] == "active"
        assert node["priority"] == 2
        assert node["progress"] == 0.0

        # Update status
        store.update_node("Goal", goal_id, {
            "status": GoalStatus.COMPLETED.value,
            "progress": 1.0,
        })
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["status"] == "completed"
        assert node["progress"] == 1.0

        # Delete
        deleted = store.delete_node("Goal", goal_id)
        assert deleted is True
        assert store.get_node("Goal", goal_id) is None
