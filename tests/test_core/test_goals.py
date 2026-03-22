"""Tests for GoalManager: goal decomposition and progress tracking (T170)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.goals import GoalManager
from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from graph.schema import GoalStatus
from graph.store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> GraphStore:
    """Provide an initialized in-memory GraphStore."""
    s = GraphStore(db_path=None)
    s.initialize()
    yield s  # type: ignore[misc]
    s.close()


def _make_mock_decomposer(task_nodes: list[TaskNode]) -> MagicMock:
    """Create a mock Decomposer that returns the given task nodes."""
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=task_nodes)
    return decomposer


def _sample_task_nodes() -> list[TaskNode]:
    """Return a sample decomposition: 1 root + 3 atomic leaves."""
    return [
        TaskNode(
            id="root-001",
            description="Root task",
            children=["leaf-001", "leaf-002", "leaf-003"],
            is_atomic=False,
            domain=Domain.SYNTHESIS,
            complexity=2,
            status=TaskStatus.CREATED,
        ),
        TaskNode(
            id="leaf-001",
            description="Sub-task A",
            parent_id="root-001",
            is_atomic=True,
            domain=Domain.CODE,
            complexity=1,
            status=TaskStatus.CREATED,
        ),
        TaskNode(
            id="leaf-002",
            description="Sub-task B",
            parent_id="root-001",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            status=TaskStatus.CREATED,
        ),
        TaskNode(
            id="leaf-003",
            description="Sub-task C",
            parent_id="root-001",
            is_atomic=True,
            domain=Domain.FILE,
            complexity=1,
            status=TaskStatus.CREATED,
        ),
    ]


# ---------------------------------------------------------------------------
# GoalManager instantiation
# ---------------------------------------------------------------------------


class TestGoalManagerInit:
    """GoalManager requires a GraphStore and Decomposer."""

    def test_init_with_store_and_decomposer(self, store: GraphStore) -> None:
        """GoalManager can be instantiated with store and decomposer."""
        decomposer = _make_mock_decomposer([])
        manager = GoalManager(store=store, decomposer=decomposer)
        assert manager is not None

    def test_init_stores_references(self, store: GraphStore) -> None:
        """GoalManager stores references to store and decomposer."""
        decomposer = _make_mock_decomposer([])
        manager = GoalManager(store=store, decomposer=decomposer)
        assert manager._store is store
        assert manager._decomposer is decomposer


# ---------------------------------------------------------------------------
# decompose_goal
# ---------------------------------------------------------------------------


class TestDecomposeGoal:
    """Tests for decompose_goal() method."""

    @pytest.mark.asyncio
    async def test_decompose_goal_creates_task_nodes(self, store: GraphStore) -> None:
        """decompose_goal creates Task nodes in the graph for each sub-task."""
        goal_id = store.create_node("Goal", {
            "description": "Ship MVP",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        assert len(sub_task_ids) > 0
        # Each returned ID should correspond to a Task node in the store
        for tid in sub_task_ids:
            node = store.get_node("Task", tid)
            assert node is not None, f"Task {tid} not found in store"

    @pytest.mark.asyncio
    async def test_decompose_goal_calls_decomposer_with_description(
        self, store: GraphStore
    ) -> None:
        """decompose_goal passes the goal description to the Decomposer."""
        goal_id = store.create_node("Goal", {
            "description": "Build the authentication system",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        await manager.decompose_goal(goal_id)

        decomposer.decompose.assert_called_once()
        call_args = decomposer.decompose.call_args
        assert call_args[0][0] == "Build the authentication system"

    @pytest.mark.asyncio
    async def test_decompose_goal_creates_decomposes_to_edges(
        self, store: GraphStore
    ) -> None:
        """decompose_goal creates DECOMPOSES_TO edges from Goal to each sub-Task."""
        goal_id = store.create_node("Goal", {
            "id": "goal-decomp-edge",
            "description": "Edge test goal",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        # Query the graph for DECOMPOSES_TO edges from this goal
        rows = store.query(
            "MATCH (g:Goal)-[:DECOMPOSES_TO]->(t:Task) "
            "WHERE g.id = $gid RETURN t.id ORDER BY t.id",
            {"gid": "goal-decomp-edge"},
        )
        edge_task_ids = sorted([r["t.id"] for r in rows])
        assert len(edge_task_ids) == len(sub_task_ids)
        assert edge_task_ids == sorted(sub_task_ids)

    @pytest.mark.asyncio
    async def test_decompose_goal_sets_initial_progress_zero(
        self, store: GraphStore
    ) -> None:
        """After decomposition, goal progress is set to 0.0."""
        goal_id = store.create_node("Goal", {
            "description": "Progress init test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.5,  # pre-existing value, should be reset
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        await manager.decompose_goal(goal_id)

        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["progress"] == 0.0

    @pytest.mark.asyncio
    async def test_decompose_goal_raises_for_missing_goal(
        self, store: GraphStore
    ) -> None:
        """decompose_goal raises ValueError if goal_id does not exist."""
        decomposer = _make_mock_decomposer(_sample_task_nodes())
        manager = GoalManager(store=store, decomposer=decomposer)

        with pytest.raises(ValueError, match="Goal .* not found"):
            await manager.decompose_goal("nonexistent-goal-id")

    @pytest.mark.asyncio
    async def test_decompose_goal_only_stores_atomic_subtasks(
        self, store: GraphStore
    ) -> None:
        """Only atomic (leaf) task nodes get stored and linked as sub-goals."""
        goal_id = store.create_node("Goal", {
            "description": "Atomic only test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        # Only the 3 atomic leaves should be returned, not the root
        assert len(sub_task_ids) == 3

    @pytest.mark.asyncio
    async def test_decompose_goal_stores_task_description(
        self, store: GraphStore
    ) -> None:
        """Each sub-task stored in the graph has the correct description."""
        goal_id = store.create_node("Goal", {
            "description": "Description check",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        descriptions = set()
        for tid in sub_task_ids:
            node = store.get_node("Task", tid)
            assert node is not None
            descriptions.add(node["description"])

        assert "Sub-task A" in descriptions
        assert "Sub-task B" in descriptions
        assert "Sub-task C" in descriptions

    @pytest.mark.asyncio
    async def test_decompose_goal_stores_task_domain(
        self, store: GraphStore
    ) -> None:
        """Each sub-task stored in the graph has a domain set."""
        goal_id = store.create_node("Goal", {
            "description": "Domain check",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        domains = set()
        for tid in sub_task_ids:
            node = store.get_node("Task", tid)
            assert node is not None
            domains.add(node["domain"])

        # Our sample has CODE, WEB, FILE
        assert domains == {"code", "web", "file"}

    @pytest.mark.asyncio
    async def test_decompose_goal_stores_task_status_created(
        self, store: GraphStore
    ) -> None:
        """Each sub-task is stored with status 'created'."""
        goal_id = store.create_node("Goal", {
            "description": "Status check",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        for tid in sub_task_ids:
            node = store.get_node("Task", tid)
            assert node is not None
            assert node["status"] == "created"


# ---------------------------------------------------------------------------
# update_progress
# ---------------------------------------------------------------------------


class TestUpdateProgress:
    """Tests for update_progress() method."""

    @pytest.mark.asyncio
    async def test_progress_zero_when_no_tasks_completed(
        self, store: GraphStore
    ) -> None:
        """Progress is 0.0 when no sub-tasks are completed."""
        goal_id = store.create_node("Goal", {
            "id": "goal-prog-0",
            "description": "No progress yet",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        progress = manager.update_progress(goal_id)

        assert progress == 0.0
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["progress"] == 0.0

    @pytest.mark.asyncio
    async def test_progress_one_third_when_one_of_three_completed(
        self, store: GraphStore
    ) -> None:
        """Progress is ~0.333 when 1 of 3 sub-tasks is completed."""
        goal_id = store.create_node("Goal", {
            "id": "goal-prog-33",
            "description": "One third done",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark one sub-task as completed
        store.update_node("Task", sub_task_ids[0], {
            "status": TaskStatus.COMPLETED.value,
        })

        progress = manager.update_progress(goal_id)

        assert abs(progress - 1.0 / 3.0) < 0.01
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert abs(node["progress"] - 1.0 / 3.0) < 0.01

    @pytest.mark.asyncio
    async def test_progress_full_when_all_completed(
        self, store: GraphStore
    ) -> None:
        """Progress is 1.0 when all sub-tasks are completed."""
        goal_id = store.create_node("Goal", {
            "id": "goal-prog-100",
            "description": "All done",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark all sub-tasks as completed
        for tid in sub_task_ids:
            store.update_node("Task", tid, {
                "status": TaskStatus.COMPLETED.value,
            })

        progress = manager.update_progress(goal_id)

        assert progress == 1.0
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["progress"] == 1.0

    @pytest.mark.asyncio
    async def test_progress_two_thirds_when_two_of_three_completed(
        self, store: GraphStore
    ) -> None:
        """Progress is ~0.667 when 2 of 3 sub-tasks are completed."""
        goal_id = store.create_node("Goal", {
            "id": "goal-prog-66",
            "description": "Two thirds done",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark two sub-tasks as completed
        store.update_node("Task", sub_task_ids[0], {
            "status": TaskStatus.COMPLETED.value,
        })
        store.update_node("Task", sub_task_ids[1], {
            "status": TaskStatus.COMPLETED.value,
        })

        progress = manager.update_progress(goal_id)

        assert abs(progress - 2.0 / 3.0) < 0.01
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert abs(node["progress"] - 2.0 / 3.0) < 0.01

    def test_update_progress_raises_for_missing_goal(
        self, store: GraphStore
    ) -> None:
        """update_progress raises ValueError if goal_id does not exist."""
        decomposer = _make_mock_decomposer([])
        manager = GoalManager(store=store, decomposer=decomposer)

        with pytest.raises(ValueError, match="Goal .* not found"):
            manager.update_progress("nonexistent-goal-id")

    @pytest.mark.asyncio
    async def test_progress_zero_when_no_subtasks_exist(
        self, store: GraphStore
    ) -> None:
        """Progress is 0.0 when goal has no sub-tasks (no DECOMPOSES_TO edges)."""
        goal_id = store.create_node("Goal", {
            "id": "goal-no-subtasks",
            "description": "No subtasks",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        decomposer = _make_mock_decomposer([])
        manager = GoalManager(store=store, decomposer=decomposer)

        progress = manager.update_progress(goal_id)

        assert progress == 0.0

    @pytest.mark.asyncio
    async def test_progress_marks_goal_completed_at_full(
        self, store: GraphStore
    ) -> None:
        """When progress reaches 1.0, goal status is set to 'completed'."""
        goal_id = store.create_node("Goal", {
            "id": "goal-auto-complete",
            "description": "Auto complete test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark all sub-tasks as completed
        for tid in sub_task_ids:
            store.update_node("Task", tid, {
                "status": TaskStatus.COMPLETED.value,
            })

        manager.update_progress(goal_id)

        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["status"] == GoalStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_progress_keeps_active_when_not_full(
        self, store: GraphStore
    ) -> None:
        """When progress is less than 1.0, goal status stays 'active'."""
        goal_id = store.create_node("Goal", {
            "id": "goal-still-active",
            "description": "Still active",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark only one sub-task as completed
        store.update_node("Task", sub_task_ids[0], {
            "status": TaskStatus.COMPLETED.value,
        })

        manager.update_progress(goal_id)

        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["status"] == GoalStatus.ACTIVE.value


# ---------------------------------------------------------------------------
# get_sub_tasks
# ---------------------------------------------------------------------------


class TestGetSubTasks:
    """Tests for get_sub_tasks() method."""

    @pytest.mark.asyncio
    async def test_get_sub_tasks_returns_task_ids(
        self, store: GraphStore
    ) -> None:
        """get_sub_tasks returns list of task IDs linked to the goal."""
        goal_id = store.create_node("Goal", {
            "id": "goal-subtask-list",
            "description": "Subtask listing",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        task_nodes = _sample_task_nodes()
        decomposer = _make_mock_decomposer(task_nodes)
        manager = GoalManager(store=store, decomposer=decomposer)

        sub_task_ids = await manager.decompose_goal(goal_id)
        retrieved = manager.get_sub_tasks(goal_id)

        assert sorted(retrieved) == sorted(sub_task_ids)

    def test_get_sub_tasks_empty_when_no_decomposition(
        self, store: GraphStore
    ) -> None:
        """get_sub_tasks returns empty list for a goal with no sub-tasks."""
        goal_id = store.create_node("Goal", {
            "id": "goal-no-subs",
            "description": "No subs",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        decomposer = _make_mock_decomposer([])
        manager = GoalManager(store=store, decomposer=decomposer)

        retrieved = manager.get_sub_tasks(goal_id)
        assert retrieved == []

    def test_get_sub_tasks_raises_for_missing_goal(
        self, store: GraphStore
    ) -> None:
        """get_sub_tasks raises ValueError if goal does not exist."""
        decomposer = _make_mock_decomposer([])
        manager = GoalManager(store=store, decomposer=decomposer)

        with pytest.raises(ValueError, match="Goal .* not found"):
            manager.get_sub_tasks("nonexistent-goal")
