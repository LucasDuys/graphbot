"""Tests for cron-triggered goal evaluation script (T173).

Verifies:
  - Active goals are discovered and evaluated
  - Progress is updated via GoalManager.update_progress()
  - Ready sub-tasks are triggered via Orchestrator
  - WhatsApp notification is sent when WHATSAPP_BRIDGE_URL is set
  - No notification when WHATSAPP_BRIDGE_URL is not set
  - Graceful handling of errors during evaluation and notification
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.goals import GoalManager
from core_gb.types import Domain, ExecutionResult, TaskNode, TaskStatus
from graph.schema import GoalStatus
from graph.store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> GraphStore:
    """In-memory Kuzu graph store with schema initialized."""
    s = GraphStore(db_path=None)
    s.initialize()
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def mock_decomposer() -> MagicMock:
    """Mock Decomposer that returns 3 atomic sub-tasks."""
    task_nodes = [
        TaskNode(
            id="leaf-a",
            description="Sub-task A",
            parent_id="root",
            is_atomic=True,
            domain=Domain.CODE,
            complexity=1,
            status=TaskStatus.CREATED,
        ),
        TaskNode(
            id="leaf-b",
            description="Sub-task B",
            parent_id="root",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            status=TaskStatus.CREATED,
        ),
        TaskNode(
            id="leaf-c",
            description="Sub-task C",
            parent_id="root",
            is_atomic=True,
            domain=Domain.FILE,
            complexity=1,
            status=TaskStatus.CREATED,
        ),
    ]
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=task_nodes)
    return decomposer


def _create_active_goal_with_tasks(
    store: GraphStore, mock_decomposer: MagicMock, goal_id: str = "goal-eval-001"
) -> tuple[str, list[str], GoalManager]:
    """Helper: create an active goal with decomposed sub-tasks.

    Returns (goal_id, sub_task_ids, goal_manager).
    """
    gid = store.create_node("Goal", {
        "id": goal_id,
        "description": "Evaluation test goal",
        "status": GoalStatus.ACTIVE.value,
        "priority": 1,
        "progress": 0.0,
    })
    manager = GoalManager(store=store, decomposer=mock_decomposer)
    return gid, [], manager


# ---------------------------------------------------------------------------
# Test: evaluate_goals discovers and evaluates active goals
# ---------------------------------------------------------------------------


class TestEvaluateActiveGoals:
    """Active goals are discovered and their progress is updated."""

    @pytest.mark.asyncio
    async def test_active_goal_evaluated(self, store: GraphStore, mock_decomposer: MagicMock) -> None:
        """An active goal has its progress updated by evaluate_goals."""
        from scripts.evaluate_goals import evaluate_goals

        goal_id = store.create_node("Goal", {
            "id": "goal-active-eval",
            "description": "Active goal for evaluation",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        manager = GoalManager(store=store, decomposer=mock_decomposer)

        # Decompose the goal to create sub-tasks
        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark one sub-task as completed
        store.update_node("Task", sub_task_ids[0], {
            "status": TaskStatus.COMPLETED.value,
        })

        result = await evaluate_goals(store=store, decomposer=mock_decomposer)

        # Goal progress should be updated
        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["progress"] > 0.0
        assert result.goals_evaluated >= 1

    @pytest.mark.asyncio
    async def test_paused_goal_not_evaluated(self, store: GraphStore, mock_decomposer: MagicMock) -> None:
        """Paused goals are skipped during evaluation."""
        from scripts.evaluate_goals import evaluate_goals

        store.create_node("Goal", {
            "id": "goal-paused",
            "description": "Paused goal",
            "status": GoalStatus.PAUSED.value,
            "priority": 1,
            "progress": 0.5,
        })

        result = await evaluate_goals(store=store, decomposer=mock_decomposer)

        assert result.goals_evaluated == 0

    @pytest.mark.asyncio
    async def test_completed_goal_not_evaluated(self, store: GraphStore, mock_decomposer: MagicMock) -> None:
        """Completed goals are skipped during evaluation."""
        from scripts.evaluate_goals import evaluate_goals

        store.create_node("Goal", {
            "id": "goal-completed",
            "description": "Completed goal",
            "status": GoalStatus.COMPLETED.value,
            "priority": 1,
            "progress": 1.0,
        })

        result = await evaluate_goals(store=store, decomposer=mock_decomposer)

        assert result.goals_evaluated == 0

    @pytest.mark.asyncio
    async def test_multiple_active_goals_all_evaluated(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """All active goals are evaluated, not just the first one."""
        from scripts.evaluate_goals import evaluate_goals

        for i in range(3):
            store.create_node("Goal", {
                "id": f"goal-multi-{i}",
                "description": f"Multi goal {i}",
                "status": GoalStatus.ACTIVE.value,
                "priority": 1,
                "progress": 0.0,
            })

        result = await evaluate_goals(store=store, decomposer=mock_decomposer)

        assert result.goals_evaluated == 3


# ---------------------------------------------------------------------------
# Test: progress is updated correctly
# ---------------------------------------------------------------------------


class TestProgressUpdate:
    """GoalManager.update_progress is called for each active goal."""

    @pytest.mark.asyncio
    async def test_progress_updated_after_evaluation(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """After evaluation, goal progress reflects sub-task completion."""
        from scripts.evaluate_goals import evaluate_goals

        goal_id = store.create_node("Goal", {
            "id": "goal-progress-update",
            "description": "Progress update test",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        manager = GoalManager(store=store, decomposer=mock_decomposer)
        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark 2 of 3 tasks completed
        store.update_node("Task", sub_task_ids[0], {"status": TaskStatus.COMPLETED.value})
        store.update_node("Task", sub_task_ids[1], {"status": TaskStatus.COMPLETED.value})

        await evaluate_goals(store=store, decomposer=mock_decomposer)

        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert abs(node["progress"] - 2.0 / 3.0) < 0.01

    @pytest.mark.asyncio
    async def test_goal_auto_completed_when_all_done(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """Goal status becomes 'completed' when all sub-tasks are done."""
        from scripts.evaluate_goals import evaluate_goals

        goal_id = store.create_node("Goal", {
            "id": "goal-auto-complete",
            "description": "Auto complete via evaluation",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        manager = GoalManager(store=store, decomposer=mock_decomposer)
        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark all tasks completed
        for tid in sub_task_ids:
            store.update_node("Task", tid, {"status": TaskStatus.COMPLETED.value})

        await evaluate_goals(store=store, decomposer=mock_decomposer)

        node = store.get_node("Goal", goal_id)
        assert node is not None
        assert node["status"] == GoalStatus.COMPLETED.value
        assert node["progress"] == 1.0


# ---------------------------------------------------------------------------
# Test: ready sub-tasks trigger Orchestrator execution
# ---------------------------------------------------------------------------


class TestReadyTaskExecution:
    """Ready sub-tasks discovered during evaluation are triggered via Orchestrator."""

    @pytest.mark.asyncio
    async def test_ready_task_triggered(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """A sub-task with status 'ready' is executed via Orchestrator.process."""
        from scripts.evaluate_goals import evaluate_goals

        goal_id = store.create_node("Goal", {
            "id": "goal-trigger-ready",
            "description": "Trigger ready tasks",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        manager = GoalManager(store=store, decomposer=mock_decomposer)
        sub_task_ids = await manager.decompose_goal(goal_id)

        # Mark one sub-task as 'ready'
        store.update_node("Task", sub_task_ids[0], {"status": TaskStatus.READY.value})

        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value=ExecutionResult(
            root_id="result-001",
            output="Task completed successfully",
            success=True,
        ))

        result = await evaluate_goals(
            store=store,
            decomposer=mock_decomposer,
            orchestrator=mock_orchestrator,
        )

        mock_orchestrator.process.assert_called()
        assert result.tasks_triggered >= 1

    @pytest.mark.asyncio
    async def test_created_task_not_triggered(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """Sub-tasks with status 'created' are not triggered."""
        from scripts.evaluate_goals import evaluate_goals

        goal_id = store.create_node("Goal", {
            "id": "goal-no-trigger",
            "description": "Do not trigger created tasks",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })
        manager = GoalManager(store=store, decomposer=mock_decomposer)
        await manager.decompose_goal(goal_id)

        # All sub-tasks remain 'created' status (not ready)
        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock()

        result = await evaluate_goals(
            store=store,
            decomposer=mock_decomposer,
            orchestrator=mock_orchestrator,
        )

        mock_orchestrator.process.assert_not_called()
        assert result.tasks_triggered == 0


# ---------------------------------------------------------------------------
# Test: WhatsApp notification
# ---------------------------------------------------------------------------


class TestWhatsAppNotification:
    """WhatsApp status notification is sent when WHATSAPP_BRIDGE_URL is set."""

    @pytest.mark.asyncio
    async def test_notification_sent_when_bridge_url_set(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """A WhatsApp notification is sent when WHATSAPP_BRIDGE_URL is configured."""
        from scripts.evaluate_goals import evaluate_goals, send_whatsapp_notification

        store.create_node("Goal", {
            "id": "goal-notify",
            "description": "Notification test goal",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.5,
        })

        with patch("scripts.evaluate_goals.send_whatsapp_notification", new_callable=AsyncMock) as mock_notify:
            await evaluate_goals(
                store=store,
                decomposer=mock_decomposer,
                whatsapp_bridge_url="http://localhost:3001/send",
            )

            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            # First positional arg is the bridge URL
            assert call_args[0][0] == "http://localhost:3001/send"
            # Second positional arg is the summary message (a string)
            assert isinstance(call_args[0][1], str)

    @pytest.mark.asyncio
    async def test_no_notification_when_bridge_url_not_set(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """No notification is sent when WHATSAPP_BRIDGE_URL is not configured."""
        from scripts.evaluate_goals import evaluate_goals

        store.create_node("Goal", {
            "id": "goal-no-notify",
            "description": "No notification goal",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        with patch("scripts.evaluate_goals.send_whatsapp_notification", new_callable=AsyncMock) as mock_notify:
            await evaluate_goals(
                store=store,
                decomposer=mock_decomposer,
                whatsapp_bridge_url=None,
            )

            mock_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_notification_contains_goal_summary(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """Notification message includes goal description and progress."""
        from scripts.evaluate_goals import evaluate_goals

        store.create_node("Goal", {
            "id": "goal-summary-notify",
            "description": "Build the API",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        with patch("scripts.evaluate_goals.send_whatsapp_notification", new_callable=AsyncMock) as mock_notify:
            await evaluate_goals(
                store=store,
                decomposer=mock_decomposer,
                whatsapp_bridge_url="http://localhost:3001/send",
            )

            mock_notify.assert_called_once()
            message = mock_notify.call_args[0][1]
            assert "Build the API" in message

    @pytest.mark.asyncio
    async def test_notification_error_does_not_crash_evaluation(
        self, store: GraphStore, mock_decomposer: MagicMock,
    ) -> None:
        """If notification fails, evaluation still completes without error."""
        from scripts.evaluate_goals import evaluate_goals

        store.create_node("Goal", {
            "id": "goal-notify-error",
            "description": "Notification failure goal",
            "status": GoalStatus.ACTIVE.value,
            "priority": 1,
            "progress": 0.0,
        })

        with patch(
            "scripts.evaluate_goals.send_whatsapp_notification",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            # Should not raise
            result = await evaluate_goals(
                store=store,
                decomposer=mock_decomposer,
                whatsapp_bridge_url="http://localhost:3001/send",
            )

            assert result.goals_evaluated == 1


# ---------------------------------------------------------------------------
# Test: send_whatsapp_notification function
# ---------------------------------------------------------------------------


class TestSendWhatsAppNotification:
    """Unit tests for the send_whatsapp_notification function itself."""

    @pytest.mark.asyncio
    async def test_send_posts_to_bridge_url(self) -> None:
        """send_whatsapp_notification posts the message to the bridge URL."""
        from scripts.evaluate_goals import send_whatsapp_notification

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await send_whatsapp_notification(
                "http://localhost:3001/send",
                "Goal progress: 50%",
            )

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:3001/send"

    @pytest.mark.asyncio
    async def test_send_raises_on_http_error(self) -> None:
        """send_whatsapp_notification propagates HTTP errors."""
        from scripts.evaluate_goals import send_whatsapp_notification

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("500 Internal Server Error")
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(Exception, match="500"):
                await send_whatsapp_notification(
                    "http://localhost:3001/send",
                    "Goal progress: 50%",
                )


# ---------------------------------------------------------------------------
# Test: EvaluationResult dataclass
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    """EvaluationResult contains summary fields for the evaluation run."""

    def test_result_fields(self) -> None:
        """EvaluationResult has the expected fields with defaults."""
        from scripts.evaluate_goals import EvaluationResult

        result = EvaluationResult()
        assert result.goals_evaluated == 0
        assert result.tasks_triggered == 0
        assert result.errors == []

    def test_result_with_values(self) -> None:
        """EvaluationResult stores custom values."""
        from scripts.evaluate_goals import EvaluationResult

        result = EvaluationResult(
            goals_evaluated=5,
            tasks_triggered=2,
            errors=["Goal X failed"],
        )
        assert result.goals_evaluated == 5
        assert result.tasks_triggered == 2
        assert result.errors == ["Goal X failed"]
