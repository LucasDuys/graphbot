"""Goal decomposition into session-sized sub-goals with progress tracking.

Provides GoalManager, which uses the Decomposer to break a high-level Goal
into atomic sub-Tasks, links them via DECOMPOSES_TO edges in the knowledge
graph, and tracks completion progress as a percentage of completed sub-tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core_gb.types import TaskStatus
from graph.schema import GoalStatus

if TYPE_CHECKING:
    from core_gb.decomposer import Decomposer
    from graph.store import GraphStore

logger = logging.getLogger(__name__)


class GoalManager:
    """Manages goal decomposition and progress tracking.

    Uses a Decomposer to break a Goal node into atomic sub-Tasks (Task nodes),
    creates DECOMPOSES_TO edges from the Goal to each sub-Task, and provides
    progress tracking based on the completion status of those sub-tasks.

    Args:
        store: The graph store for reading/writing nodes and edges.
        decomposer: The task decomposer for breaking goals into sub-tasks.
    """

    def __init__(self, store: GraphStore, decomposer: Decomposer) -> None:
        self._store = store
        self._decomposer = decomposer

    async def decompose_goal(self, goal_id: str) -> list[str]:
        """Decompose a goal into atomic sub-tasks and link them in the graph.

        Reads the Goal node, calls the Decomposer with the goal's description,
        creates a Task node for each atomic sub-task, and creates DECOMPOSES_TO
        edges from the Goal to each sub-Task. Resets the goal's progress to 0.0.

        Args:
            goal_id: The ID of the Goal node to decompose.

        Returns:
            List of created sub-task IDs (only atomic/leaf tasks).

        Raises:
            ValueError: If the goal_id does not correspond to an existing Goal.
        """
        goal_node = self._store.get_node("Goal", goal_id)
        if goal_node is None:
            raise ValueError(f"Goal '{goal_id}' not found in the graph store")

        description = str(goal_node.get("description", ""))
        logger.info("Decomposing goal '%s': %s", goal_id, description[:80])

        # Call the decomposer to break the goal into task nodes
        task_nodes = await self._decomposer.decompose(description)

        # Filter to atomic (leaf) tasks only -- these are the actionable sub-goals
        atomic_nodes = [node for node in task_nodes if node.is_atomic]

        sub_task_ids: list[str] = []
        for node in atomic_nodes:
            # Create a Task node in the graph store
            task_id = self._store.create_node("Task", {
                "id": node.id,
                "description": node.description,
                "domain": node.domain.value,
                "complexity": node.complexity,
                "status": TaskStatus.CREATED.value,
            })
            sub_task_ids.append(task_id)

            # Create DECOMPOSES_TO edge from Goal to Task
            self._store.create_edge("DECOMPOSES_TO", goal_id, task_id)

        # Reset progress to 0.0 after decomposition
        self._store.update_node("Goal", goal_id, {"progress": 0.0})

        logger.info(
            "Goal '%s' decomposed into %d sub-tasks", goal_id, len(sub_task_ids)
        )
        return sub_task_ids

    def update_progress(self, goal_id: str) -> float:
        """Recalculate and update goal progress based on sub-task statuses.

        Queries all sub-tasks linked via DECOMPOSES_TO edges, counts how many
        have status 'completed', and sets the goal's progress to the ratio of
        completed tasks to total tasks.

        If all sub-tasks are completed (progress == 1.0), the goal's status
        is automatically set to 'completed'.

        Args:
            goal_id: The ID of the Goal node to update.

        Returns:
            The updated progress value (0.0 to 1.0).

        Raises:
            ValueError: If the goal_id does not correspond to an existing Goal.
        """
        goal_node = self._store.get_node("Goal", goal_id)
        if goal_node is None:
            raise ValueError(f"Goal '{goal_id}' not found in the graph store")

        # Query all sub-tasks linked to this goal
        rows = self._store.query(
            "MATCH (g:Goal)-[:DECOMPOSES_TO]->(t:Task) "
            "WHERE g.id = $gid RETURN t.id, t.status",
            {"gid": goal_id},
        )

        total = len(rows)
        if total == 0:
            logger.debug("Goal '%s' has no sub-tasks, progress remains 0.0", goal_id)
            self._store.update_node("Goal", goal_id, {"progress": 0.0})
            return 0.0

        completed = sum(
            1 for row in rows if row.get("t.status") == TaskStatus.COMPLETED.value
        )
        progress = completed / total

        # Update the goal's progress in the store
        update_props: dict[str, object] = {"progress": progress}

        # Auto-complete the goal when all sub-tasks are done
        if progress >= 1.0:
            update_props["status"] = GoalStatus.COMPLETED.value
            logger.info("Goal '%s' completed (all %d sub-tasks done)", goal_id, total)

        self._store.update_node("Goal", goal_id, update_props)

        logger.debug(
            "Goal '%s' progress: %.1f%% (%d/%d sub-tasks completed)",
            goal_id, progress * 100, completed, total,
        )
        return progress

    def get_sub_tasks(self, goal_id: str) -> list[str]:
        """Return the IDs of all sub-tasks linked to a goal.

        Args:
            goal_id: The ID of the Goal node.

        Returns:
            List of sub-task IDs.

        Raises:
            ValueError: If the goal_id does not correspond to an existing Goal.
        """
        goal_node = self._store.get_node("Goal", goal_id)
        if goal_node is None:
            raise ValueError(f"Goal '{goal_id}' not found in the graph store")

        rows = self._store.query(
            "MATCH (g:Goal)-[:DECOMPOSES_TO]->(t:Task) "
            "WHERE g.id = $gid RETURN t.id",
            {"gid": goal_id},
        )
        return [str(row["t.id"]) for row in rows]
