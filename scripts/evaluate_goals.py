"""Cron-triggered goal evaluation with optional WhatsApp notification.

Checks all active goals in the knowledge graph, updates their progress via
GoalManager.update_progress(), triggers ready sub-tasks via the Orchestrator,
and optionally sends a progress summary over WhatsApp.

Can be wired into:
  - OS cron: ``python -m scripts.evaluate_goals``
  - Nanobot's cron service: register as a cron job payload

Environment variables:
  WHATSAPP_BRIDGE_URL  -- If set, a progress summary is POSTed here after evaluation.

Usage:
    python -m scripts.evaluate_goals
    python -m scripts.evaluate_goals --db-path data/graphbot.db
    python -m scripts.evaluate_goals --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from core_gb.decomposer import Decomposer
    from core_gb.orchestrator import Orchestrator
    from graph.store import GraphStore

# Allow running from project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core_gb.goals import GoalManager
from core_gb.types import TaskStatus
from graph.schema import GoalStatus

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "graphbot.db"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Summary of a goal evaluation run.

    Attributes:
        goals_evaluated: Number of active goals whose progress was updated.
        tasks_triggered: Number of ready sub-tasks that were executed.
        errors: List of error messages encountered during evaluation.
    """

    goals_evaluated: int = 0
    tasks_triggered: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WhatsApp notification
# ---------------------------------------------------------------------------


async def send_whatsapp_notification(bridge_url: str, message: str) -> None:
    """POST a text notification to the WhatsApp bridge.

    Args:
        bridge_url: The HTTP endpoint of the WhatsApp bridge (e.g. http://localhost:3001/send).
        message: The text message to send.

    Raises:
        httpx.HTTPStatusError: If the bridge returns a non-2xx status.
        Exception: On network or other transport errors.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            bridge_url,
            json={"type": "status", "text": message},
        )
        response.raise_for_status()
    logger.info("WhatsApp notification sent (%d chars)", len(message))


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def _get_active_goals(store: GraphStore) -> list[dict[str, object]]:
    """Query all Goal nodes with status 'active'.

    Returns a list of row dicts with keys 'g.id', 'g.description', 'g.progress'.
    """
    rows = store.query(
        "MATCH (g:Goal) WHERE g.status = $status "
        "RETURN g.id, g.description, g.progress",
        {"status": GoalStatus.ACTIVE.value},
    )
    return rows


def _get_ready_subtasks(store: GraphStore, goal_id: str) -> list[dict[str, object]]:
    """Query sub-tasks of a goal that have status 'ready'.

    Returns a list of row dicts with keys 't.id', 't.description'.
    """
    rows = store.query(
        "MATCH (g:Goal)-[:DECOMPOSES_TO]->(t:Task) "
        "WHERE g.id = $gid AND t.status = $status "
        "RETURN t.id, t.description",
        {"gid": goal_id, "status": TaskStatus.READY.value},
    )
    return rows


def _build_summary(
    goals: list[dict[str, object]],
    progress_map: dict[str, float],
) -> str:
    """Build a human-readable progress summary for notification.

    Args:
        goals: Active goal rows from the graph query.
        progress_map: Mapping of goal_id to updated progress value.

    Returns:
        A formatted summary string.
    """
    lines: list[str] = ["GraphBot Goal Evaluation Report", ""]
    for goal in goals:
        goal_id = str(goal.get("g.id", "unknown"))
        description = str(goal.get("g.description", ""))[:60]
        progress = progress_map.get(goal_id, 0.0)
        pct = progress * 100
        lines.append(f"- {description}: {pct:.0f}%")
    if not goals:
        lines.append("No active goals found.")
    return "\n".join(lines)


async def evaluate_goals(
    store: GraphStore,
    decomposer: Decomposer,
    orchestrator: Orchestrator | None = None,
    whatsapp_bridge_url: str | None = None,
) -> EvaluationResult:
    """Evaluate all active goals: update progress, trigger ready tasks, notify.

    Args:
        store: The graph store containing Goal and Task nodes.
        decomposer: A Decomposer instance (used to construct GoalManager).
        orchestrator: Optional Orchestrator for executing ready sub-tasks.
        whatsapp_bridge_url: If provided, a progress summary is POSTed here.

    Returns:
        EvaluationResult summarising the evaluation run.
    """
    result = EvaluationResult()
    manager = GoalManager(store=store, decomposer=decomposer)

    # Discover active goals.
    active_goals = _get_active_goals(store)
    progress_map: dict[str, float] = {}

    for goal_row in active_goals:
        goal_id = str(goal_row.get("g.id", ""))
        if not goal_id:
            continue

        # Update progress.
        try:
            progress = manager.update_progress(goal_id)
            progress_map[goal_id] = progress
            result.goals_evaluated += 1
            logger.info(
                "Goal '%s' evaluated: progress=%.1f%%", goal_id, progress * 100
            )
        except Exception as exc:
            error_msg = f"Error evaluating goal '{goal_id}': {exc}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            continue

        # Trigger ready sub-tasks if an orchestrator is available.
        if orchestrator is not None:
            ready_tasks = _get_ready_subtasks(store, goal_id)
            for task_row in ready_tasks:
                task_id = str(task_row.get("t.id", ""))
                description = str(task_row.get("t.description", ""))
                if not description:
                    continue

                try:
                    logger.info(
                        "Triggering ready task '%s' for goal '%s'",
                        task_id, goal_id,
                    )
                    await orchestrator.process(description)
                    result.tasks_triggered += 1

                    # Mark the task as completed after successful execution.
                    store.update_node("Task", task_id, {
                        "status": TaskStatus.COMPLETED.value,
                    })
                except Exception as exc:
                    error_msg = (
                        f"Error executing task '{task_id}' "
                        f"for goal '{goal_id}': {exc}"
                    )
                    logger.error(error_msg)
                    result.errors.append(error_msg)

    # Send WhatsApp notification if configured.
    if whatsapp_bridge_url:
        summary = _build_summary(active_goals, progress_map)
        try:
            await send_whatsapp_notification(whatsapp_bridge_url, summary)
        except Exception as exc:
            error_msg = f"WhatsApp notification failed: {exc}"
            logger.warning(error_msg)
            result.errors.append(error_msg)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load .env.local from project root into environment (idempotent)."""
    env_file = _PROJECT_ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


async def _async_main(db_path: Path) -> EvaluationResult:
    """Async entry point for the CLI."""
    from core_gb.decomposer import Decomposer
    from core_gb.orchestrator import Orchestrator
    from graph.store import GraphStore
    from models.openrouter import OpenRouterProvider
    from models.router import ModelRouter

    _load_env()

    store = GraphStore(db_path=str(db_path))
    store.initialize()

    try:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        provider = OpenRouterProvider(api_key=api_key)
        router = ModelRouter(provider)
        decomposer = Decomposer(router)
        orchestrator = Orchestrator(store, router)

        whatsapp_url = os.environ.get("WHATSAPP_BRIDGE_URL")

        return await evaluate_goals(
            store=store,
            decomposer=decomposer,
            orchestrator=orchestrator,
            whatsapp_bridge_url=whatsapp_url,
        )
    finally:
        store.close()


def main() -> None:
    """CLI entry point for cron-triggered goal evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate active goals: update progress, trigger ready tasks, notify."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=_DEFAULT_DB_PATH,
        help=f"Path to Kuzu database (default: {_DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = asyncio.run(_async_main(args.db_path))

    print(
        f"Goal evaluation complete:\n"
        f"  Goals evaluated:  {result.goals_evaluated}\n"
        f"  Tasks triggered:  {result.tasks_triggered}\n"
        f"  Errors:           {len(result.errors)}"
    )

    if result.errors:
        print("\n  Error details:")
        for err in result.errors:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
