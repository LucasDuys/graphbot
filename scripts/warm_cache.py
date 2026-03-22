"""Pattern cache warming script.

Runs 30+ diverse tasks through the Orchestrator to populate the pattern cache,
then re-runs the same task types to demonstrate cache hits and token reduction.

Usage:
    python scripts/warm_cache.py [--db-path PATH]

Without --db-path, uses an in-memory database (results are transient).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _load_env() -> None:
    """Load environment variables from .env.local if present."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

from core_gb.orchestrator import Orchestrator
from core_gb.types import ExecutionResult
from graph.store import GraphStore
from models.router import ModelRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task definitions: 6 categories x 6 tasks each = 36 tasks
# ---------------------------------------------------------------------------

TASK_CATEGORIES: dict[str, list[str]] = {
    "summarization": [
        "Summarize the key points of the Python asyncio documentation",
        "Summarize the differences between REST and GraphQL APIs",
        "Summarize the main features of the Rust programming language",
        "Summarize the benefits of microservices architecture",
        "Summarize the principles of clean code by Robert C. Martin",
        "Summarize the evolution of JavaScript frameworks from 2015 to 2025",
    ],
    "code_generation": [
        "Write a Python function to merge two sorted linked lists",
        "Write a Python function to validate an email address with regex",
        "Write a Python function to implement a simple LRU cache",
        "Write a Python function to flatten a nested dictionary",
        "Write a Python function to compute the Levenshtein distance",
        "Write a Python function to parse CSV data without external libraries",
    ],
    "analysis": [
        "Analyze the time complexity of quicksort in best, average, and worst cases",
        "Analyze the trade-offs between SQL and NoSQL databases for e-commerce",
        "Analyze the security implications of using JWT tokens for authentication",
        "Analyze the performance impact of Python GIL on multi-threaded applications",
        "Analyze the pros and cons of monorepo vs polyrepo strategies",
        "Analyze the memory usage patterns of garbage-collected vs manual memory languages",
    ],
    "comparison": [
        "Compare Python and Go for building web servers",
        "Compare Docker and Podman for container management",
        "Compare PostgreSQL and MySQL for transactional workloads",
        "Compare React and Vue for building single-page applications",
        "Compare Kubernetes and Docker Swarm for container orchestration",
        "Compare FastAPI and Flask for building REST APIs",
    ],
    "explanation": [
        "Explain how a B-tree index works in a database",
        "Explain the CAP theorem with practical examples",
        "Explain how TLS handshake establishes a secure connection",
        "Explain the difference between processes and threads in operating systems",
        "Explain how consistent hashing works in distributed systems",
        "Explain the event loop model in Node.js",
    ],
    "planning": [
        "Create a deployment plan for migrating a monolith to microservices",
        "Create a testing strategy for a real-time chat application",
        "Create a database migration plan for adding multi-tenancy support",
        "Create a CI/CD pipeline plan for a Python monorepo",
        "Create a disaster recovery plan for a cloud-native application",
        "Create a performance optimization plan for a slow Django application",
    ],
}


@dataclass
class RunStats:
    """Aggregated statistics for a set of task executions."""

    total_tokens: int = 0
    total_latency_ms: float = 0.0
    tasks_run: int = 0
    successes: int = 0
    failures: int = 0
    llm_calls: int = 0
    per_task: list[dict[str, object]] = field(default_factory=list)

    def record(self, task: str, result: ExecutionResult) -> None:
        """Record a single task execution result."""
        self.tasks_run += 1
        self.total_tokens += result.total_tokens
        self.total_latency_ms += result.total_latency_ms
        self.llm_calls += result.llm_calls
        if result.success:
            self.successes += 1
        else:
            self.failures += 1
        self.per_task.append({
            "task": task[:80],
            "tokens": result.total_tokens,
            "latency_ms": result.total_latency_ms,
            "llm_calls": result.llm_calls,
            "success": result.success,
        })


def _get_graph_stats(store: GraphStore) -> dict[str, int]:
    """Query node counts for Task, PatternNode, and ExecutionTree."""
    stats: dict[str, int] = {}
    for table in ("Task", "PatternNode", "ExecutionTree"):
        try:
            rows = store.query(f"MATCH (n:{table}) RETURN count(n) AS cnt")
            stats[table] = int(rows[0]["cnt"]) if rows else 0
        except Exception:
            stats[table] = 0
    return stats


def _all_tasks() -> list[tuple[str, str]]:
    """Return all tasks as (category, task_description) pairs."""
    tasks: list[tuple[str, str]] = []
    for category, task_list in TASK_CATEGORIES.items():
        for task in task_list:
            tasks.append((category, task))
    return tasks


async def _run_pass(
    orchestrator: Orchestrator,
    tasks: list[tuple[str, str]],
    label: str,
) -> RunStats:
    """Run all tasks through the orchestrator and collect stats."""
    stats = RunStats()
    total = len(tasks)

    for idx, (category, task) in enumerate(tasks, 1):
        short = task[:60]
        print(f"  [{label}] ({idx}/{total}) [{category}] {short}...")
        try:
            result = await orchestrator.process(task)
            stats.record(task, result)
        except Exception as exc:
            logger.warning("Task failed: %s -- %s", short, exc)
            stats.tasks_run += 1
            stats.failures += 1
            stats.per_task.append({
                "task": task[:80],
                "tokens": 0,
                "latency_ms": 0.0,
                "llm_calls": 0,
                "success": False,
                "error": str(exc)[:120],
            })

    return stats


def _print_summary(
    cold_stats: RunStats,
    warm_stats: RunStats,
    pre_stats: dict[str, int],
    post_cold_stats: dict[str, int],
    post_warm_stats: dict[str, int],
) -> None:
    """Print a formatted summary of the warming benchmark."""
    print()
    print("=" * 60)
    print("CACHE WARMING SUMMARY")
    print("=" * 60)

    print()
    print("COLD RUN (no pattern cache)")
    print(f"  Tasks run:     {cold_stats.tasks_run}")
    print(f"  Successes:     {cold_stats.successes}")
    print(f"  Failures:      {cold_stats.failures}")
    print(f"  Total tokens:  {cold_stats.total_tokens}")
    print(f"  Total LLM calls: {cold_stats.llm_calls}")
    print(f"  Total latency: {cold_stats.total_latency_ms:.0f} ms")

    print()
    print("WARM RUN (with pattern cache)")
    print(f"  Tasks run:     {warm_stats.tasks_run}")
    print(f"  Successes:     {warm_stats.successes}")
    print(f"  Failures:      {warm_stats.failures}")
    print(f"  Total tokens:  {warm_stats.total_tokens}")
    print(f"  Total LLM calls: {warm_stats.llm_calls}")
    print(f"  Total latency: {warm_stats.total_latency_ms:.0f} ms")

    print()
    print("TOKEN REDUCTION")
    cold_tokens = cold_stats.total_tokens
    warm_tokens = warm_stats.total_tokens
    if cold_tokens > 0:
        reduction_pct = ((cold_tokens - warm_tokens) / cold_tokens) * 100
        print(f"  Cold tokens:   {cold_tokens}")
        print(f"  Warm tokens:   {warm_tokens}")
        print(f"  Reduction:     {reduction_pct:.1f}%")
        if reduction_pct >= 30:
            print(f"  Status:        PASS (>= 30% reduction)")
        else:
            print(f"  Status:        BELOW TARGET (< 30% reduction)")
    else:
        print("  Cold tokens:   0 (cannot compute reduction)")

    print()
    print("LLM CALL REDUCTION")
    cold_calls = cold_stats.llm_calls
    warm_calls = warm_stats.llm_calls
    if cold_calls > 0:
        call_reduction_pct = ((cold_calls - warm_calls) / cold_calls) * 100
        print(f"  Cold calls:    {cold_calls}")
        print(f"  Warm calls:    {warm_calls}")
        print(f"  Reduction:     {call_reduction_pct:.1f}%")
    else:
        print("  Cold calls:    0 (cannot compute reduction)")

    print()
    print("GRAPH GROWTH")
    print(f"  {'Node Type':<20s} {'Before':>8s} {'After Cold':>12s} {'After Warm':>12s}")
    print(f"  {'-' * 52}")
    for table in ("Task", "PatternNode", "ExecutionTree"):
        before = pre_stats.get(table, 0)
        after_cold = post_cold_stats.get(table, 0)
        after_warm = post_warm_stats.get(table, 0)
        print(f"  {table:<20s} {before:>8d} {after_cold:>12d} {after_warm:>12d}")

    pattern_count = post_warm_stats.get("PatternNode", 0)
    print()
    print(f"PATTERN CACHE: {pattern_count} templates")
    if pattern_count >= 10:
        print(f"  Status:        PASS (>= 10 templates)")
    else:
        print(f"  Status:        BELOW TARGET (< 10 templates)")

    print()
    print("=" * 60)


async def main(db_path: str | None = None) -> dict[str, object]:
    """Run the full cache warming benchmark.

    Returns a summary dict for programmatic use.
    """
    store = GraphStore(db_path=db_path)
    store.initialize()

    # Build router -- import providers dynamically based on env
    router = _build_router()
    orchestrator = Orchestrator(store, router)

    tasks = _all_tasks()
    print(f"Loaded {len(tasks)} tasks across {len(TASK_CATEGORIES)} categories")

    # Pre-warming graph stats
    pre_stats = _get_graph_stats(store)
    print(f"Pre-warming graph: {pre_stats}")

    # Cold run
    print()
    print("--- COLD RUN ---")
    cold_stats = await _run_pass(orchestrator, tasks, "COLD")
    post_cold_stats = _get_graph_stats(store)

    # Warm run (same tasks, pattern cache should be populated now)
    print()
    print("--- WARM RUN ---")
    warm_stats = await _run_pass(orchestrator, tasks, "WARM")
    post_warm_stats = _get_graph_stats(store)

    _print_summary(cold_stats, warm_stats, pre_stats, post_cold_stats, post_warm_stats)

    store.close()

    # Return summary for programmatic consumers
    cold_tokens = cold_stats.total_tokens
    warm_tokens = warm_stats.total_tokens
    reduction_pct = (
        ((cold_tokens - warm_tokens) / cold_tokens * 100)
        if cold_tokens > 0
        else 0.0
    )
    return {
        "total_tasks": len(tasks),
        "cold_tokens": cold_tokens,
        "warm_tokens": warm_tokens,
        "reduction_pct": reduction_pct,
        "cold_llm_calls": cold_stats.llm_calls,
        "warm_llm_calls": warm_stats.llm_calls,
        "pattern_count": post_warm_stats.get("PatternNode", 0),
        "graph_stats": post_warm_stats,
    }


def _build_router() -> ModelRouter:
    """Build a ModelRouter from available providers."""
    # Try importing available providers
    try:
        from models.groq_provider import GroqProvider
        provider = GroqProvider()
        return ModelRouter(provider=provider)
    except Exception:
        pass
    try:
        from models.openrouter_provider import OpenRouterProvider
        provider = OpenRouterProvider()
        return ModelRouter(provider=provider)
    except Exception:
        pass
    try:
        from models.cerebras_provider import CerebrasProvider
        provider = CerebrasProvider()
        return ModelRouter(provider=provider)
    except Exception:
        pass

    raise RuntimeError(
        "No LLM provider available. Set GROQ_API_KEY, OPENROUTER_API_KEY, "
        "or CEREBRAS_API_KEY in environment or .env.local"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm the pattern cache")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to Kuzu database directory. Uses in-memory if omitted.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = asyncio.run(main(db_path=args.db_path))
    print()
    print(f"Final summary: {result}")
