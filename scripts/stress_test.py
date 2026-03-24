"""Stress test script with 10 genuinely hard task definitions.

Runs each task through Orchestrator.process(), captures metrics
(success/failure, output, latency, cost, errors), diagnoses failures
with root cause analysis, and saves results to benchmarks/stress_test.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load .env.local file from project root."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


load_env()

from core_gb.orchestrator import Orchestrator
from core_gb.types import ExecutionResult
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Failure root cause categories
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    TIMEOUT = "timeout"
    TOOL_FAILURE = "tool_failure"
    DECOMPOSITION_ERROR = "decomposition_error"
    SAFETY_BLOCKED = "safety_blocked"
    MODEL_REFUSAL = "model_refusal"
    CONTRADICTION = "contradiction"
    CONTEXT_MISSING = "context_missing"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Hard task definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StressTask:
    """A single stress test task definition."""

    id: str
    name: str
    category: str
    description: str
    difficulty: int
    expected_behavior: str
    why_hard: str
    accept_partial: bool = False


STRESS_TASKS: list[StressTask] = [
    StressTask(
        id="stress_01_multi_hop",
        name="Multi-hop reasoning",
        category="reasoning",
        description=(
            "If the capital of France starts with P, and the 16th letter "
            "of the alphabet is P, what number is that letter?"
        ),
        difficulty=3,
        expected_behavior="Should resolve chain: Paris -> P -> 16th letter -> answer 16",
        why_hard="Requires chaining 3 independent facts without losing track",
    ),
    StressTask(
        id="stress_02_ambiguous",
        name="Ambiguous instruction",
        category="ambiguity",
        description="Make it better",
        difficulty=4,
        expected_behavior=(
            "Should detect ambiguity and either ask for clarification "
            "or explain that the instruction lacks context"
        ),
        why_hard="No context, no subject -- tests graceful handling of vague input",
        accept_partial=True,
    ),
    StressTask(
        id="stress_03_tool_chain",
        name="3+ tool chain",
        category="tool_chain",
        description=(
            "Find the Python version installed on this system, then search "
            "the web for its release date, then save the result to a file "
            "called python_version_info.txt"
        ),
        difficulty=5,
        expected_behavior=(
            "Should decompose into 3+ subtasks: shell tool for version, "
            "web search for release date, file write for output"
        ),
        why_hard="Requires 3 different tools chained with data dependencies",
    ),
    StressTask(
        id="stress_04_dynamic_tool",
        name="Dynamic tool needed",
        category="tool_discovery",
        description="Calculate the SHA256 hash of the string 'graphbot'",
        difficulty=3,
        expected_behavior=(
            "Should use code execution or shell tool to compute the hash; "
            "expected output contains "
            "e3b7a0f3c29e05d08c973617c5c7a64d8b0e1d5c0a8f7e6b3d2c1a0f9e8d7c6 "
            "or invoke hashlib"
        ),
        why_hard="No dedicated hash tool -- must figure out how to compute it",
    ),
    StressTask(
        id="stress_05_deep_context",
        name="Deep graph context",
        category="meta_query",
        description=(
            "What tools has GraphBot used most successfully in the last 10 tasks?"
        ),
        difficulty=4,
        expected_behavior=(
            "Should query the knowledge graph for recent task history and "
            "tool usage stats, or explain that graph data is needed"
        ),
        why_hard="Requires querying internal state / knowledge graph, not external info",
        accept_partial=True,
    ),
    StressTask(
        id="stress_06_contradictory",
        name="Contradictory instruction",
        category="contradiction",
        description="Write a 5-word essay that is at least 500 words long",
        difficulty=5,
        expected_behavior=(
            "Should detect the contradiction (5 words vs 500 words) and "
            "report it rather than blindly attempting"
        ),
        why_hard="Logically impossible -- tests whether the system detects contradictions",
        accept_partial=True,
    ),
    StressTask(
        id="stress_07_multi_language",
        name="Multi-language translation",
        category="parallel",
        description="Translate 'hello' to 10 different languages simultaneously",
        difficulty=3,
        expected_behavior=(
            "Should decompose into parallel subtasks, one per language, "
            "and aggregate the translations"
        ),
        why_hard="Tests parallel decomposition breadth and result aggregation",
    ),
    StressTask(
        id="stress_08_recursive_decomp",
        name="Recursive decomposition stress",
        category="decomposition",
        description=(
            "Compare 5 programming languages across 5 dimensions with "
            "code examples for each"
        ),
        difficulty=5,
        expected_behavior=(
            "Should decompose into 25+ leaf tasks (5 languages x 5 dimensions), "
            "each producing a code example, then aggregate into a comparison"
        ),
        why_hard="Deep decomposition tree with 25+ leaves and complex aggregation",
    ),
    StressTask(
        id="stress_09_time_sensitive",
        name="Time-sensitive query",
        category="temporal",
        description="What time is it right now in Tokyo?",
        difficulty=2,
        expected_behavior=(
            "Should use a tool or code execution to get the current time "
            "in the Asia/Tokyo timezone"
        ),
        why_hard="Requires real-time data that LLM training data cannot provide",
    ),
    StressTask(
        id="stress_10_meta_reasoning",
        name="Meta-reasoning",
        category="meta",
        description=(
            "Explain step by step how you would solve this task, then "
            "solve it: What is 47 * 83?"
        ),
        difficulty=3,
        expected_behavior=(
            "Should produce both a reasoning trace and the correct answer (3901)"
        ),
        why_hard="Requires both meta-cognitive explanation and correct execution",
    ),
]


# ---------------------------------------------------------------------------
# Failure diagnosis
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    """Root cause analysis for a failed stress task."""

    category: str
    root_cause: str
    suggestion: str


def diagnose_failure(
    task: StressTask,
    result: ExecutionResult | None,
    error: Exception | None,
    elapsed_ms: float,
) -> DiagnosisResult:
    """Determine the root cause of a task failure.

    Inspects the result, exception, and timing to classify the failure
    into a specific category with an actionable suggestion.
    """
    # Timeout detection (>60 seconds is suspicious)
    if elapsed_ms > 60_000:
        return DiagnosisResult(
            category=FailureCategory.TIMEOUT,
            root_cause=f"Task took {elapsed_ms:.0f}ms (>{60_000}ms threshold)",
            suggestion="Add timeout limits to executor or reduce decomposition depth",
        )

    # Runtime exception
    if error is not None:
        error_str = str(error).lower()

        if "timeout" in error_str or "timed out" in error_str:
            return DiagnosisResult(
                category=FailureCategory.TIMEOUT,
                root_cause=f"Exception with timeout: {error}",
                suggestion="Increase timeout or simplify task decomposition",
            )

        if "tool" in error_str or "registry" in error_str:
            return DiagnosisResult(
                category=FailureCategory.TOOL_FAILURE,
                root_cause=f"Tool-related exception: {error}",
                suggestion="Check tool registry and ensure required tools are available",
            )

        return DiagnosisResult(
            category=FailureCategory.RUNTIME_ERROR,
            root_cause=f"Unhandled exception: {type(error).__name__}: {error}",
            suggestion="Add error handling for this exception type in the executor",
        )

    # Result-based diagnosis
    if result is not None:
        error_messages = " ".join(result.errors).lower() if result.errors else ""
        output_lower = result.output.lower() if result.output else ""

        # Safety blocked
        if "blocked" in error_messages or "safety" in error_messages:
            return DiagnosisResult(
                category=FailureCategory.SAFETY_BLOCKED,
                root_cause="Task was blocked by safety classifier",
                suggestion="Review safety rules -- this task should not be blocked",
            )

        if "refused" in output_lower or "cannot" in output_lower:
            return DiagnosisResult(
                category=FailureCategory.MODEL_REFUSAL,
                root_cause="Model refused or claimed inability to complete the task",
                suggestion="Improve prompt engineering or use a more capable model",
            )

        # Decomposition issues
        if task.category == "decomposition" and result.total_nodes < 3:
            return DiagnosisResult(
                category=FailureCategory.DECOMPOSITION_ERROR,
                root_cause=(
                    f"Expected deep decomposition but got only "
                    f"{result.total_nodes} node(s)"
                ),
                suggestion="Check decomposer prompts for complex multi-part tasks",
            )

        # Contradiction handling
        if task.category == "contradiction":
            return DiagnosisResult(
                category=FailureCategory.CONTRADICTION,
                root_cause="System did not detect the contradictory requirements",
                suggestion="Add contradiction detection to the intake parser",
            )

        # Context-dependent tasks
        if task.category == "meta_query":
            return DiagnosisResult(
                category=FailureCategory.CONTEXT_MISSING,
                root_cause="Task requires internal graph context that was unavailable",
                suggestion="Ensure graph query tools are accessible during execution",
            )

        # Tool chain failures
        if task.category == "tool_chain" and result.tools_used == 0:
            return DiagnosisResult(
                category=FailureCategory.TOOL_FAILURE,
                root_cause="No tools were invoked for a task requiring tool usage",
                suggestion="Check tool routing logic in the executor",
            )

    return DiagnosisResult(
        category=FailureCategory.UNKNOWN,
        root_cause="Could not determine a specific root cause",
        suggestion="Manual investigation required -- check logs for details",
    )


# ---------------------------------------------------------------------------
# Stress test result
# ---------------------------------------------------------------------------

@dataclass
class StressTestEntry:
    """Captured metrics for a single stress task run."""

    task_id: str
    name: str
    category: str
    difficulty: int
    description: str
    why_hard: str
    success: bool
    output_preview: str = ""
    total_nodes: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    model_used: str = ""
    tools_used: int = 0
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    exception: str = ""
    diagnosis: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "category": self.category,
            "difficulty": self.difficulty,
            "description": self.description,
            "why_hard": self.why_hard,
            "success": self.success,
            "output_preview": self.output_preview,
            "total_nodes": self.total_nodes,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms),
            "cost": self.cost,
            "model_used": self.model_used,
            "tools_used": self.tools_used,
            "llm_calls": self.llm_calls,
            "errors": self.errors,
            "exception": self.exception,
            "diagnosis": self.diagnosis,
        }


# ---------------------------------------------------------------------------
# Dry run support
# ---------------------------------------------------------------------------

def dry_run_report(tasks: list[StressTask]) -> dict[str, Any]:
    """Generate a report of tasks that would be executed without running them."""
    entries: list[dict[str, Any]] = []
    for task in tasks:
        entries.append({
            "task_id": task.id,
            "name": task.name,
            "category": task.category,
            "difficulty": task.difficulty,
            "description": task.description,
            "why_hard": task.why_hard,
            "expected_behavior": task.expected_behavior,
            "accept_partial": task.accept_partial,
            "status": "dry_run_skipped",
        })

    return {
        "mode": "dry_run",
        "total_tasks": len(tasks),
        "categories": list({t.category for t in tasks}),
        "avg_difficulty": sum(t.difficulty for t in tasks) / len(tasks),
        "tasks": entries,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_stress_test(dry_run: bool = False) -> None:
    """Execute all stress tasks and save results to benchmarks/stress_test.json."""
    project_root = Path(__file__).parent.parent
    out_file = project_root / "benchmarks" / "stress_test.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # --dry-run: list tasks and exit
    if dry_run:
        report = dry_run_report(STRESS_TASKS)
        out_file.write_text(json.dumps(report, indent=2))
        print(f"DRY RUN: {len(STRESS_TASKS)} tasks defined")
        print(f"Categories: {', '.join(report['categories'])}")
        print(f"Average difficulty: {report['avg_difficulty']:.1f}/5")
        print()
        for task in STRESS_TASKS:
            print(f"  [{task.id}] {task.name}")
            print(f"    Category: {task.category} | Difficulty: {task.difficulty}/5")
            print(f"    Description: {task.description[:80]}...")
            print(f"    Why hard: {task.why_hard}")
            print()
        print(f"Dry run report saved to {out_file}")
        return

    # Live run: set up orchestrator
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    db_path = project_root / "data" / "stress_test.db"
    store = GraphStore(str(db_path))
    store.initialize()
    orchestrator = Orchestrator(store, router)

    entries: list[StressTestEntry] = []
    total_start = time.perf_counter()

    for i, task in enumerate(STRESS_TASKS, 1):
        print(f"\n[{i}/{len(STRESS_TASKS)}] {task.id}: {task.name}")
        print(f"  Description: {task.description[:80]}...")
        print(f"  Difficulty: {task.difficulty}/5 | Category: {task.category}")

        result: ExecutionResult | None = None
        error: Exception | None = None
        start = time.perf_counter()

        try:
            result = await orchestrator.process(task.description)
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            error = exc
            logging.error(
                "Task %s raised %s: %s",
                task.id,
                type(exc).__name__,
                exc,
            )

        # Build entry
        entry = StressTestEntry(
            task_id=task.id,
            name=task.name,
            category=task.category,
            difficulty=task.difficulty,
            description=task.description,
            why_hard=task.why_hard,
            latency_ms=elapsed_ms,
            success=False,
        )

        if result is not None:
            entry.success = result.success
            entry.output_preview = result.output[:300] if result.output else ""
            entry.total_nodes = result.total_nodes
            entry.total_tokens = result.total_tokens
            entry.cost = result.total_cost
            entry.model_used = result.model_used
            entry.tools_used = result.tools_used
            entry.llm_calls = result.llm_calls
            entry.errors = list(result.errors) if result.errors else []

        if error is not None:
            entry.exception = f"{type(error).__name__}: {error}"

        # Diagnose failures
        if not entry.success:
            diagnosis = diagnose_failure(task, result, error, elapsed_ms)
            entry.diagnosis = {
                "category": diagnosis.category,
                "root_cause": diagnosis.root_cause,
                "suggestion": diagnosis.suggestion,
            }

        entries.append(entry)

        # Print result line
        status = "OK" if entry.success else "FAIL"
        print(
            f"  {status} | {entry.total_nodes} nodes | {entry.total_tokens} tok"
            f" | {elapsed_ms:.0f}ms | ${entry.cost:.6f}"
        )
        if not entry.success and entry.diagnosis:
            print(f"  Diagnosis: [{entry.diagnosis['category']}] {entry.diagnosis['root_cause']}")

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # Assemble output
    success_count = sum(1 for e in entries if e.success)
    fail_count = len(entries) - success_count
    total_tokens = sum(e.total_tokens for e in entries)
    total_cost = sum(e.cost for e in entries)

    # Failure breakdown by diagnosis category
    failure_breakdown: dict[str, int] = {}
    for e in entries:
        if not e.success and e.diagnosis:
            cat = e.diagnosis.get("category", "unknown")
            failure_breakdown[cat] = failure_breakdown.get(cat, 0) + 1

    output: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(entries),
        "success_count": success_count,
        "failure_count": fail_count,
        "success_rate": round(success_count / len(entries), 3) if entries else 0.0,
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "total_latency_ms": round(total_elapsed),
        "failure_breakdown": failure_breakdown,
        "tasks": [e.to_dict() for e in entries],
    }

    out_file.write_text(json.dumps(output, indent=2))

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"STRESS TEST SUMMARY ({len(entries)} tasks)")
    print(f"{'=' * 70}")
    print(f"Success rate:   {success_count}/{len(entries)} ({output['success_rate']:.1%})")
    print(f"Total tokens:   {total_tokens}")
    print(f"Total cost:     ${total_cost:.6f}")
    print(f"Total time:     {total_elapsed:.0f}ms")
    print()

    if failure_breakdown:
        print("Failure breakdown:")
        for cat, count in sorted(failure_breakdown.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
        print()

    # Per-task summary table
    print(f"{'ID':<30s} {'Status':<6s} {'Nodes':>5s} {'Tokens':>7s} {'Ms':>7s} {'Diagnosis'}")
    print("-" * 90)
    for e in entries:
        status = "OK" if e.success else "FAIL"
        diag = e.diagnosis.get("category", "") if not e.success else ""
        print(
            f"{e.task_id:<30s} {status:<6s} {e.total_nodes:>5d} "
            f"{e.total_tokens:>7d} {e.latency_ms:>7.0f} {diag}"
        )

    print(f"\nResults saved to {out_file}")
    store.close()


def main() -> None:
    """Entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="GraphBot stress test with 10 hard task definitions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all tasks and their metadata without executing them",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for detailed pipeline traces",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    asyncio.run(run_stress_test(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
