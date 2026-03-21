"""3-way comparison: GraphBot pipeline vs 8B model vs 70B model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def load_env() -> None:
    """Load environment variables from .env.local if present."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


load_env()

from core_gb.orchestrator import Orchestrator
from core_gb.executor import SimpleExecutor
from core_gb.types import ExecutionResult
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter


@dataclass
class RunResult:
    """Captured metrics from a single comparison run."""

    label: str
    output: str
    tokens: int
    latency_ms: float
    cost: float
    success: bool
    total_nodes: int
    output_length: int
    keyword_hits: int
    keyword_total: int


def compute_quality(
    result: ExecutionResult,
    expected_keywords: list[str] | None = None,
) -> tuple[int, int, int]:
    """Return (output_length, keyword_hits, keyword_total).

    Quality heuristic based on output length and keyword coverage.
    """
    output_length = len(result.output)
    keywords = expected_keywords or []
    keyword_total = len(keywords)
    keyword_hits = sum(
        1 for kw in keywords if kw.lower() in result.output.lower()
    )
    return output_length, keyword_hits, keyword_total


async def run_single_comparison(
    task: str,
    orchestrator: Orchestrator,
    executor: SimpleExecutor,
    expected_keywords: list[str] | None = None,
) -> list[RunResult]:
    """Run 3-way comparison for a single task.

    A) GraphBot full pipeline (Orchestrator.process)
    B) Single 8B model call (SimpleExecutor with complexity=1)
    C) Single 70B model call (SimpleExecutor with complexity=3)
    """
    results: list[RunResult] = []

    configs: list[tuple[str, str]] = [
        ("A_graphbot", "GraphBot Full Pipeline"),
        ("B_8b_model", "Single 8B Model"),
        ("C_70b_model", "Single 70B Model"),
    ]

    for config_key, label in configs:
        start = time.perf_counter()

        if config_key == "A_graphbot":
            result = await orchestrator.process(task)
        elif config_key == "B_8b_model":
            result = await executor.execute(task, complexity=1)
        else:
            result = await executor.execute(task, complexity=3)

        latency = (time.perf_counter() - start) * 1000
        output_length, keyword_hits, keyword_total = compute_quality(
            result, expected_keywords
        )

        results.append(RunResult(
            label=label,
            output=result.output,
            tokens=result.total_tokens,
            latency_ms=latency,
            cost=result.total_cost,
            success=result.success,
            total_nodes=result.total_nodes,
            output_length=output_length,
            keyword_hits=keyword_hits,
            keyword_total=keyword_total,
        ))

    return results


def pick_winner(values: list[float], lower_is_better: bool = True) -> int:
    """Return the index of the winning value."""
    if lower_is_better:
        return values.index(min(values))
    return values.index(max(values))


def format_individual_report(
    task: str,
    results: list[RunResult],
) -> str:
    """Format a comparison report for a single task."""
    labels = [r.label for r in results]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = [
        f"# 3-Way Comparison Report",
        f"**Date:** {now}",
        f"**Task:** {task}",
        "",
    ]

    for r in results:
        keyword_str = (
            f"{r.keyword_hits}/{r.keyword_total}"
            if r.keyword_total > 0
            else "N/A"
        )
        lines.extend([
            f"## {r.label}",
            f"- **Output:** {r.output[:500]}",
            f"- **Nodes:** {r.total_nodes}",
            f"- **Tokens:** {r.tokens}",
            f"- **Latency:** {r.latency_ms:.0f}ms",
            f"- **Cost:** ${r.cost:.6f}",
            f"- **Success:** {r.success}",
            f"- **Output Length:** {r.output_length} chars",
            f"- **Keyword Coverage:** {keyword_str}",
            "",
        ])

    # Comparison table
    token_winner = pick_winner([float(r.tokens) for r in results])
    latency_winner = pick_winner([r.latency_ms for r in results])
    cost_winner = pick_winner([r.cost for r in results])
    length_winner = pick_winner(
        [float(r.output_length) for r in results], lower_is_better=False
    )

    lines.extend([
        "## Comparison",
        "| Metric | GraphBot | 8B Model | 70B Model | Winner |",
        "|--------|----------|----------|-----------|--------|",
        f"| Tokens | {results[0].tokens} | {results[1].tokens} | {results[2].tokens} | {labels[token_winner]} |",
        f"| Latency | {results[0].latency_ms:.0f}ms | {results[1].latency_ms:.0f}ms | {results[2].latency_ms:.0f}ms | {labels[latency_winner]} |",
        f"| Cost | ${results[0].cost:.6f} | ${results[1].cost:.6f} | ${results[2].cost:.6f} | {labels[cost_winner]} |",
        f"| Output Length | {results[0].output_length} | {results[1].output_length} | {results[2].output_length} | {labels[length_winner]} |",
        f"| Nodes | {results[0].total_nodes} | {results[1].total_nodes} | {results[2].total_nodes} | - |",
    ])

    if results[0].keyword_total > 0:
        kw_winner = pick_winner(
            [float(r.keyword_hits) for r in results], lower_is_better=False
        )
        lines.append(
            f"| Keywords | {results[0].keyword_hits}/{results[0].keyword_total} "
            f"| {results[1].keyword_hits}/{results[1].keyword_total} "
            f"| {results[2].keyword_hits}/{results[2].keyword_total} "
            f"| {labels[kw_winner]} |"
        )

    lines.append("")
    return "\n".join(lines)


def format_aggregate_report(
    all_results: dict[str, list[RunResult]],
) -> str:
    """Format an aggregate report across all tasks."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    task_count = len(all_results)

    # Accumulate totals per config (index 0=GraphBot, 1=8B, 2=70B)
    totals: dict[str, dict[str, float]] = {
        "GraphBot Full Pipeline": {"tokens": 0, "latency": 0, "cost": 0, "length": 0, "kw_hits": 0, "kw_total": 0, "success": 0},
        "Single 8B Model": {"tokens": 0, "latency": 0, "cost": 0, "length": 0, "kw_hits": 0, "kw_total": 0, "success": 0},
        "Single 70B Model": {"tokens": 0, "latency": 0, "cost": 0, "length": 0, "kw_hits": 0, "kw_total": 0, "success": 0},
    }

    for task_desc, results in all_results.items():
        for r in results:
            t = totals[r.label]
            t["tokens"] += r.tokens
            t["latency"] += r.latency_ms
            t["cost"] += r.cost
            t["length"] += r.output_length
            t["kw_hits"] += r.keyword_hits
            t["kw_total"] += r.keyword_total
            t["success"] += 1 if r.success else 0

    labels = list(totals.keys())
    lines: list[str] = [
        f"# Aggregate 3-Way Comparison Report",
        f"**Date:** {now}",
        f"**Tasks:** {task_count}",
        "",
        "## Totals",
        "| Metric | GraphBot | 8B Model | 70B Model |",
        "|--------|----------|----------|-----------|",
    ]

    for metric_label, key in [
        ("Total Tokens", "tokens"),
        ("Total Latency", "latency"),
        ("Total Cost", "cost"),
        ("Total Output Length", "length"),
        ("Keyword Hits", "kw_hits"),
        ("Successes", "success"),
    ]:
        vals = [totals[l][key] for l in labels]
        if key == "cost":
            formatted = [f"${v:.6f}" for v in vals]
        elif key == "latency":
            formatted = [f"{v:.0f}ms" for v in vals]
        elif key == "kw_hits":
            formatted = [
                f"{int(totals[l]['kw_hits'])}/{int(totals[l]['kw_total'])}"
                for l in labels
            ]
        else:
            formatted = [f"{int(v)}" for v in vals]
        lines.append(
            f"| {metric_label} | {formatted[0]} | {formatted[1]} | {formatted[2]} |"
        )

    lines.extend(["", "## Averages (per task)"])
    lines.append("| Metric | GraphBot | 8B Model | 70B Model |")
    lines.append("|--------|----------|----------|-----------|")

    for metric_label, key in [
        ("Avg Tokens", "tokens"),
        ("Avg Latency", "latency"),
        ("Avg Cost", "cost"),
        ("Avg Output Length", "length"),
    ]:
        vals = [totals[l][key] / task_count for l in labels]
        if key == "cost":
            formatted = [f"${v:.6f}" for v in vals]
        elif key == "latency":
            formatted = [f"{v:.0f}ms" for v in vals]
        else:
            formatted = [f"{int(v)}" for v in vals]
        lines.append(
            f"| {metric_label} | {formatted[0]} | {formatted[1]} | {formatted[2]} |"
        )

    lines.append("")
    return "\n".join(lines)


async def run_comparison(task: str) -> None:
    """Run a 3-way comparison for a single task."""
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    store = GraphStore()
    store.initialize()

    orchestrator = Orchestrator(store, router)
    executor = SimpleExecutor(store, router)

    try:
        results = await run_single_comparison(task, orchestrator, executor)
        report = format_individual_report(task, results)

        out_dir = Path(__file__).parent.parent / "benchmarks" / "comparisons"
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = task[:40].lower().replace(" ", "-").replace(",", "")
        filename = f"{datetime.now().strftime('%Y-%m-%d')}-{slug}.md"
        (out_dir / filename).write_text(report)

        print(report)
        print(f"\nReport saved to benchmarks/comparisons/{filename}")
    finally:
        store.close()


async def run_all_comparisons() -> None:
    """Run 3-way comparison for all 15 benchmark tasks and generate aggregate report."""
    tasks_file = Path(__file__).parent.parent / "benchmarks" / "tasks.json"
    tasks: list[dict[str, Any]] = json.loads(tasks_file.read_text())["tasks"]

    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    store = GraphStore()
    store.initialize()

    orchestrator = Orchestrator(store, router)
    executor = SimpleExecutor(store, router)

    out_dir = Path(__file__).parent.parent / "benchmarks" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[RunResult]] = {}

    try:
        for i, task in enumerate(tasks, 1):
            desc = task["description"]
            expected_keywords = task.get("expected_answer_contains")
            print(f"\n[{i}/{len(tasks)}] {task['id']}: {desc[:60]}...")

            results = await run_single_comparison(
                desc, orchestrator, executor, expected_keywords
            )
            all_results[desc] = results

            # Save individual report
            individual_report = format_individual_report(desc, results)
            slug = task["id"]
            filename = f"{datetime.now().strftime('%Y-%m-%d')}-{slug}.md"
            (out_dir / filename).write_text(individual_report)

            # Print summary line
            for r in results:
                kw_str = (
                    f" kw={r.keyword_hits}/{r.keyword_total}"
                    if r.keyword_total > 0
                    else ""
                )
                status = "OK" if r.success else "FAIL"
                print(
                    f"  {r.label}: {status} | {r.tokens} tok | "
                    f"{r.latency_ms:.0f}ms | ${r.cost:.6f}{kw_str}"
                )

        # Generate and save aggregate report
        aggregate = format_aggregate_report(all_results)
        agg_filename = f"{datetime.now().strftime('%Y-%m-%d')}-aggregate.md"
        (out_dir / agg_filename).write_text(aggregate)

        print("\n" + aggregate)
        print(f"\nAggregate report saved to benchmarks/comparisons/{agg_filename}")
        print(f"Individual reports saved to benchmarks/comparisons/")
    finally:
        store.close()


def main() -> None:
    """Parse arguments and run comparisons."""
    parser = argparse.ArgumentParser(
        description="3-way comparison: GraphBot vs 8B model vs 70B model"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task",
        type=str,
        help="Single task description to compare",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run comparison on all 15 benchmark tasks from tasks.json",
    )
    args = parser.parse_args()

    if args.all:
        asyncio.run(run_all_comparisons())
    else:
        asyncio.run(run_comparison(args.task))


if __name__ == "__main__":
    main()
