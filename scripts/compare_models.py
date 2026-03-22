"""Model tier comparison benchmark.

Compares model tiers (free, mid, frontier, cheapest-frontier) in two modes:
  (A) Direct single-call to each model
  (B) Through the GraphBot pipeline

Measures output quality (placeholder 1-5), total tokens, cost, and latency.
Computes per-tier averages and token reduction percentages.
Saves results to benchmarks/model_comparison.json and prints a summary table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
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
from core_gb.types import CompletionResult, ExecutionResult
from graph.store import GraphStore
from models.base import ModelProvider
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Model tier definitions
# ---------------------------------------------------------------------------

MODEL_TIERS: dict[str, list[str]] = {
    "free": [
        "meta-llama/llama-3.1-8b-instruct",
    ],
    "mid": [
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-4o-mini",
    ],
    "frontier": [
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-6",
        "openai/gpt-4o",
    ],
}

CHEAPEST_FRONTIER_FALLBACK: str = "openai/gpt-4o-mini"

# ---------------------------------------------------------------------------
# Benchmark tasks (15+ diverse tasks)
# ---------------------------------------------------------------------------

BENCHMARK_TASKS: list[dict[str, Any]] = [
    {
        "id": "math_01",
        "description": "What is 247 * 38?",
        "category": "math",
        "expected_keywords": ["9386"],
    },
    {
        "id": "factual_01",
        "description": "What is the capital of France?",
        "category": "factual",
        "expected_keywords": ["Paris"],
    },
    {
        "id": "definition_01",
        "description": "Define the term 'recursion' in computer science.",
        "category": "definition",
        "expected_keywords": ["function", "itself"],
    },
    {
        "id": "conversion_01",
        "description": "Convert 72 degrees Fahrenheit to Celsius.",
        "category": "math",
        "expected_keywords": ["22"],
    },
    {
        "id": "comparison_01",
        "description": "Compare the pros and cons of PostgreSQL vs MongoDB.",
        "category": "comparison",
        "expected_keywords": ["PostgreSQL", "MongoDB"],
    },
    {
        "id": "explanation_01",
        "description": "Explain how a binary search tree works.",
        "category": "explanation",
        "expected_keywords": ["binary", "tree", "node"],
    },
    {
        "id": "coding_01",
        "description": "Write a Python function to check if a string is a palindrome.",
        "category": "coding",
        "expected_keywords": ["def", "palindrome"],
    },
    {
        "id": "reasoning_01",
        "description": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "category": "reasoning",
        "expected_keywords": ["cannot", "conclude"],
    },
    {
        "id": "summarize_01",
        "description": "Summarize the key differences between TCP and UDP protocols.",
        "category": "summarize",
        "expected_keywords": ["TCP", "UDP", "reliable"],
    },
    {
        "id": "creative_01",
        "description": "Write a haiku about programming.",
        "category": "creative",
        "expected_keywords": [],
    },
    {
        "id": "multi_step_01",
        "description": "First explain what Docker is, then describe how Docker Compose works, then explain when to use Kubernetes instead.",
        "category": "multi_step",
        "expected_keywords": ["Docker", "Compose", "Kubernetes"],
    },
    {
        "id": "analysis_01",
        "description": "Analyze the time complexity of quicksort in best, average, and worst cases.",
        "category": "analysis",
        "expected_keywords": ["O(n log n)", "O(n"],
    },
    {
        "id": "comparison_02",
        "description": "Compare React, Vue, and Svelte for building web applications.",
        "category": "comparison",
        "expected_keywords": ["React", "Vue", "Svelte"],
    },
    {
        "id": "factual_02",
        "description": "List the planets in our solar system in order from the Sun.",
        "category": "factual",
        "expected_keywords": ["Mercury", "Venus", "Earth", "Mars"],
    },
    {
        "id": "explanation_02",
        "description": "Explain the difference between supervised and unsupervised machine learning.",
        "category": "explanation",
        "expected_keywords": ["supervised", "unsupervised", "labeled"],
    },
    {
        "id": "coding_02",
        "description": "Write a Python function to find the nth Fibonacci number using dynamic programming.",
        "category": "coding",
        "expected_keywords": ["def", "fibonacci"],
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelRunResult:
    """Captured metrics from a single model run on a single task."""

    model: str
    tier: str
    mode: str  # "direct" or "pipeline"
    task_id: str
    output: str
    quality: int  # placeholder 1-5
    tokens: int
    cost: float
    latency_ms: float
    success: bool
    keyword_hits: int
    keyword_total: int


@dataclass
class TierSummary:
    """Aggregated metrics for a single tier."""

    tier: str
    avg_quality: float
    avg_tokens: float
    avg_cost: float
    avg_latency_ms: float
    total_tasks: int
    success_rate: float
    token_reduction_vs_frontier_pct: float


# ---------------------------------------------------------------------------
# Cheapest frontier auto-detection
# ---------------------------------------------------------------------------


async def detect_cheapest_frontier(provider: OpenRouterProvider) -> str:
    """Auto-detect the cheapest 'good' model on OpenRouter.

    Attempts to query the OpenRouter API for model pricing. Falls back to
    a known cheap frontier model if the API call fails.
    """
    try:
        import httpx

        api_key = provider._api_key
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return CHEAPEST_FRONTIER_FALLBACK

            data = resp.json().get("data", [])
            # Filter for models that are reasonably capable (frontier-quality)
            frontier_keywords: list[str] = [
                "gpt-4o", "claude-3", "claude-sonnet", "gemini-pro",
                "gemini-2", "llama-3.3-70b",
            ]
            candidates: list[tuple[float, str]] = []
            for model_info in data:
                model_id: str = model_info.get("id", "")
                pricing = model_info.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "999"))
                completion_price = float(pricing.get("completion", "999"))
                total_price = prompt_price + completion_price

                if any(kw in model_id.lower() for kw in frontier_keywords):
                    candidates.append((total_price, model_id))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]

    except Exception:
        pass

    return CHEAPEST_FRONTIER_FALLBACK


# ---------------------------------------------------------------------------
# Quality scoring (placeholder heuristic)
# ---------------------------------------------------------------------------


def score_quality(
    output: str,
    expected_keywords: list[str],
) -> int:
    """Score output quality on a 1-5 scale (placeholder heuristic).

    Scoring rubric:
      1 = empty or failed output
      2 = output present but no keyword matches
      3 = some keyword matches (< 50%)
      4 = most keyword matches (>= 50%)
      5 = all keywords matched or no keywords expected and output is substantial
    """
    if not output or len(output.strip()) == 0:
        return 1

    if not expected_keywords:
        # No keywords to check; score based on output length
        if len(output) >= 100:
            return 5
        if len(output) >= 30:
            return 4
        return 3

    hits = sum(
        1 for kw in expected_keywords if kw.lower() in output.lower()
    )
    ratio = hits / len(expected_keywords)

    if ratio >= 1.0:
        return 5
    if ratio >= 0.5:
        return 4
    if ratio > 0.0:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Direct single-call execution
# ---------------------------------------------------------------------------


async def run_direct_call(
    provider: ModelProvider,
    model: str,
    task_description: str,
) -> CompletionResult:
    """Execute a single direct LLM call for a given model."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task_description},
    ]
    return await provider.complete(messages, model)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


async def run_task_benchmark(
    task: dict[str, Any],
    provider: ModelProvider,
    orchestrator: Orchestrator,
    tier_name: str,
    model: str,
) -> list[ModelRunResult]:
    """Run both modes (direct + pipeline) for a single task and model.

    Returns two ModelRunResult entries: one for direct, one for pipeline.
    """
    results: list[ModelRunResult] = []
    task_id: str = task["id"]
    description: str = task["description"]
    expected_keywords: list[str] = task.get("expected_keywords", [])

    # Mode A: Direct single-call
    start = time.perf_counter()
    try:
        completion = await run_direct_call(provider, model, description)
        latency = (time.perf_counter() - start) * 1000
        output = completion.content
        tokens = completion.tokens_in + completion.tokens_out
        cost = completion.cost
        success = True
    except Exception:
        latency = (time.perf_counter() - start) * 1000
        output = ""
        tokens = 0
        cost = 0.0
        success = False

    keyword_hits = sum(
        1 for kw in expected_keywords if kw.lower() in output.lower()
    )
    quality = score_quality(output, expected_keywords)

    results.append(ModelRunResult(
        model=model,
        tier=tier_name,
        mode="direct",
        task_id=task_id,
        output=output,
        quality=quality,
        tokens=tokens,
        cost=cost,
        latency_ms=latency,
        success=success,
        keyword_hits=keyword_hits,
        keyword_total=len(expected_keywords),
    ))

    # Mode B: Through GraphBot pipeline
    start = time.perf_counter()
    try:
        exec_result: ExecutionResult = await orchestrator.process(description)
        latency = (time.perf_counter() - start) * 1000
        output = exec_result.output
        tokens = exec_result.total_tokens
        cost = exec_result.total_cost
        success = exec_result.success
    except Exception:
        latency = (time.perf_counter() - start) * 1000
        output = ""
        tokens = 0
        cost = 0.0
        success = False

    keyword_hits = sum(
        1 for kw in expected_keywords if kw.lower() in output.lower()
    )
    quality = score_quality(output, expected_keywords)

    results.append(ModelRunResult(
        model=model,
        tier=tier_name,
        mode="pipeline",
        task_id=task_id,
        output=output,
        quality=quality,
        tokens=tokens,
        cost=cost,
        latency_ms=latency,
        success=success,
        keyword_hits=keyword_hits,
        keyword_total=len(expected_keywords),
    ))

    return results


async def run_full_benchmark(
    tasks: list[dict[str, Any]],
    tiers: dict[str, list[str]],
    provider: ModelProvider,
    orchestrator: Orchestrator,
) -> list[ModelRunResult]:
    """Run all tasks across all tiers and modes. Returns flat list of results."""
    all_results: list[ModelRunResult] = []

    for tier_name, models in tiers.items():
        for model in models:
            print(f"\n--- Tier: {tier_name} | Model: {model} ---")
            for i, task in enumerate(tasks, 1):
                print(
                    f"  [{i}/{len(tasks)}] {task['id']}: "
                    f"{task['description'][:50]}..."
                )
                task_results = await run_task_benchmark(
                    task, provider, orchestrator, tier_name, model,
                )
                all_results.extend(task_results)

                for r in task_results:
                    status = "OK" if r.success else "FAIL"
                    print(
                        f"    {r.mode}: {status} | q={r.quality} | "
                        f"{r.tokens} tok | ${r.cost:.6f} | "
                        f"{r.latency_ms:.0f}ms"
                    )

    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_tier_summaries(
    results: list[ModelRunResult],
    mode: str,
) -> dict[str, TierSummary]:
    """Compute per-tier averages for a given mode (direct or pipeline)."""
    by_tier: dict[str, list[ModelRunResult]] = {}
    for r in results:
        if r.mode != mode:
            continue
        by_tier.setdefault(r.tier, []).append(r)

    summaries: dict[str, TierSummary] = {}

    # Compute frontier avg tokens for reduction calculation
    frontier_results = by_tier.get("frontier", [])
    frontier_avg_tokens = (
        sum(r.tokens for r in frontier_results) / len(frontier_results)
        if frontier_results
        else 0.0
    )

    for tier_name, tier_results in by_tier.items():
        count = len(tier_results)
        if count == 0:
            continue

        avg_quality = sum(r.quality for r in tier_results) / count
        avg_tokens = sum(r.tokens for r in tier_results) / count
        avg_cost = sum(r.cost for r in tier_results) / count
        avg_latency = sum(r.latency_ms for r in tier_results) / count
        success_count = sum(1 for r in tier_results if r.success)
        success_rate = success_count / count

        if frontier_avg_tokens > 0 and tier_name != "frontier":
            token_reduction = (
                (frontier_avg_tokens - avg_tokens) / frontier_avg_tokens * 100
            )
        else:
            token_reduction = 0.0

        summaries[tier_name] = TierSummary(
            tier=tier_name,
            avg_quality=avg_quality,
            avg_tokens=avg_tokens,
            avg_cost=avg_cost,
            avg_latency_ms=avg_latency,
            total_tasks=count,
            success_rate=success_rate,
            token_reduction_vs_frontier_pct=token_reduction,
        )

    return summaries


def compute_pipeline_token_reduction(
    results: list[ModelRunResult],
) -> dict[str, float]:
    """Compute token reduction % of pipeline vs direct for each tier."""
    reductions: dict[str, float] = {}

    tiers_seen: set[str] = {r.tier for r in results}
    for tier in tiers_seen:
        direct = [r for r in results if r.tier == tier and r.mode == "direct"]
        pipeline = [r for r in results if r.tier == tier and r.mode == "pipeline"]

        if not direct or not pipeline:
            reductions[tier] = 0.0
            continue

        avg_direct = sum(r.tokens for r in direct) / len(direct)
        avg_pipeline = sum(r.tokens for r in pipeline) / len(pipeline)

        if avg_direct > 0:
            reductions[tier] = (avg_direct - avg_pipeline) / avg_direct * 100
        else:
            reductions[tier] = 0.0

    return reductions


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_summary_table(
    direct_summaries: dict[str, TierSummary],
    pipeline_summaries: dict[str, TierSummary],
    pipeline_reductions: dict[str, float],
) -> str:
    """Format a summary table comparing all tiers and modes."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"Model Tier Comparison Benchmark -- {now}")
    lines.append("=" * 80)

    # Direct mode table
    lines.append("")
    lines.append("MODE A: Direct Single-Call")
    lines.append("-" * 80)
    lines.append(
        f"{'Tier':<20} {'Avg Quality':>11} {'Avg Tokens':>10} "
        f"{'Avg Cost':>10} {'Avg Latency':>11} {'Success':>8} "
        f"{'Tok Reduction':>14}"
    )
    lines.append("-" * 80)

    tier_order: list[str] = ["free", "mid", "frontier", "cheapest-frontier"]
    for tier in tier_order:
        s = direct_summaries.get(tier)
        if s is None:
            continue
        lines.append(
            f"{tier:<20} {s.avg_quality:>11.2f} {s.avg_tokens:>10.0f} "
            f"${s.avg_cost:>9.6f} {s.avg_latency_ms:>10.0f}ms "
            f"{s.success_rate:>7.0%} "
            f"{s.token_reduction_vs_frontier_pct:>13.1f}%"
        )

    # Pipeline mode table
    lines.append("")
    lines.append("MODE B: GraphBot Pipeline")
    lines.append("-" * 80)
    lines.append(
        f"{'Tier':<20} {'Avg Quality':>11} {'Avg Tokens':>10} "
        f"{'Avg Cost':>10} {'Avg Latency':>11} {'Success':>8} "
        f"{'Tok Reduction':>14}"
    )
    lines.append("-" * 80)

    for tier in tier_order:
        s = pipeline_summaries.get(tier)
        if s is None:
            continue
        lines.append(
            f"{tier:<20} {s.avg_quality:>11.2f} {s.avg_tokens:>10.0f} "
            f"${s.avg_cost:>9.6f} {s.avg_latency_ms:>10.0f}ms "
            f"{s.success_rate:>7.0%} "
            f"{s.token_reduction_vs_frontier_pct:>13.1f}%"
        )

    # Pipeline vs direct token reduction
    lines.append("")
    lines.append("Pipeline vs Direct Token Reduction")
    lines.append("-" * 40)
    for tier in tier_order:
        reduction = pipeline_reductions.get(tier)
        if reduction is None:
            continue
        lines.append(f"  {tier:<20} {reduction:>10.1f}%")

    lines.append("")
    return "\n".join(lines)


def results_to_json(
    results: list[ModelRunResult],
    direct_summaries: dict[str, TierSummary],
    pipeline_summaries: dict[str, TierSummary],
    pipeline_reductions: dict[str, float],
) -> dict[str, Any]:
    """Convert all results and summaries to a JSON-serializable dict."""
    return {
        "timestamp": datetime.now().isoformat(),
        "task_count": len(BENCHMARK_TASKS),
        "results": [
            {
                "model": r.model,
                "tier": r.tier,
                "mode": r.mode,
                "task_id": r.task_id,
                "quality": r.quality,
                "tokens": r.tokens,
                "cost": r.cost,
                "latency_ms": round(r.latency_ms, 2),
                "success": r.success,
                "keyword_hits": r.keyword_hits,
                "keyword_total": r.keyword_total,
            }
            for r in results
        ],
        "tier_summaries": {
            "direct": {
                tier: {
                    "avg_quality": round(s.avg_quality, 2),
                    "avg_tokens": round(s.avg_tokens, 1),
                    "avg_cost": round(s.avg_cost, 6),
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                    "total_tasks": s.total_tasks,
                    "success_rate": round(s.success_rate, 3),
                    "token_reduction_vs_frontier_pct": round(
                        s.token_reduction_vs_frontier_pct, 1
                    ),
                }
                for tier, s in direct_summaries.items()
            },
            "pipeline": {
                tier: {
                    "avg_quality": round(s.avg_quality, 2),
                    "avg_tokens": round(s.avg_tokens, 1),
                    "avg_cost": round(s.avg_cost, 6),
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                    "total_tasks": s.total_tasks,
                    "success_rate": round(s.success_rate, 3),
                    "token_reduction_vs_frontier_pct": round(
                        s.token_reduction_vs_frontier_pct, 1
                    ),
                }
                for tier, s in pipeline_summaries.items()
            },
        },
        "pipeline_vs_direct_token_reduction": {
            tier: round(v, 1) for tier, v in pipeline_reductions.items()
        },
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


async def run_benchmark(
    tasks: list[dict[str, Any]] | None = None,
    include_cheapest_frontier: bool = True,
) -> None:
    """Run the full model tier comparison benchmark."""
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    store = GraphStore()
    store.initialize()

    orchestrator = Orchestrator(store, router)

    tiers = dict(MODEL_TIERS)

    if include_cheapest_frontier:
        cheapest = await detect_cheapest_frontier(provider)
        tiers["cheapest-frontier"] = [cheapest]
        print(f"Cheapest frontier model detected: {cheapest}")

    benchmark_tasks = tasks or BENCHMARK_TASKS

    try:
        all_results = await run_full_benchmark(
            benchmark_tasks, tiers, provider, orchestrator,
        )

        # Compute summaries
        direct_summaries = compute_tier_summaries(all_results, "direct")
        pipeline_summaries = compute_tier_summaries(all_results, "pipeline")
        pipeline_reductions = compute_pipeline_token_reduction(all_results)

        # Print summary table
        table = format_summary_table(
            direct_summaries, pipeline_summaries, pipeline_reductions,
        )
        print("\n" + table)

        # Save to JSON
        out_dir = Path(__file__).parent.parent / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "model_comparison.json"

        json_data = results_to_json(
            all_results, direct_summaries, pipeline_summaries,
            pipeline_reductions,
        )
        out_file.write_text(json.dumps(json_data, indent=2))

        print(f"\nResults saved to {out_file}")
    finally:
        store.close()


def main() -> None:
    """Parse arguments and run the model tier comparison benchmark."""
    parser = argparse.ArgumentParser(
        description="Model tier comparison benchmark: free vs mid vs frontier"
    )
    parser.add_argument(
        "--no-cheapest-frontier",
        action="store_true",
        help="Skip auto-detection of cheapest frontier model",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Run a single task by ID (e.g. math_01)",
    )
    args = parser.parse_args()

    tasks: list[dict[str, Any]] | None = None
    if args.task:
        matching = [t for t in BENCHMARK_TASKS if t["id"] == args.task]
        if not matching:
            print(f"Unknown task ID: {args.task}")
            print(f"Available: {', '.join(t['id'] for t in BENCHMARK_TASKS)}")
            sys.exit(1)
        tasks = matching

    asyncio.run(
        run_benchmark(
            tasks=tasks,
            include_cheapest_frontier=not args.no_cheapest_frontier,
        )
    )


if __name__ == "__main__":
    main()
