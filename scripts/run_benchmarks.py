"""Run all benchmark tasks through GraphBot and record results."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


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
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter


async def run_all() -> None:
    """Execute every benchmark task and save results to JSON."""
    # Load tasks
    tasks_file = Path(__file__).parent.parent / "benchmarks" / "tasks.json"
    tasks: list[dict] = json.loads(tasks_file.read_text())["tasks"]

    # Setup
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    store = GraphStore()
    store.initialize()
    orchestrator = Orchestrator(store, router)

    results: list[dict] = []

    for task in tasks:
        print(f"\n[{task['id']}] {task['description'][:60]}...")
        start = time.perf_counter()

        try:
            result = await orchestrator.process(task["description"])
            elapsed = (time.perf_counter() - start) * 1000

            # Check if expected keywords are in output
            answer_check = all(
                kw.lower() in result.output.lower()
                for kw in task.get("expected_answer_contains", [])
            )

            entry: dict = {
                "task_id": task["id"],
                "category": task["category"],
                "expected_behavior": task["expected_behavior"],
                "success": result.success,
                "answer_correct": answer_check,
                "nodes": result.total_nodes,
                "expected_min_nodes": task["expected_min_nodes"],
                "decomposition_worked": result.total_nodes >= task["expected_min_nodes"],
                "tokens": result.total_tokens,
                "latency_ms": round(elapsed),
                "cost": result.total_cost,
                "model": result.model_used,
                "output_preview": result.output[:200],
                "errors": list(result.errors) if result.errors else [],
            }
            results.append(entry)

            status = "OK" if result.success else "FAIL"
            decomp = f"{result.total_nodes} nodes" if result.total_nodes > 1 else "direct"
            print(
                f"  {status} | {decomp} | {result.total_tokens} tok"
                f" | {elapsed:.0f}ms | ${result.total_cost:.6f}"
            )

        except Exception as exc:
            results.append({
                "task_id": task["id"],
                "category": task["category"],
                "success": False,
                "error": str(exc),
            })
            print(f"  ERROR: {exc}")

    # Save results
    out_dir = Path(__file__).parent.parent / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{datetime.now().strftime('%Y-%m-%d')}-full.json"
    out_file.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY ({len(results)} tasks)")
    print(f"{'='*70}")

    success_count = sum(1 for r in results if r.get("success"))
    correct_count = sum(1 for r in results if r.get("answer_correct"))
    decomp_count = sum(1 for r in results if r.get("decomposition_worked"))
    multi_node_tasks = [t for t in tasks if t["expected_min_nodes"] > 1]
    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_cost = sum(r.get("cost", 0) for r in results)

    print(f"Success rate: {success_count}/{len(results)}")
    print(f"Answer correct: {correct_count}/{len(results)}")
    print(f"Decomposition worked: {decomp_count}/{len(multi_node_tasks)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"\nResults saved to {out_file}")

    store.close()


if __name__ == "__main__":
    asyncio.run(run_all())
