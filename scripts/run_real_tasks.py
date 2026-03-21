"""Run real-world tool-using tasks through GraphBot and record results."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
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
    """Execute every real-world task and save results to JSON."""
    # Load tasks
    tasks_file = Path(__file__).parent.parent / "benchmarks" / "real_tasks.json"
    tasks: list[dict] = json.loads(tasks_file.read_text())["tasks"]

    # Setup with persistent graph
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    db_path = Path(__file__).parent.parent / "data" / "graphbot.db"
    store = GraphStore(str(db_path))
    store.initialize()
    orchestrator = Orchestrator(store, router)

    results: list[dict] = []

    for task in tasks:
        print(f"\n[{task['id']}] {task['description'][:60]}...")
        start = time.perf_counter()

        try:
            result = await orchestrator.process(task["description"])
            elapsed = (time.perf_counter() - start) * 1000

            # Detect whether a tool was used (model_used starts with "tool:")
            tool_used = result.model_used.startswith("tool:") if result.model_used else False

            entry: dict = {
                "task_id": task["id"],
                "category": task["category"],
                "expected_behavior": task["expected_behavior"],
                "tools_needed": task["tools_needed"],
                "difficulty": task["difficulty"],
                "success": result.success,
                "tool_used": tool_used,
                "model": result.model_used,
                "nodes": result.total_nodes,
                "tokens": result.total_tokens,
                "latency_ms": round(elapsed),
                "cost": result.total_cost,
                "output_preview": result.output[:300],
                "errors": list(result.errors) if result.errors else [],
            }
            results.append(entry)

            status = "OK" if result.success else "FAIL"
            tool_tag = "TOOL" if tool_used else "LLM"
            print(
                f"  {status} | {tool_tag} | {result.model_used}"
                f" | {result.total_tokens} tok | {elapsed:.0f}ms"
                f" | ${result.total_cost:.6f}"
            )

        except Exception as exc:
            results.append({
                "task_id": task["id"],
                "category": task["category"],
                "success": False,
                "tool_used": False,
                "error": str(exc),
            })
            print(f"  ERROR: {exc}")

    # Save results
    out_dir = Path(__file__).parent.parent / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{datetime.now().strftime('%Y-%m-%d')}-real.json"
    out_file.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*70}")
    print(f"REAL-WORLD TASK SUMMARY ({len(results)} tasks)")
    print(f"{'='*70}")

    success_count = sum(1 for r in results if r.get("success"))
    tool_count = sum(1 for r in results if r.get("tool_used"))
    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_cost = sum(r.get("cost", 0) for r in results)

    # Category breakdown
    categories: Counter[str] = Counter()
    category_success: Counter[str] = Counter()
    category_tool_used: Counter[str] = Counter()
    for r in results:
        cat = r.get("category", "unknown")
        categories[cat] += 1
        if r.get("success"):
            category_success[cat] += 1
        if r.get("tool_used"):
            category_tool_used[cat] += 1

    print(f"Success rate:     {success_count}/{len(results)}")
    print(f"Tool usage rate:  {tool_count}/{len(results)}")
    print(f"Total tokens:     {total_tokens}")
    print(f"Total cost:       ${total_cost:.6f}")

    print(f"\nBy category:")
    for cat in sorted(categories):
        total = categories[cat]
        ok = category_success[cat]
        tools = category_tool_used[cat]
        print(f"  {cat:10s}: {ok}/{total} success, {tools}/{total} used tools")

    # Model distribution
    model_counts: Counter[str] = Counter()
    for r in results:
        model = r.get("model", "unknown")
        model_counts[model] += 1

    print(f"\nModel distribution:")
    for model, count in model_counts.most_common():
        print(f"  {model}: {count}")

    print(f"\nResults saved to {out_file}")

    store.close()


if __name__ == "__main__":
    asyncio.run(run_all())
