"""A/B comparison: GraphBot pipeline vs single LLM call."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path


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
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter


async def run_comparison(task: str) -> None:
    """Run a side-by-side comparison of full pipeline vs single LLM call."""
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    store = GraphStore()
    store.initialize()

    orchestrator = Orchestrator(store, router)
    executor = SimpleExecutor(store, router)

    # Run A: GraphBot full pipeline
    start_a = time.perf_counter()
    result_a = await orchestrator.process(task)
    time_a = (time.perf_counter() - start_a) * 1000

    # Run B: Single LLM call (baseline)
    start_b = time.perf_counter()
    result_b = await executor.execute(task, complexity=3)
    time_b = (time.perf_counter() - start_b) * 1000

    # Generate report
    report = f"""# A/B Comparison Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Task:** {task}

## GraphBot (full pipeline)
- **Output:** {result_a.output[:500]}
- **Nodes:** {result_a.total_nodes}
- **Tokens:** {result_a.total_tokens}
- **Latency:** {time_a:.0f}ms
- **Cost:** ${result_a.total_cost:.6f}
- **Success:** {result_a.success}

## Baseline (single LLM call, 70B model)
- **Output:** {result_b.output[:500]}
- **Tokens:** {result_b.total_tokens}
- **Latency:** {time_b:.0f}ms
- **Cost:** ${result_b.total_cost:.6f}
- **Success:** {result_b.success}

## Comparison
| Metric | GraphBot | Baseline | Winner |
|--------|----------|----------|--------|
| Tokens | {result_a.total_tokens} | {result_b.total_tokens} | {'GraphBot' if result_a.total_tokens < result_b.total_tokens else 'Baseline'} |
| Latency | {time_a:.0f}ms | {time_b:.0f}ms | {'GraphBot' if time_a < time_b else 'Baseline'} |
| Cost | ${result_a.total_cost:.6f} | ${result_b.total_cost:.6f} | {'GraphBot' if result_a.total_cost <= result_b.total_cost else 'Baseline'} |
| Nodes | {result_a.total_nodes} | 1 | - |
"""

    # Save report
    out_dir = Path(__file__).parent.parent / "benchmarks" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = task[:40].lower().replace(" ", "-").replace(",", "")
    filename = f"{datetime.now().strftime('%Y-%m-%d')}-{slug}.md"
    (out_dir / filename).write_text(report)

    print(report)
    print(f"\nReport saved to benchmarks/comparisons/{filename}")

    store.close()


if __name__ == "__main__":
    task = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the weather in Amsterdam, London, and Berlin?"
    )
    asyncio.run(run_comparison(task))
