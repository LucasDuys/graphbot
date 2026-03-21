"""Live integration test -- runs real LLM calls through the full pipeline."""

import asyncio
import os
import sys
from pathlib import Path


def load_env_local() -> None:
    """Load .env.local file from project root."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if not env_file.exists():
        print(f"ERROR: {env_file} not found. Create it with OPENROUTER_API_KEY=sk-...")
        sys.exit(1)
    for line in env_file.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env_local()

from core_gb.executor import SimpleExecutor
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter


async def run_test(executor: SimpleExecutor, task: str, complexity: int = 1) -> None:
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"Complexity: {complexity}")
    print(f"{'='*60}")

    result = await executor.execute(task, complexity=complexity)

    status = "OK" if result.success else "FAIL"
    print(f"Status:   {status}")
    print(f"Model:    {result.model_used}")
    print(f"Output:   {result.output[:300]}")
    print(f"Tokens:   {result.total_tokens} (context: {result.context_tokens})")
    print(f"Latency:  {result.total_latency_ms:.0f}ms")
    print(f"Cost:     ${result.total_cost:.6f}")
    if result.errors:
        print(f"Errors:   {result.errors}")


async def main() -> None:
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    store = GraphStore()
    store.initialize()

    # Seed graph context
    store.create_node("User", {
        "id": "lucas", "name": "Lucas Duys", "role": "CSE student",
        "institution": "TU/e", "interests": "AI, knowledge graphs",
    })
    store.create_node("Project", {
        "id": "graphbot", "name": "GraphBot", "path": "C:/dev/graphbot",
        "language": "Python", "framework": "kuzu", "status": "phase1",
    })
    store.create_node("Memory", {
        "id": "mem1",
        "content": "GraphBot uses recursive DAG execution with temporal knowledge graph",
        "category": "architecture",
    })
    store.create_edge("OWNS", "lucas", "graphbot")

    executor = SimpleExecutor(store, router)

    # Test 1: Simple math (complexity 1 -- small model)
    await run_test(executor, "What is 247 * 38?", complexity=1)

    # Test 2: With graph context (complexity 1)
    await run_test(executor, "What do you know about Lucas and GraphBot?", complexity=1)

    # Test 3: Harder question (complexity 3 -- 70B model)
    await run_test(
        executor,
        "Explain in 2 sentences why a small model with perfect context "
        "might outperform a large model with no context.",
        complexity=3,
    )

    store.close()
    print("\n\nAll tests complete.")


if __name__ == "__main__":
    asyncio.run(main())
