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

from core_gb.orchestrator import Orchestrator
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter


async def run_test(orchestrator: Orchestrator, message: str) -> None:
    print(f"\n{'='*60}")
    print(f"Message: {message}")
    print(f"{'='*60}")

    result = await orchestrator.process(message)

    status = "OK" if result.success else "FAIL"
    print(f"Status:   {status}")
    print(f"Model:    {result.model_used}")
    print(f"Nodes:    {result.total_nodes}")
    print(f"Output:   {result.output[:400]}")
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
        "language": "Python", "framework": "kuzu", "status": "phase2",
    })
    store.create_node("Memory", {
        "id": "mem1",
        "content": "GraphBot uses recursive DAG execution with temporal knowledge graph",
        "category": "architecture",
    })
    store.create_edge("OWNS", "lucas", "graphbot")

    orchestrator = Orchestrator(store, router)

    # Test 1: Simple task (should skip decomposition)
    await run_test(orchestrator, "What is 247 * 38?")

    # Test 2: Simple task with context
    await run_test(orchestrator, "What do you know about Lucas?")

    # Test 3: Complex task (should trigger decomposition)
    await run_test(
        orchestrator,
        "Compare the weather in Amsterdam, London, and Berlin"
    )

    store.close()
    print("\n\nAll tests complete.")


if __name__ == "__main__":
    asyncio.run(main())
