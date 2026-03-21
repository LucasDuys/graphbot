"""Interactive CLI chat with GraphBot."""

import asyncio
import os
import sys
from pathlib import Path


def load_env() -> None:
    """Load .env.local from project root into environment."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


load_env()

from core_gb.orchestrator import Orchestrator  # noqa: E402
from graph.store import GraphStore  # noqa: E402
from models.openrouter import OpenRouterProvider  # noqa: E402
from models.router import ModelRouter  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
GRAPH_PATH = DATA_DIR / "graphbot.db"
HISTORY: list[dict] = []


def print_header() -> None:
    """Print the CLI welcome banner."""
    print("=" * 50)
    print("  GraphBot CLI")
    print("  Type a message to chat. Commands:")
    print("    /stats  -- show knowledge graph statistics")
    print("    /history -- show last 10 interactions")
    print("    /quit   -- exit")
    print("=" * 50)
    print()


def print_stats(store: GraphStore) -> None:
    """Print knowledge graph statistics."""
    tables = ["User", "Project", "Service", "Memory", "Task", "PatternNode", "ExecutionTree"]
    print("\n--- Graph Statistics ---")
    total = 0
    for table in tables:
        try:
            rows = store.query(f"MATCH (n:{table}) RETURN count(n) AS cnt")
            count = rows[0]["cnt"] if rows else 0
            if count > 0:
                print(f"  {table:20s}: {count}")
            total += count
        except Exception:
            pass
    print(f"  {'Total':20s}: {total}")

    # Pattern count
    try:
        patterns = store.query("MATCH (p:PatternNode) RETURN count(p) AS cnt")
        pcnt = patterns[0]["cnt"] if patterns else 0
        print(f"  Cached patterns: {pcnt}")
    except Exception:
        pass
    print()


def print_history() -> None:
    """Print last 10 interactions."""
    if not HISTORY:
        print("\nNo history yet.\n")
        return
    print("\n--- Last 10 Interactions ---")
    for entry in HISTORY[-10:]:
        print(f"  > {entry['message'][:60]}")
        print(f"    {entry['nodes']} nodes | {entry['tokens']} tok | ${entry['cost']:.6f}")
    print()


async def main() -> None:
    """Run the interactive chat loop."""
    print_header()

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set. Create .env.local with your key.")
        sys.exit(1)

    # Set up persistent graph
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = GraphStore(str(GRAPH_PATH))
    store.initialize()

    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    orchestrator = Orchestrator(store, router)

    print("Ready. Knowledge graph at:", GRAPH_PATH)
    print()

    try:
        while True:
            try:
                message = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not message:
                continue

            if message.lower() == "/quit":
                print("Goodbye.")
                break
            elif message.lower() == "/stats":
                print_stats(store)
                continue
            elif message.lower() == "/history":
                print_history()
                continue
            elif message.startswith("/"):
                print(f"Unknown command: {message}")
                continue

            # Process through orchestrator
            print("thinking...")
            try:
                result = await orchestrator.process(message)

                if result.success:
                    print()
                    print(result.output)
                    print()
                    print(
                        f"  [{result.total_nodes} nodes | "
                        f"{result.total_tokens} tok | "
                        f"{result.total_latency_ms:.0f}ms | "
                        f"${result.total_cost:.6f}]"
                    )
                else:
                    print(f"\nError: {result.errors}")

                HISTORY.append({
                    "message": message,
                    "output": result.output[:200],
                    "nodes": result.total_nodes,
                    "tokens": result.total_tokens,
                    "cost": result.total_cost,
                    "success": result.success,
                })

            except Exception as exc:
                print(f"\nError: {exc}")

            print()

    finally:
        store.close()
        print("Graph saved.")


if __name__ == "__main__":
    asyncio.run(main())
