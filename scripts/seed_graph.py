"""Seed the GraphBot knowledge graph with initial user data."""

from __future__ import annotations

import sys
from pathlib import Path


def load_env() -> None:
    """Load environment variables from .env.local if present."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        import os

        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


load_env()

from graph.store import GraphStore

DATA_DIR = Path(__file__).parent.parent / "data"
GRAPH_PATH = DATA_DIR / "graphbot.db"


def seed(store: GraphStore) -> None:
    """Populate the knowledge graph with initial nodes and edges."""
    store.initialize()

    # Check if already seeded
    existing = store.query(
        "MATCH (u:User) WHERE u.id = $id RETURN u.id", {"id": "lucas"}
    )
    if existing:
        print("Graph already seeded. Skipping.")
        return

    # User
    store.create_node("User", {
        "id": "lucas",
        "name": "Lucas Duys",
        "role": "CSE student",
        "institution": "TU/e",
        "interests": "AI, knowledge graphs, recursive systems",
    })

    # Projects
    store.create_node("Project", {
        "id": "graphbot",
        "name": "GraphBot",
        "path": "C:/dev/graphbot",
        "language": "Python",
        "framework": "kuzu",
        "status": "active",
    })
    store.create_node("Project", {
        "id": "pitchr",
        "name": "Pitchr",
        "path": "C:/dev/pitchr",
        "language": "TypeScript",
        "framework": "next.js",
        "status": "active",
    })

    # Services
    store.create_node("Service", {
        "id": "openrouter",
        "name": "OpenRouter",
        "type": "LLM API",
        "url": "https://openrouter.ai",
        "status": "active",
    })
    store.create_node("Service", {
        "id": "kuzu",
        "name": "Kuzu",
        "type": "graph database",
        "url": "https://kuzudb.com",
        "status": "archived",
    })
    store.create_node("Service", {
        "id": "langsmith",
        "name": "LangSmith",
        "type": "observability",
        "url": "https://smith.langchain.com",
        "status": "active",
    })

    # Memories
    store.create_node("Memory", {
        "id": "mem_thesis",
        "content": (
            "GraphBot core thesis: a small model with perfect context "
            "beats a large model with no context"
        ),
        "category": "architecture",
    })
    store.create_node("Memory", {
        "id": "mem_dag",
        "content": (
            "GraphBot uses recursive DAG decomposition with parallel "
            "execution on free LLMs"
        ),
        "category": "architecture",
    })
    store.create_node("Memory", {
        "id": "mem_kuzu",
        "content": (
            "Kuzu v0.11.3 is archived but functional. "
            "LadybugDB fork planned for migration"
        ),
        "category": "technical",
    })
    store.create_node("Memory", {
        "id": "mem_tue",
        "content": (
            "Lucas is a 2nd year CSE student at TU/e with 7.3 GPA, "
            "exploring TUM for masters"
        ),
        "category": "personal",
    })
    store.create_node("Memory", {
        "id": "mem_models",
        "content": (
            "Free model tier: Groq (Llama 3.3 70B, 30 RPM), "
            "Cerebras (Qwen3 235B), Google (Gemini 2.5 Pro, 5 RPM)"
        ),
        "category": "technical",
    })

    # Edges
    store.create_edge("OWNS", "lucas", "graphbot")
    store.create_edge("OWNS", "lucas", "pitchr")
    store.create_edge("USES", "lucas", "openrouter")
    store.create_edge("USES", "lucas", "kuzu")
    store.create_edge("USES", "lucas", "langsmith")
    store.create_edge("ABOUT", "mem_thesis", "lucas")
    store.create_edge("ABOUT", "mem_tue", "lucas")
    store.create_edge("ABOUT_PROJECT", "mem_dag", "graphbot")
    store.create_edge("ABOUT_PROJECT", "mem_kuzu", "graphbot")

    print("Seeded graph with user, 2 projects, 3 services, 5 memories, 9 edges")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = GraphStore(str(GRAPH_PATH))
    try:
        seed(store)
    finally:
        store.close()
