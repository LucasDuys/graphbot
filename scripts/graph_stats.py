"""Print knowledge graph statistics."""

from __future__ import annotations

import os
import sys
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

from graph.store import GraphStore


def print_stats(store: GraphStore) -> None:
    """Print node counts, edge counts, and pattern cache stats."""
    store.initialize()

    # Count nodes by type
    node_tables = [
        "User", "Project", "Service", "Contact", "Memory",
        "Task", "PatternNode", "ExecutionTree", "Skill", "File",
    ]

    print("GRAPH STATISTICS")
    print("=" * 40)

    total_nodes = 0
    for table in node_tables:
        try:
            rows = store.query(f"MATCH (n:{table}) RETURN count(n) AS cnt")
            count = rows[0]["cnt"] if rows else 0
            if count > 0:
                print(f"  {table:20s}: {count}")
            total_nodes += count
        except Exception:
            pass

    print(f"  {'TOTAL':20s}: {total_nodes}")

    # Count edges
    edge_tables = [
        "OWNS", "USES", "STUDIES_AT", "ABOUT", "ABOUT_PROJECT",
        "PRODUCED", "CREATED_PATTERN", "DEPENDS_ON", "CONTEXT_FROM",
        "INVOLVES", "DERIVED_FROM", "HAS_SKILL",
    ]

    print()
    total_edges = 0
    for table in edge_tables:
        try:
            rows = store.query(f"MATCH ()-[r:{table}]->() RETURN count(r) AS cnt")
            count = rows[0]["cnt"] if rows else 0
            if count > 0:
                print(f"  {table:20s}: {count}")
            total_edges += count
        except Exception:
            pass

    print(f"  {'TOTAL EDGES':20s}: {total_edges}")

    # Pattern cache stats
    print()
    try:
        patterns = store.query(
            "MATCH (p:PatternNode) "
            "RETURN p.trigger_template, p.success_count "
            "ORDER BY p.success_count DESC"
        )
        print(f"PATTERNS: {len(patterns)}")
        for p in patterns[:5]:
            trigger = str(p.get("p.trigger_template", ""))[:50]
            count = p.get("p.success_count", 0)
            print(f"  [{count}x] {trigger}")
    except Exception:
        print("PATTERNS: 0")


if __name__ == "__main__":
    # Use in-memory for quick check, or persistent path
    db_path: str | None = sys.argv[1] if len(sys.argv) > 1 else None
    store = GraphStore(db_path)
    try:
        print_stats(store)
    finally:
        store.close()
