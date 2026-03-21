"""Benchmark Kuzu graph performance at various scales.

Measures 2-hop traversal, context assembly, and entity resolution latency
across configurable node counts. Results are printed as a formatted table
and appended to benchmarks/graph_perf.jsonl.

Usage:
    python scripts/bench_graph.py
    python scripts/bench_graph.py --scales 100,500,1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from graph.resolver import EntityResolver
from graph.store import GraphStore

# ---------------------------------------------------------------------------
# Node distribution ratios (must sum to 1.0)
# ---------------------------------------------------------------------------
_NODE_DISTRIBUTION: list[tuple[str, float, dict[str, object]]] = [
    ("User", 0.30, {}),
    ("Memory", 0.20, {}),
    ("Task", 0.20, {}),
    ("Project", 0.15, {}),
    # "Other" bucket: split among File, Service, Contact, Skill
    ("File", 0.04, {}),
    ("Service", 0.04, {}),
    ("Contact", 0.04, {}),
    ("Skill", 0.03, {}),
]

# Valid edge types: (edge_name, from_type, to_type)
_VALID_EDGES: list[tuple[str, str, str]] = [
    ("OWNS", "User", "Project"),
    ("USES", "User", "Service"),
    ("ABOUT", "Memory", "User"),
    ("ABOUT_PROJECT", "Memory", "Project"),
    ("PRODUCED", "Task", "File"),
    ("DEPENDS_ON", "Task", "Task"),
    ("CONTEXT_FROM", "Task", "Memory"),
    ("INVOLVES", "Task", "Service"),
    ("HAS_SKILL", "User", "Skill"),
]


def _make_node_props(table: str, index: int) -> dict[str, object]:
    """Generate synthetic properties for a node of the given table type.

    TIMESTAMP fields use real datetime objects (Kuzu rejects ISO strings).
    Optional TIMESTAMP fields that would be None are omitted entirely so
    Kuzu stores them as NULL.
    """
    now = datetime.now(timezone.utc)
    if table == "User":
        return {"name": f"user_{index}", "role": "tester", "institution": "", "interests": ""}
    elif table == "Memory":
        return {
            "content": f"Memory about topic {index}",
            "category": "general",
            "confidence": 0.9,
            "source_episode": "",
            "valid_from": now,
            # valid_until omitted -> NULL in Kuzu
        }
    elif table == "Task":
        return {
            "description": f"Task {index}",
            "domain": "system",
            "complexity": 1,
            "status": "completed",
            "tokens_used": 0,
            "latency_ms": 0.0,
            "created_at": now,
            # completed_at omitted -> NULL
        }
    elif table == "Project":
        return {
            "name": f"project_{index}",
            "path": f"/tmp/proj_{index}",
            "language": "python",
            "framework": "",
            "status": "active",
        }
    elif table == "File":
        return {"path": f"/tmp/file_{index}.py", "type": "python", "description": f"File {index}"}
    elif table == "Service":
        return {"name": f"service_{index}", "type": "api", "url": f"https://svc{index}.test", "status": "active"}
    elif table == "Contact":
        return {"name": f"contact_{index}", "relationship": "colleague", "platform": "email"}
    elif table == "Skill":
        return {"name": f"skill_{index}", "description": f"Skill number {index}", "path": ""}
    else:
        return {"name": f"node_{index}"}


def seed_graph(store: GraphStore, n: int, rng: random.Random) -> dict[str, list[str]]:
    """Seed the graph with N nodes and ~2*N edges. Returns {table: [node_ids]}."""
    # Build node list according to distribution
    node_plan: list[tuple[str, int]] = []
    remaining = n
    for i, (table, ratio, _) in enumerate(_NODE_DISTRIBUTION):
        if i == len(_NODE_DISTRIBUTION) - 1:
            count = remaining  # last bucket gets the rest
        else:
            count = int(n * ratio)
            remaining -= count
        node_plan.append((table, count))

    # Create nodes
    ids_by_table: dict[str, list[str]] = {}
    all_ids: list[tuple[str, str]] = []  # (table, id)
    global_idx = 0
    for table, count in node_plan:
        ids_by_table.setdefault(table, [])
        for _ in range(count):
            props = _make_node_props(table, global_idx)
            node_id = store.create_node(table, props)
            ids_by_table[table].append(node_id)
            all_ids.append((table, node_id))
            global_idx += 1

    # Create ~2*N edges randomly between valid pairs
    target_edges = 2 * n
    created = 0
    attempts = 0
    max_attempts = target_edges * 10  # prevent infinite loop

    # Build lookup for quick random selection
    while created < target_edges and attempts < max_attempts:
        attempts += 1
        edge_name, from_type, to_type = rng.choice(_VALID_EDGES)
        from_ids = ids_by_table.get(from_type, [])
        to_ids = ids_by_table.get(to_type, [])
        if not from_ids or not to_ids:
            continue
        from_id = rng.choice(from_ids)
        to_id = rng.choice(to_ids)
        if from_id == to_id and from_type == to_type:
            continue
        try:
            store.create_edge(edge_name, from_id, to_id)
            created += 1
        except Exception:
            pass

    return ids_by_table


def _time_fn(fn: callable, iterations: int = 10) -> tuple[float, float]:
    """Run fn() for N iterations, return (p50_ms, p95_ms)."""
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
    timings.sort()
    p50 = statistics.median(timings)
    p95_idx = int(0.95 * len(timings)) - 1
    p95 = timings[max(p95_idx, 0)]
    return p50, p95


def run_benchmark(
    scale: int,
    iterations: int = 10,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Run benchmark at a given scale. Returns list of result dicts."""
    rng = random.Random(seed)

    store = GraphStore(db_path=None)
    store.initialize()
    ids_by_table = seed_graph(store, scale, rng)
    resolver = EntityResolver(store)

    # Collect all node ids for random picking
    all_ids: list[tuple[str, str]] = []
    for table, ids in ids_by_table.items():
        for nid in ids:
            all_ids.append((table, nid))

    results: list[dict[str, object]] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- 2-hop traversal ---
    def two_hop() -> None:
        table, nid = rng.choice(all_ids)
        store.query(
            f"MATCH (a:{table})-[*1..2]-(b) WHERE a.id = $id RETURN b.id LIMIT 50",
            {"id": nid},
        )

    p50, p95 = _time_fn(two_hop, iterations)
    results.append({"date": today, "scale": scale, "op": "2hop", "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)})

    # --- Context assembly ---
    def context_assembly() -> None:
        _, nid = rng.choice(all_ids)
        store.get_context([nid])

    p50, p95 = _time_fn(context_assembly, iterations)
    results.append({"date": today, "scale": scale, "op": "context", "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)})

    # --- Entity resolution ---
    user_ids = ids_by_table.get("User", [])

    def entity_resolve() -> None:
        idx = rng.randint(0, max(len(user_ids) - 1, 0))
        resolver.resolve(f"user_{idx}")

    p50, p95 = _time_fn(entity_resolve, iterations)
    results.append({"date": today, "scale": scale, "op": "resolve", "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)})

    store.close()
    return results


def print_table(all_results: list[dict[str, object]]) -> None:
    """Print results as a formatted table."""
    header = f"{'Scale':>8}  {'Operation':>10}  {'P50 (ms)':>10}  {'P95 (ms)':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['scale']:>8}  {r['op']:>10}  {r['p50_ms']:>10.3f}  {r['p95_ms']:>10.3f}")


def append_jsonl(all_results: list[dict[str, object]], path: Path) -> None:
    """Append results to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")


def main(scales: Sequence[int] | None = None) -> None:
    """Run benchmarks at specified scales."""
    if scales is None:
        scales = [100, 1000, 5000, 10000]

    all_results: list[dict[str, object]] = []
    for scale in scales:
        print(f"Benchmarking scale={scale} ...")
        results = run_benchmark(scale)
        all_results.extend(results)

    print()
    print_table(all_results)

    jsonl_path = Path(__file__).resolve().parent.parent / "benchmarks" / "graph_perf.jsonl"
    append_jsonl(all_results, jsonl_path)
    print(f"\nResults appended to {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Kuzu graph performance")
    parser.add_argument(
        "--scales",
        type=str,
        default="100,1000,5000,10000",
        help="Comma-separated list of node counts (default: 100,1000,5000,10000)",
    )
    args = parser.parse_args()
    scale_list = [int(s.strip()) for s in args.scales.split(",")]
    main(scale_list)
