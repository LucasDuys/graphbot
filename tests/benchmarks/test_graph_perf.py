"""Smoke tests for graph query performance.

These tests use generous CI-friendly thresholds. The real performance
targets (from TESTS.md) are validated by the full benchmark script
at scripts/bench_graph.py.
"""

from __future__ import annotations

import random
import time

import pytest

from graph.resolver import EntityResolver
from graph.store import GraphStore

# Import seed helper from the benchmark script
from scripts.bench_graph import seed_graph


class TestGraphPerformanceSmoke:
    """Smoke tests: 100 nodes, 200 edges with generous CI thresholds."""

    @pytest.fixture(autouse=True)
    def setup_graph(self) -> None:
        """Create and seed an in-memory graph with 100 nodes and ~200 edges."""
        self.store = GraphStore(db_path=None)
        self.store.initialize()
        rng = random.Random(42)
        self.ids_by_table = seed_graph(self.store, 100, rng)

        # Flatten all node ids
        self.all_ids: list[tuple[str, str]] = []
        for table, ids in self.ids_by_table.items():
            for nid in ids:
                self.all_ids.append((table, nid))

        yield
        self.store.close()

    def test_two_hop_query_under_50ms(self) -> None:
        """2-hop traversal on 100 nodes completes in <50ms (CI threshold)."""
        rng = random.Random(123)
        timings: list[float] = []
        for _ in range(10):
            table, nid = rng.choice(self.all_ids)
            start = time.perf_counter()
            self.store.query(
                f"MATCH (a:{table})-[*1..2]-(b) WHERE a.id = $id RETURN b.id LIMIT 50",
                {"id": nid},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms)

        median = sorted(timings)[len(timings) // 2]
        assert median < 50.0, f"2-hop P50 = {median:.2f}ms exceeds 50ms CI threshold"

    def test_context_assembly_under_500ms(self) -> None:
        """Context assembly on 100 nodes completes in <500ms (CI threshold)."""
        rng = random.Random(456)
        timings: list[float] = []
        for _ in range(10):
            _, nid = rng.choice(self.all_ids)
            start = time.perf_counter()
            ctx = self.store.get_context([nid])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms)

        median = sorted(timings)[len(timings) // 2]
        assert median < 500.0, f"Context P50 = {median:.2f}ms exceeds 500ms CI threshold"
        # Also verify we actually got a result (not just returning empty instantly)
        # The last ctx should be a valid GraphContext
        assert ctx is not None

    def test_entity_resolution_under_500ms(self) -> None:
        """Entity resolution on 100 nodes completes in <500ms (CI threshold)."""
        resolver = EntityResolver(self.store)
        rng = random.Random(789)
        timings: list[float] = []
        for _ in range(10):
            idx = rng.randint(0, 29)  # 30 users at scale=100
            start = time.perf_counter()
            results = resolver.resolve(f"user_{idx}")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms)

        median = sorted(timings)[len(timings) // 2]
        assert median < 500.0, f"Resolve P50 = {median:.2f}ms exceeds 500ms CI threshold"
        # Verify resolution actually returns matches
        assert len(results) > 0, "Entity resolution returned no matches"

    def test_seed_creates_expected_node_counts(self) -> None:
        """Verify seeding created approximately the right distribution."""
        user_count = len(self.ids_by_table.get("User", []))
        memory_count = len(self.ids_by_table.get("Memory", []))
        task_count = len(self.ids_by_table.get("Task", []))
        project_count = len(self.ids_by_table.get("Project", []))

        # At scale=100: User=30, Memory=20, Task=20, Project=15
        assert user_count == 30, f"Expected 30 users, got {user_count}"
        assert memory_count == 20, f"Expected 20 memories, got {memory_count}"
        assert task_count == 20, f"Expected 20 tasks, got {task_count}"
        assert project_count == 15, f"Expected 15 projects, got {project_count}"
