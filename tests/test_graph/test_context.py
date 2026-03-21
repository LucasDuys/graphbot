"""Tests for GraphStore.get_context() -- context assembly from knowledge graph."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from core_gb.types import GraphContext
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    s = GraphStore(db_path=None)
    s.initialize()
    return s


def _seed_basic_graph(store: GraphStore) -> dict[str, str]:
    """Seed a graph with User, Project, 1 active Memory, 1 expired Memory.

    Returns dict of node ids keyed by role.
    """
    user_id = store.create_node("User", {
        "id": "user-1",
        "name": "Alice",
        "role": "student",
        "institution": "TU/e",
        "interests": "Python, graphs",
    })
    project_id = store.create_node("Project", {
        "id": "proj-1",
        "name": "GraphBot",
        "path": "/dev/graphbot",
        "language": "Python",
        "status": "active",
    })
    active_mem_id = store.create_node("Memory", {
        "id": "mem-active",
        "content": "User prefers Python for all projects",
        "category": "preference",
    })
    expired_mem_id = store.create_node("Memory", {
        "id": "mem-expired",
        "content": "User was using Java last year",
        "category": "preference",
        "valid_until": datetime(2020, 1, 1, tzinfo=timezone.utc),
    })

    # Edges
    store.create_edge("OWNS", user_id, project_id)
    store.create_edge("ABOUT", active_mem_id, user_id)
    store.create_edge("ABOUT", expired_mem_id, user_id)

    return {
        "user": user_id,
        "project": project_id,
        "active_memory": active_mem_id,
        "expired_memory": expired_mem_id,
    }


class TestGetContextBasic:
    """Basic context assembly tests."""

    def test_returns_graph_context_instance(self) -> None:
        """get_context returns a GraphContext dataclass."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        assert isinstance(ctx, GraphContext)
        store.close()

    def test_active_memory_included(self) -> None:
        """Active memories (valid_until IS NULL) appear in active_memories."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        assert any("Python" in m for m in ctx.active_memories)
        store.close()

    def test_expired_memory_excluded(self) -> None:
        """Expired memories (valid_until in the past) are excluded."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        assert not any("Java" in m for m in ctx.active_memories)
        store.close()

    def test_entity_info_in_relevant_entities(self) -> None:
        """Entity properties appear in relevant_entities."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        names = [e.get("name") for e in ctx.relevant_entities]
        assert "Alice" in names
        store.close()

    def test_connected_entity_found_via_traversal(self) -> None:
        """2-hop traversal finds connected Project from User."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        names = [e.get("name") for e in ctx.relevant_entities]
        assert "GraphBot" in names
        store.close()

    def test_total_tokens_set(self) -> None:
        """total_tokens is set to a positive value."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        assert ctx.total_tokens > 0
        store.close()

    def test_result_uses_tuples(self) -> None:
        """GraphContext fields are tuples, not lists."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        assert isinstance(ctx.relevant_entities, tuple)
        assert isinstance(ctx.active_memories, tuple)
        assert isinstance(ctx.matching_patterns, tuple)
        store.close()


class TestGetContextEdgeCases:
    """Edge cases for context assembly."""

    def test_empty_entity_ids_returns_empty_context(self) -> None:
        """Empty entity_ids list returns an empty GraphContext."""
        store = _make_store()
        _seed_basic_graph(store)
        ctx = store.get_context([])
        assert ctx.relevant_entities == ()
        assert ctx.active_memories == ()
        assert ctx.total_tokens == 0
        store.close()

    def test_nonexistent_entity_id(self) -> None:
        """Non-existent entity_id is gracefully skipped."""
        store = _make_store()
        _seed_basic_graph(store)
        ctx = store.get_context(["does-not-exist-anywhere"])
        assert isinstance(ctx, GraphContext)
        store.close()

    def test_max_tokens_respected(self) -> None:
        """total_tokens does not exceed max_tokens budget."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]], max_tokens=2500)
        assert ctx.total_tokens <= 2500
        store.close()

    def test_max_tokens_truncation(self) -> None:
        """Very small max_tokens causes truncation of results."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        # With a tiny budget, not everything fits
        ctx_small = store.get_context([ids["user"]], max_tokens=50)
        ctx_large = store.get_context([ids["user"]], max_tokens=2500)
        # Small context should have fewer or equal items
        total_items_small = len(ctx_small.relevant_entities) + len(ctx_small.active_memories)
        total_items_large = len(ctx_large.relevant_entities) + len(ctx_large.active_memories)
        assert total_items_small <= total_items_large
        assert ctx_small.total_tokens <= 50
        store.close()

    def test_user_summary_populated(self) -> None:
        """user_summary is populated when a User entity is in the query."""
        store = _make_store()
        ids = _seed_basic_graph(store)
        ctx = store.get_context([ids["user"]])
        assert ctx.user_summary != ""
        assert "Alice" in ctx.user_summary
        store.close()


class TestGetContextPerformance:
    """Performance benchmarks for context assembly."""

    def test_context_assembly_performance(self) -> None:
        """get_context with 100 connected nodes completes in reasonable time."""
        store = _make_store()

        # Seed 100 Memory nodes connected to a single User
        user_id = store.create_node("User", {
            "id": "perf-user",
            "name": "PerfTest",
            "role": "tester",
        })
        for i in range(100):
            mem_id = store.create_node("Memory", {
                "id": f"perf-mem-{i}",
                "content": f"Memory number {i} with some content about topic {i % 10}",
                "category": "fact",
            })
            store.create_edge("ABOUT", mem_id, user_id)

        # Warm up
        store.get_context([user_id])

        # Benchmark (median of 3 runs)
        times: list[float] = []
        for _ in range(3):
            start = time.perf_counter()
            ctx = store.get_context([user_id])
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        median_ms = sorted(times)[1]
        assert median_ms < 2000.0, f"get_context median {median_ms:.2f}ms, expected < 2000ms"
        assert isinstance(ctx, GraphContext)
        assert len(ctx.active_memories) > 0
        store.close()
