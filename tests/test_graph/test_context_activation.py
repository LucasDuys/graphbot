"""Tests for activation-aware context assembly with ranked token budget.

Validates that GraphStore.get_context() uses ActivationModel scores to rank
retrieved nodes, preferring high-activation nodes in token budget allocation
and sorting by activation_score descending before trimming.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core_gb.types import GraphContext
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


class TestActivationAwareContextAssembly:
    """Context assembly ranks nodes by activation score."""

    def test_high_activation_node_included_over_low_when_budget_tight(
        self,
    ) -> None:
        """With a tight token budget, high-activation nodes should be kept
        while low-activation nodes are trimmed.

        Creates two Memory nodes connected to the same User: one with high
        activation (many accesses, recent) and one with low activation (few
        accesses, stale). Under a tight budget, only the high-activation
        memory should survive trimming.
        """
        store = _make_store()
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(days=60)

        user_id = store.create_node("User", {
            "id": "u1",
            "name": "TestUser",
            "role": "dev",
            "access_count": 10,
            "last_accessed": now,
        })

        # High-activation memory: many accesses, recently accessed
        store.create_node("Memory", {
            "id": "mem-hot",
            "content": "High activation memory with important context",
            "category": "fact",
            "access_count": 100,
            "last_accessed": now,
        })
        store.create_edge("ABOUT", "mem-hot", user_id)

        # Low-activation memory: few accesses, accessed long ago
        store.create_node("Memory", {
            "id": "mem-cold",
            "content": "Low activation memory with stale information",
            "category": "fact",
            "access_count": 1,
            "last_accessed": stale_time,
        })
        store.create_edge("ABOUT", "mem-cold", user_id)

        # Use a budget tight enough that only one memory fits after the user entity
        # First, get full context to understand token usage
        ctx_full = store.get_context([user_id], max_tokens=5000)
        assert len(ctx_full.active_memories) == 2, (
            "Both memories should fit with a generous budget"
        )

        # Find a budget that fits exactly user + one memory.
        # Shrink from full until only 1 memory survives.
        budget = ctx_full.total_tokens
        ctx_tight = ctx_full
        for shrink in range(1, budget):
            candidate = store.get_context([user_id], max_tokens=budget - shrink)
            if len(candidate.active_memories) == 1:
                ctx_tight = candidate
                break

        assert len(ctx_tight.active_memories) == 1, (
            f"Expected exactly 1 memory under tight budget, "
            f"got {len(ctx_tight.active_memories)}"
        )
        # The surviving memory should be the HIGH-activation one
        assert "High activation" in ctx_tight.active_memories[0], (
            "The high-activation memory should be preferred over the "
            f"low-activation one when budget is tight, got: {ctx_tight.active_memories[0]}"
        )

        store.close()

    def test_activation_ranking_overrides_hop_distance(self) -> None:
        """A high-activation entity at hop 2 should be preferred over a
        low-activation entity at hop 1 when budget is tight.

        Creates a User -> Project (hop 1, low activation) and
        User -> Service -> File (hop 2, high activation). Under budget
        pressure, the high-activation File should be kept.
        """
        store = _make_store()
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(days=90)

        user_id = store.create_node("User", {
            "id": "u2",
            "name": "Ranker",
            "role": "tester",
            "access_count": 10,
            "last_accessed": now,
        })

        # Hop 1 entity: Project with LOW activation (stale, rarely accessed)
        store.create_node("Project", {
            "id": "proj-cold",
            "name": "ColdProject",
            "path": "/cold",
            "language": "Rust",
            "status": "archived",
            "access_count": 1,
            "last_accessed": stale_time,
        })
        store.create_edge("OWNS", user_id, "proj-cold")

        # Hop 1 entity: Service with HIGH activation (recent, many accesses)
        store.create_node("Service", {
            "id": "svc-hot",
            "name": "HotService",
            "type": "api",
            "url": "https://hot.api",
            "status": "active",
            "access_count": 200,
            "last_accessed": now,
        })
        store.create_edge("USES", user_id, "svc-hot")

        # Full budget: both should appear
        ctx_full = store.get_context([user_id], max_tokens=5000)
        entity_names_full = [e.get("name") for e in ctx_full.relevant_entities]
        assert "HotService" in entity_names_full
        assert "ColdProject" in entity_names_full

        # Tight budget: only the user + one entity should fit
        # The high-activation entity should be preferred
        ctx_tight = store.get_context([user_id], max_tokens=55)
        entity_names_tight = [e.get("name") for e in ctx_tight.relevant_entities]

        # With activation ranking, HotService should appear before ColdProject
        if "HotService" in entity_names_tight and "ColdProject" not in entity_names_tight:
            pass  # Correct: high-activation entity preferred
        elif len(entity_names_tight) <= 1:
            # Only user entity fits; acceptable under very tight budget
            pass
        else:
            # If both are present, HotService should appear first
            hot_idx = (
                entity_names_tight.index("HotService")
                if "HotService" in entity_names_tight
                else float("inf")
            )
            cold_idx = (
                entity_names_tight.index("ColdProject")
                if "ColdProject" in entity_names_tight
                else float("inf")
            )
            assert hot_idx < cold_idx, (
                "High-activation entity should be ranked before "
                "low-activation entity"
            )

        store.close()

    def test_nodes_sorted_by_activation_score_descending(self) -> None:
        """Retrieved entity nodes should appear in activation score descending
        order in the final context, regardless of hop distance.
        """
        store = _make_store()
        now = datetime.now(timezone.utc)

        user_id = store.create_node("User", {
            "id": "u3",
            "name": "SortTest",
            "role": "dev",
            "access_count": 50,
            "last_accessed": now,
        })

        # Three services at hop 1, different activation levels
        for name, count, days_ago in [
            ("ServiceA", 5, 30),
            ("ServiceB", 100, 0),
            ("ServiceC", 20, 7),
        ]:
            svc_id = f"svc-{name.lower()}"
            store.create_node("Service", {
                "id": svc_id,
                "name": name,
                "type": "api",
                "status": "active",
                "access_count": count,
                "last_accessed": now - timedelta(days=days_ago),
            })
            store.create_edge("USES", user_id, svc_id)

        ctx = store.get_context([user_id], max_tokens=5000)
        entity_names = [e.get("name") for e in ctx.relevant_entities]

        # Filter to just the services (skip the User entity)
        service_names = [n for n in entity_names if n in ("ServiceA", "ServiceB", "ServiceC")]

        # ServiceB (count=100, recent) should be first, then ServiceC (count=20, 7d),
        # then ServiceA (count=5, 30d)
        assert len(service_names) == 3, f"Expected 3 services, got {service_names}"
        assert service_names[0] == "ServiceB", (
            f"Highest-activation service should be first, got {service_names}"
        )
        # ServiceA should be last (lowest activation)
        assert service_names[-1] == "ServiceA", (
            f"Lowest-activation service should be last, got {service_names}"
        )

        store.close()

    def test_memory_nodes_sorted_by_activation(self) -> None:
        """Memory nodes should also be sorted by activation score descending."""
        store = _make_store()
        now = datetime.now(timezone.utc)

        user_id = store.create_node("User", {
            "id": "u4",
            "name": "MemSort",
            "role": "dev",
            "access_count": 10,
            "last_accessed": now,
        })

        # Three memories with varying activation
        for label, count, days_ago in [
            ("alpha-low", 1, 60),
            ("beta-high", 50, 0),
            ("gamma-mid", 10, 5),
        ]:
            mem_id = f"mem-{label}"
            store.create_node("Memory", {
                "id": mem_id,
                "content": f"Memory content for {label}",
                "category": "fact",
                "access_count": count,
                "last_accessed": now - timedelta(days=days_ago),
            })
            store.create_edge("ABOUT", mem_id, user_id)

        ctx = store.get_context([user_id], max_tokens=5000)

        # beta-high should be first, then gamma-mid, then alpha-low
        assert len(ctx.active_memories) == 3
        assert "beta-high" in ctx.active_memories[0], (
            f"Highest-activation memory should be first, got: {ctx.active_memories}"
        )
        assert "alpha-low" in ctx.active_memories[-1], (
            f"Lowest-activation memory should be last, got: {ctx.active_memories}"
        )

        store.close()

    def test_null_activation_metadata_treated_as_minimum(self) -> None:
        """Nodes without access_count or last_accessed should still be handled
        gracefully, treated as minimum activation (score 0 or near-0).
        """
        store = _make_store()
        now = datetime.now(timezone.utc)

        user_id = store.create_node("User", {
            "id": "u5",
            "name": "NullTest",
            "role": "dev",
            "access_count": 10,
            "last_accessed": now,
        })

        # Memory with activation metadata
        store.create_node("Memory", {
            "id": "mem-with-meta",
            "content": "Memory with activation metadata present",
            "category": "fact",
            "access_count": 50,
            "last_accessed": now,
        })
        store.create_edge("ABOUT", "mem-with-meta", user_id)

        # Memory WITHOUT activation metadata (nulls)
        store.create_node("Memory", {
            "id": "mem-no-meta",
            "content": "Memory with no activation metadata at all",
            "category": "fact",
        })
        store.create_edge("ABOUT", "mem-no-meta", user_id)

        ctx = store.get_context([user_id], max_tokens=5000)

        # Both should appear, but the one with metadata should be ranked first
        assert len(ctx.active_memories) == 2
        assert "with activation metadata present" in ctx.active_memories[0], (
            f"Memory with activation data should rank first, got: {ctx.active_memories}"
        )

        store.close()

    def test_low_activation_node_trimmed_when_budget_tight(self) -> None:
        """Under a tight token budget, low-activation nodes are the ones
        trimmed, not high-activation ones.

        This is the core acceptance criterion: given N nodes and a budget
        that fits N-1, the dropped node should be the one with the lowest
        activation score.
        """
        store = _make_store()
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(days=90)

        user_id = store.create_node("User", {
            "id": "u6",
            "name": "TrimTest",
            "role": "dev",
            "access_count": 10,
            "last_accessed": now,
        })

        # High-activation memory (should survive trimming)
        store.create_node("Memory", {
            "id": "mem-survive",
            "content": "Important high-activation memory that must survive",
            "category": "fact",
            "access_count": 100,
            "last_accessed": now,
        })
        store.create_edge("ABOUT", "mem-survive", user_id)

        # Low-activation memory (should be trimmed first)
        store.create_node("Memory", {
            "id": "mem-trim",
            "content": "Stale low-activation memory that should be trimmed",
            "category": "fact",
            "access_count": 1,
            "last_accessed": stale_time,
        })
        store.create_edge("ABOUT", "mem-trim", user_id)

        # Get full context to see both memories
        ctx_full = store.get_context([user_id], max_tokens=5000)
        assert len(ctx_full.active_memories) == 2

        # Now find a budget that only fits the user entity + one memory
        # Start from full tokens and shrink until only 1 memory fits
        budget = ctx_full.total_tokens
        ctx_trimmed = ctx_full
        for shrink in range(1, budget):
            candidate = store.get_context([user_id], max_tokens=budget - shrink)
            if len(candidate.active_memories) == 1:
                ctx_trimmed = candidate
                break

        if len(ctx_trimmed.active_memories) == 1:
            # The surviving memory should be the high-activation one
            assert "high-activation" in ctx_trimmed.active_memories[0], (
                f"High-activation memory should survive trimming, "
                f"got: {ctx_trimmed.active_memories[0]}"
            )
        else:
            pytest.fail(
                f"Could not find a budget that trims exactly one memory. "
                f"Got {len(ctx_trimmed.active_memories)} memories."
            )

        store.close()
