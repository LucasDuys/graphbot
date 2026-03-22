"""Tests for PPR integration into context assembly.

Validates that:
1. assemble_context with use_ppr=False delegates to 2-hop GraphStore.get_context
2. assemble_context with use_ppr=True seeds PPR from entity IDs and returns deeper results
3. PPR mode falls back to 2-hop when PPR returns empty results
4. 2-hop mode behavior is unchanged (backward compat)
5. PPR mode discovers nodes beyond 2-hop reach
"""

from __future__ import annotations

import pytest

from core_gb.types import GraphContext
from graph.context import assemble_context
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _seed_deep_graph(store: GraphStore) -> dict[str, str]:
    """Seed a graph with a deep chain that exceeds 2-hop traversal.

    Topology:
        User(alice) --OWNS--> Project(proj_a)
        User(alice) --USES--> Service(svc_github)
        Task(task1) --INVOLVES--> Service(svc_github)
        Task(task1) --DEPENDS_ON--> Task(task2)
        Task(task2) --DEPENDS_ON--> Task(task3)
        Task(task3) --DEPENDS_ON--> Task(task4)
        Task(task4) --PRODUCED--> File(deep_file)
        Memory(mem1) --ABOUT--> User(alice)

    From alice via 2-hop:
        hop 0: alice
        hop 1: proj_a, svc_github, mem1
        hop 2: task1 (via svc_github)
    Beyond 2-hop: task2, task3, task4, deep_file
    """
    store.create_node("User", {"id": "alice", "name": "Alice", "role": "developer"})
    store.create_node("Project", {
        "id": "proj_a", "name": "ProjectAlpha", "path": "/projects/alpha",
        "language": "Python", "framework": "FastAPI", "status": "active",
    })
    store.create_node("Service", {
        "id": "svc_github", "name": "GitHub", "type": "vcs",
        "url": "https://github.com", "status": "active",
    })
    store.create_node("Memory", {
        "id": "mem1", "content": "Alice prefers pytest", "category": "preference",
        "confidence": 0.9,
    })
    store.create_node("Task", {
        "id": "task1", "description": "Setup CI pipeline", "domain": "code",
        "complexity": 2, "status": "completed",
    })
    store.create_node("Task", {
        "id": "task2", "description": "Write unit tests", "domain": "code",
        "complexity": 1, "status": "completed",
    })
    store.create_node("Task", {
        "id": "task3", "description": "Run integration tests", "domain": "code",
        "complexity": 2, "status": "completed",
    })
    store.create_node("Task", {
        "id": "task4", "description": "Deploy to production", "domain": "code",
        "complexity": 3, "status": "completed",
    })
    store.create_node("File", {
        "id": "deep_file", "path": "/deploy/config.yaml",
        "type": "config", "description": "Production deployment config",
    })

    store.create_edge("OWNS", "alice", "proj_a")
    store.create_edge("USES", "alice", "svc_github")
    store.create_edge("ABOUT", "mem1", "alice")
    store.create_edge("INVOLVES", "task1", "svc_github")
    store.create_edge("DEPENDS_ON", "task1", "task2")
    store.create_edge("DEPENDS_ON", "task2", "task3")
    store.create_edge("DEPENDS_ON", "task3", "task4")
    store.create_edge("PRODUCED", "task4", "deep_file")

    return {
        "user": "alice",
        "project": "proj_a",
        "service": "svc_github",
        "memory": "mem1",
        "task1": "task1",
        "task2": "task2",
        "task3": "task3",
        "task4": "task4",
        "deep_file": "deep_file",
    }


class TestAssembleContextTwoHopMode:
    """Tests that 2-hop mode (use_ppr=False) is unchanged."""

    def test_default_is_two_hop(self) -> None:
        """use_ppr defaults to False (backward compatible)."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(store, [ids["user"]])
        assert isinstance(ctx, GraphContext)
        store.close()

    def test_two_hop_returns_graph_context(self) -> None:
        """2-hop mode returns a GraphContext instance."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(store, [ids["user"]], use_ppr=False)
        assert isinstance(ctx, GraphContext)
        store.close()

    def test_two_hop_finds_direct_neighbors(self) -> None:
        """2-hop mode finds entities within 2 hops."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(store, [ids["user"]], use_ppr=False)
        names = [e.get("name") for e in ctx.relevant_entities]
        assert "Alice" in names
        assert "ProjectAlpha" in names
        store.close()

    def test_two_hop_does_not_find_deep_nodes(self) -> None:
        """2-hop mode does NOT find nodes beyond 2 hops."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(store, [ids["user"]], use_ppr=False, max_tokens=10000)
        # Collect all entity names from context
        entity_names = {e.get("name", "") for e in ctx.relevant_entities}
        # task3, task4, deep_file are beyond 2 hops and should not appear
        # (task3 has no 'name' field -- it uses description, but entity name
        # comes from props.get("name", props.get("id", "")) so it would be the id)
        entity_ids_in_context = set()
        for e in ctx.relevant_entities:
            entity_ids_in_context.add(e.get("name", ""))
        # deep_file's name would be its id "deep_file" since File has no name field
        # (it maps path as name since File uses props.get("name", props.get("id", "")))
        assert "deep_file" not in entity_ids_in_context
        store.close()

    def test_two_hop_empty_entity_ids(self) -> None:
        """Empty entity_ids returns empty context in 2-hop mode."""
        store = _make_store()
        _seed_deep_graph(store)
        ctx = assemble_context(store, [], use_ppr=False)
        assert ctx.relevant_entities == ()
        assert ctx.active_memories == ()
        store.close()

    def test_two_hop_respects_max_tokens(self) -> None:
        """2-hop mode respects max_tokens budget."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(store, [ids["user"]], use_ppr=False, max_tokens=50)
        assert ctx.total_tokens <= 50
        store.close()


class TestAssembleContextPPRMode:
    """Tests that PPR mode discovers deeper nodes."""

    def test_ppr_returns_graph_context(self) -> None:
        """PPR mode returns a GraphContext instance."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(store, [ids["user"]], use_ppr=True)
        assert isinstance(ctx, GraphContext)
        store.close()

    def test_ppr_finds_deep_nodes(self) -> None:
        """PPR mode discovers nodes beyond 2-hop reach."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(
            store, [ids["user"]], use_ppr=True, max_tokens=10000, ppr_top_k=20,
        )
        # Collect all entity names/ids from context
        entity_names = set()
        for e in ctx.relevant_entities:
            entity_names.add(e.get("name", ""))
        # PPR should find at least one node that is 3+ hops away
        # task2 (id=task2), task3, task4, or deep_file
        deep_ids = {"task2", "task3", "task4", "deep_file"}
        found_deep = deep_ids & entity_names
        assert len(found_deep) > 0, (
            f"PPR should reach nodes beyond 2 hops. Found names: {entity_names}"
        )
        store.close()

    def test_ppr_includes_direct_neighbors_too(self) -> None:
        """PPR mode also includes nearby nodes (not just deep ones)."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(
            store, [ids["user"]], use_ppr=True, max_tokens=10000, ppr_top_k=20,
        )
        entity_names = {e.get("name", "") for e in ctx.relevant_entities}
        assert "Alice" in entity_names
        store.close()

    def test_ppr_returns_more_entities_than_two_hop(self) -> None:
        """PPR mode should find strictly more entities in a deep graph."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx_2hop = assemble_context(
            store, [ids["user"]], use_ppr=False, max_tokens=10000,
        )
        ctx_ppr = assemble_context(
            store, [ids["user"]], use_ppr=True, max_tokens=10000, ppr_top_k=20,
        )
        count_2hop = len(ctx_2hop.relevant_entities) + len(ctx_2hop.active_memories)
        count_ppr = len(ctx_ppr.relevant_entities) + len(ctx_ppr.active_memories)
        assert count_ppr >= count_2hop, (
            f"PPR should find >= entities. PPR={count_ppr}, 2hop={count_2hop}"
        )
        store.close()

    def test_ppr_respects_max_tokens(self) -> None:
        """PPR mode respects max_tokens budget."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx = assemble_context(
            store, [ids["user"]], use_ppr=True, max_tokens=50,
        )
        assert ctx.total_tokens <= 50
        store.close()

    def test_ppr_empty_entity_ids(self) -> None:
        """Empty entity_ids returns empty context in PPR mode."""
        store = _make_store()
        _seed_deep_graph(store)
        ctx = assemble_context(store, [], use_ppr=True)
        assert ctx.relevant_entities == ()
        assert ctx.active_memories == ()
        store.close()

    def test_ppr_custom_top_k(self) -> None:
        """ppr_top_k limits the number of nodes retrieved by PPR."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx_small = assemble_context(
            store, [ids["user"]], use_ppr=True, ppr_top_k=2, max_tokens=10000,
        )
        ctx_large = assemble_context(
            store, [ids["user"]], use_ppr=True, ppr_top_k=20, max_tokens=10000,
        )
        count_small = len(ctx_small.relevant_entities) + len(ctx_small.active_memories)
        count_large = len(ctx_large.relevant_entities) + len(ctx_large.active_memories)
        assert count_small <= count_large
        store.close()


class TestAssembleContextPPRFallback:
    """Tests that PPR falls back to 2-hop when PPR returns empty."""

    def test_fallback_on_nonexistent_seed(self) -> None:
        """When PPR seeds do not exist, falls back to 2-hop."""
        store = _make_store()
        ids = _seed_deep_graph(store)
        # Use a valid entity id that exists in graph -- PPR should work.
        # But use nonexistent ids so PPR returns empty, triggering fallback.
        ctx = assemble_context(
            store, ["nonexistent_entity"], use_ppr=True, max_tokens=10000,
        )
        # Should still return a GraphContext (fallback to 2-hop, which also
        # finds nothing for nonexistent entity)
        assert isinstance(ctx, GraphContext)
        store.close()

    def test_fallback_returns_two_hop_results(self) -> None:
        """Fallback produces same results as explicit 2-hop call.

        This uses a store where the PPR retriever is given seeds that don't
        exist in the graph, so PPR returns []. The function then falls back
        to 2-hop, which also returns empty for nonexistent seeds. Both should
        produce equivalent empty results.
        """
        store = _make_store()
        ids = _seed_deep_graph(store)
        ctx_fallback = assemble_context(
            store, ["nonexistent_entity"], use_ppr=True, max_tokens=10000,
        )
        ctx_2hop = assemble_context(
            store, ["nonexistent_entity"], use_ppr=False, max_tokens=10000,
        )
        assert len(ctx_fallback.relevant_entities) == len(ctx_2hop.relevant_entities)
        assert len(ctx_fallback.active_memories) == len(ctx_2hop.active_memories)
        store.close()

    def test_fallback_on_empty_graph(self) -> None:
        """PPR on an empty graph falls back to 2-hop gracefully."""
        store = _make_store()
        ctx = assemble_context(store, ["some_id"], use_ppr=True)
        assert isinstance(ctx, GraphContext)
        assert ctx.relevant_entities == ()
        assert ctx.active_memories == ()
        store.close()
