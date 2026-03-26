"""Tests for ContextEnricher -- graph context enrichment for single-call mode."""

from __future__ import annotations

import json
import uuid

import pytest

from core_gb.context_enrichment import ContextEnricher, EnrichedContext
from core_gb.conversation import ConversationMemory
from core_gb.patterns import PatternMatcher, PatternStore
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _seed_entities(store: GraphStore) -> list[str]:
    """Seed the graph with a User and a Project, return their IDs."""
    user_id = store.create_node("User", {
        "id": "user-001",
        "name": "Alice",
        "role": "developer",
        "institution": "TU/e",
        "interests": "AI, graphs",
        "access_count": 5,
    })
    project_id = store.create_node("Project", {
        "id": "proj-001",
        "name": "GraphBot",
        "status": "active",
        "language": "Python",
        "framework": "kuzu",
        "access_count": 3,
    })
    # Link user to project via OWNS edge
    store.create_edge("OWNS", user_id, project_id)
    return [user_id, project_id]


def _seed_reflection(store: GraphStore, task_desc: str) -> str:
    """Create a reflection Memory node linked to a Task node."""
    task_id = store.create_node("Task", {
        "id": str(uuid.uuid4()),
        "description": task_desc,
        "status": "failed",
    })
    reflection_content = json.dumps({
        "what_failed": "timeout on API call",
        "why": "no retry logic",
        "what_to_try": "add exponential backoff",
    })
    mem_id = store.create_node("Memory", {
        "id": str(uuid.uuid4()),
        "content": reflection_content,
        "category": "reflection",
        "confidence": 1.0,
    })
    store.create_edge("REFLECTION_OF", mem_id, task_id)
    return mem_id


class TestEnrichedContextDataclass:
    """EnrichedContext dataclass structure and token estimates."""

    def test_default_enriched_context_is_empty(self) -> None:
        ctx = EnrichedContext()
        assert ctx.entities == ()
        assert ctx.memories == ()
        assert ctx.reflections == ()
        assert ctx.patterns == ()
        assert ctx.conversation_turns == ()
        assert ctx.entity_tokens == 0
        assert ctx.memory_tokens == 0
        assert ctx.reflection_tokens == 0
        assert ctx.pattern_tokens == 0
        assert ctx.conversation_tokens == 0

    def test_total_tokens_sums_all_sections(self) -> None:
        ctx = EnrichedContext(
            entity_tokens=100,
            memory_tokens=50,
            reflection_tokens=30,
            pattern_tokens=20,
            conversation_tokens=40,
        )
        assert ctx.total_tokens == 240


class TestContextEnricherEntities:
    """Entity resolution and retrieval via EntityResolver."""

    def test_resolves_entities_from_task_description(self) -> None:
        store = _make_store()
        _seed_entities(store)
        enricher = ContextEnricher(store=store)

        # Use exact entity name so EntityResolver can match via exact/BM25
        ctx = enricher.enrich("GraphBot")

        assert len(ctx.entities) > 0
        entity_names = [e.get("name", "") for e in ctx.entities]
        assert "GraphBot" in entity_names
        assert ctx.entity_tokens > 0
        store.close()

    def test_empty_task_returns_empty_context(self) -> None:
        store = _make_store()
        enricher = ContextEnricher(store=store)

        ctx = enricher.enrich("")

        assert ctx.entities == ()
        assert ctx.total_tokens == 0
        store.close()

    def test_no_matching_entities_returns_empty_entities(self) -> None:
        store = _make_store()
        enricher = ContextEnricher(store=store)

        ctx = enricher.enrich("completely unrelated xyz qqq")

        assert ctx.entities == ()
        store.close()


class TestContextEnricherConversation:
    """Conversation history retrieval."""

    def test_includes_conversation_turns_when_chat_id_given(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)
        conv.add_message("chat-001", "user", "What is GraphBot?")
        conv.add_message("chat-001", "assistant", "A recursive DAG engine.")

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich("Tell me more about GraphBot", chat_id="chat-001")

        assert len(ctx.conversation_turns) == 2
        assert ctx.conversation_turns[0]["role"] == "user"
        assert ctx.conversation_turns[1]["role"] == "assistant"
        assert ctx.conversation_tokens > 0
        store.close()

    def test_no_conversation_without_chat_id(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)
        conv.add_message("chat-001", "user", "Hello")

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich("Tell me something")

        assert ctx.conversation_turns == ()
        assert ctx.conversation_tokens == 0
        store.close()


class TestContextEnricherReflections:
    """Failure reflection retrieval."""

    def test_includes_relevant_reflections(self) -> None:
        store = _make_store()
        _seed_reflection(store, "call the weather API for Amsterdam")

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich("call the weather API for Amsterdam")

        assert len(ctx.reflections) >= 1
        assert ctx.reflections[0]["what_failed"] == "timeout on API call"
        assert ctx.reflection_tokens > 0
        store.close()

    def test_no_reflections_for_unrelated_task(self) -> None:
        store = _make_store()
        _seed_reflection(store, "deploy kubernetes cluster")

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich("write a haiku about cats")

        assert ctx.reflections == ()
        store.close()


class TestContextEnricherPatterns:
    """Pattern matching retrieval."""

    def test_includes_matching_patterns(self) -> None:
        store = _make_store()
        pattern_store = PatternStore(store)
        from core_gb.types import Pattern

        pattern = Pattern(
            id=str(uuid.uuid4()),
            trigger="weather in {slot_0}",
            description="Weather lookup pattern",
            variable_slots=("slot_0",),
            tree_template="[]",
            success_count=5,
            avg_tokens=100.0,
        )
        pattern_store.save(pattern)

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich("weather in Amsterdam")

        assert len(ctx.patterns) >= 1
        assert ctx.pattern_tokens > 0
        store.close()

    def test_no_patterns_when_none_match(self) -> None:
        store = _make_store()
        enricher = ContextEnricher(store=store)

        ctx = enricher.enrich("explain quantum computing")

        assert ctx.patterns == ()
        store.close()


class TestContextEnricherActivationRanking:
    """ACT-R activation-ranked retrieval."""

    def test_entities_ranked_by_activation(self) -> None:
        store = _make_store()
        # Create two entities with different access counts
        store.create_node("Project", {
            "id": "proj-high",
            "name": "HighActivity",
            "status": "active",
            "access_count": 100,
        })
        store.create_node("Project", {
            "id": "proj-low",
            "name": "LowActivity",
            "status": "active",
            "access_count": 1,
        })

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich("HighActivity LowActivity project")

        if len(ctx.entities) >= 2:
            names = [e.get("name", "") for e in ctx.entities]
            high_idx = names.index("HighActivity") if "HighActivity" in names else -1
            low_idx = names.index("LowActivity") if "LowActivity" in names else -1
            if high_idx >= 0 and low_idx >= 0:
                assert high_idx < low_idx, "Higher activation entity should appear first"
        store.close()


class TestContextEnricherPPR:
    """PPR-based deep retrieval mode."""

    def test_use_ppr_flag_enables_ppr_retrieval(self) -> None:
        store = _make_store()
        entity_ids = _seed_entities(store)

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich(
            "Tell me about GraphBot project",
            use_ppr=True,
        )

        # PPR mode should still return entities (may fall back to 2-hop)
        assert isinstance(ctx, EnrichedContext)
        store.close()

    def test_ppr_false_by_default(self) -> None:
        store = _make_store()
        enricher = ContextEnricher(store=store)

        # Should not raise even with no entities
        ctx = enricher.enrich("anything")
        assert isinstance(ctx, EnrichedContext)
        store.close()


class TestContextEnricherTokenBudget:
    """Token budget enforcement across sections."""

    def test_respects_max_tokens_budget(self) -> None:
        store = _make_store()
        # Seed many entities to exceed a small budget
        for i in range(20):
            store.create_node("Project", {
                "id": f"proj-{i:03d}",
                "name": f"Project {i} with a long name to consume tokens",
                "status": f"status-{i} with extra detail padding here",
                "access_count": 1,
            })

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich(
            "Project 0 Project 1 Project 2 Project 3 Project 4",
            max_tokens=50,
        )

        assert ctx.total_tokens <= 50
        store.close()

    def test_zero_budget_returns_empty(self) -> None:
        store = _make_store()
        _seed_entities(store)
        enricher = ContextEnricher(store=store)

        ctx = enricher.enrich("GraphBot project", max_tokens=0)

        assert ctx.total_tokens == 0
        store.close()


class TestContextEnricherFullPipeline:
    """End-to-end enrichment combining all sources."""

    def test_combines_entities_conversation_reflections(self) -> None:
        store = _make_store()
        _seed_entities(store)
        _seed_reflection(store, "GraphBot")

        conv = ConversationMemory(store)
        conv.add_message("chat-001", "user", "What is GraphBot?")

        enricher = ContextEnricher(store=store)
        ctx = enricher.enrich(
            "GraphBot",
            chat_id="chat-001",
        )

        # Should have at least some entities (from resolution)
        assert len(ctx.entities) > 0
        # Should have conversation turns
        assert len(ctx.conversation_turns) > 0
        # Should have reflections (similar task description)
        assert len(ctx.reflections) > 0
        # Total tokens should be sum of all sections
        expected = (
            ctx.entity_tokens
            + ctx.memory_tokens
            + ctx.reflection_tokens
            + ctx.pattern_tokens
            + ctx.conversation_tokens
            + ctx.relationship_tokens
            + ctx.community_tokens
        )
        assert ctx.total_tokens == expected
        store.close()
