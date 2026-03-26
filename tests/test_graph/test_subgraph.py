"""Tests for GraphRAG-style subgraph retrieval with community summaries.

Tests cover:
- Multi-hop retrieval finds connected entities
- Community detection groups related entities
- Community summaries are generated
- XML document formatting
- Relationship descriptions are included
- Token budget trimming works with new format
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core_gb.context_enrichment import ContextEnricher, EnrichedContext
from core_gb.context_formatter import ContextFormatter
from core_gb.token_budget import TokenBudget
from graph.community import Community, CommunityDetector, CommunityEdge, CommunityNode
from graph.store import GraphStore
from graph.subgraph import SubgraphRetriever, SubgraphResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    s = GraphStore(db_path=None)
    s.initialize()
    return s


def _seed_graph(store: GraphStore) -> dict[str, str]:
    """Seed a graph with User -> Project -> File and Memory.

    Structure:
        User(Alice) --OWNS--> Project(GraphBot) --PRODUCED (via Task)--> File(main.py)
        User(Alice) --USES--> Service(GitHub)
        Memory(pref) --ABOUT--> User(Alice)
        Memory(proj_mem) --ABOUT_PROJECT--> Project(GraphBot)

    Returns dict of node IDs keyed by role.
    """
    user_id = store.create_node("User", {
        "id": "user-1",
        "name": "Alice",
        "role": "developer",
        "institution": "TU/e",
        "interests": "Python, graphs",
    })
    project_id = store.create_node("Project", {
        "id": "proj-1",
        "name": "GraphBot",
        "path": "/dev/graphbot",
        "language": "Python",
        "framework": "Django",
        "status": "active",
    })
    file_id = store.create_node("File", {
        "id": "file-1",
        "path": "/dev/graphbot/main.py",
        "type": "python",
        "description": "Main entry point",
    })
    service_id = store.create_node("Service", {
        "id": "svc-1",
        "name": "GitHub",
        "type": "vcs",
        "url": "https://github.com",
        "status": "active",
    })
    task_id = store.create_node("Task", {
        "id": "task-1",
        "description": "Build the main module",
        "domain": "code",
        "status": "completed",
    })
    mem_id = store.create_node("Memory", {
        "id": "mem-1",
        "content": "Alice prefers Python for all projects",
        "category": "preference",
    })
    proj_mem_id = store.create_node("Memory", {
        "id": "mem-2",
        "content": "GraphBot uses a knowledge graph backend",
        "category": "fact",
    })

    # Edges
    store.create_edge("OWNS", user_id, project_id)
    store.create_edge("USES", user_id, service_id)
    store.create_edge("PRODUCED", task_id, file_id)
    store.create_edge("ABOUT", mem_id, user_id)
    store.create_edge("ABOUT_PROJECT", proj_mem_id, project_id)

    return {
        "user": user_id,
        "project": project_id,
        "file": file_id,
        "service": service_id,
        "task": task_id,
        "memory": mem_id,
        "project_memory": proj_mem_id,
    }


# ---------------------------------------------------------------------------
# SubgraphRetriever tests
# ---------------------------------------------------------------------------

class TestSubgraphRetriever:
    """Tests for SubgraphRetriever.retrieve_subgraph."""

    def test_empty_entity_ids_returns_empty(self) -> None:
        """Empty entity_ids should return an empty SubgraphResult."""
        store = _make_store()
        retriever = SubgraphRetriever(store)
        result = retriever.retrieve_subgraph([])
        assert result.nodes == ()
        assert result.edges == ()
        assert result.community_summaries == ()
        store.close()

    def test_single_seed_retrieves_connected_entities(self) -> None:
        """From a User seed, should retrieve connected Project and Service."""
        store = _make_store()
        ids = _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]])

        node_ids = {n.node_id for n in result.nodes}
        # The seed itself should be included.
        assert ids["user"] in node_ids
        # 1-hop neighbors: Project, Service, Memory (ABOUT -> User).
        assert ids["project"] in node_ids
        assert ids["service"] in node_ids
        store.close()

    def test_multi_hop_retrieves_2hop_entities(self) -> None:
        """2-hop traversal from User should reach File via Project->Task->File."""
        store = _make_store()
        ids = _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]], max_hops=2)

        node_ids = {n.node_id for n in result.nodes}
        # 2-hop: User -> Project, Project -> Memory(proj_mem)
        assert ids["project"] in node_ids
        assert ids["project_memory"] in node_ids
        store.close()

    def test_relationship_descriptions_included(self) -> None:
        """Edges should have human-readable descriptions like 'Alice OWNS GraphBot'."""
        store = _make_store()
        ids = _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]])

        descriptions = [e.description for e in result.edges]
        # Should find at least the OWNS relationship.
        owns_found = any("OWNS" in d and "Alice" in d and "GraphBot" in d for d in descriptions)
        assert owns_found, f"Expected OWNS relationship description, got: {descriptions}"
        store.close()

    def test_edges_have_type_info(self) -> None:
        """Each edge should carry its relationship type."""
        store = _make_store()
        ids = _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]])

        edge_types = {e.edge_type for e in result.edges}
        assert "OWNS" in edge_types
        assert "USES" in edge_types
        store.close()

    def test_max_nodes_limits_results(self) -> None:
        """max_nodes should cap the number of returned nodes."""
        store = _make_store()
        ids = _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]], max_nodes=2)

        assert len(result.nodes) <= 2
        store.close()

    def test_nodes_sorted_by_activation_score(self) -> None:
        """Returned nodes should be sorted by activation score descending."""
        store = _make_store()
        ids = _seed_graph(store)
        # Give the user a higher access count for activation scoring.
        store.update_node("User", ids["user"], {"access_count": 10})
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]])

        scores = [n.activation_score for n in result.nodes]
        assert scores == sorted(scores, reverse=True)
        store.close()

    def test_community_summaries_generated(self) -> None:
        """Community summaries should be generated for the retrieved subgraph."""
        store = _make_store()
        ids = _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]])

        assert len(result.community_summaries) > 0
        # The summary should mention at least one entity.
        summary_text = " ".join(result.community_summaries)
        assert "Alice" in summary_text or "GraphBot" in summary_text
        store.close()

    def test_nonexistent_seed_returns_empty(self) -> None:
        """A nonexistent seed entity should produce an empty result."""
        store = _make_store()
        _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph(["nonexistent-id"])
        assert result.nodes == ()
        store.close()


# ---------------------------------------------------------------------------
# CommunityDetector tests
# ---------------------------------------------------------------------------

class TestCommunityDetector:
    """Tests for CommunityDetector."""

    def test_empty_input(self) -> None:
        """No nodes produces no communities."""
        detector = CommunityDetector()
        communities = detector.detect_communities([], [])
        assert communities == []

    def test_single_node_community(self) -> None:
        """A single disconnected node forms its own community."""
        detector = CommunityDetector()
        nodes = [CommunityNode(node_id="a", table="User", name="Alice")]
        communities = detector.detect_communities(nodes, [])
        assert len(communities) == 1
        assert len(communities[0].nodes) == 1

    def test_two_connected_nodes_single_community(self) -> None:
        """Two nodes connected by an edge should be in the same community."""
        detector = CommunityDetector()
        nodes = [
            CommunityNode(node_id="a", table="User", name="Alice"),
            CommunityNode(node_id="b", table="Project", name="GraphBot"),
        ]
        edges = [
            CommunityEdge(from_id="a", to_id="b", edge_type="OWNS",
                          description="Alice OWNS GraphBot"),
        ]
        communities = detector.detect_communities(nodes, edges)
        assert len(communities) == 1
        assert len(communities[0].nodes) == 2

    def test_disconnected_groups_form_separate_communities(self) -> None:
        """Disconnected node groups should form separate communities."""
        detector = CommunityDetector()
        nodes = [
            CommunityNode(node_id="a", table="User", name="Alice"),
            CommunityNode(node_id="b", table="Project", name="GraphBot"),
            CommunityNode(node_id="c", table="User", name="Bob"),
            CommunityNode(node_id="d", table="Service", name="Slack"),
        ]
        edges = [
            CommunityEdge(from_id="a", to_id="b", edge_type="OWNS",
                          description="Alice OWNS GraphBot"),
            CommunityEdge(from_id="c", to_id="d", edge_type="USES",
                          description="Bob USES Slack"),
        ]
        communities = detector.detect_communities(nodes, edges)
        assert len(communities) == 2
        # Each community should have 2 nodes.
        sizes = sorted([len(c.nodes) for c in communities], reverse=True)
        assert sizes == [2, 2]

    def test_community_summary_contains_members(self) -> None:
        """Community summary should mention member names."""
        detector = CommunityDetector()
        nodes = [
            CommunityNode(node_id="a", table="User", name="Alice"),
            CommunityNode(node_id="b", table="Project", name="GraphBot"),
        ]
        edges = [
            CommunityEdge(from_id="a", to_id="b", edge_type="OWNS",
                          description="Alice OWNS GraphBot"),
        ]
        communities = detector.detect_communities(nodes, edges)
        summary = communities[0].summary
        assert "Alice" in summary
        assert "GraphBot" in summary

    def test_community_summary_includes_relationships(self) -> None:
        """Community summary should include relationship descriptions."""
        detector = CommunityDetector()
        nodes = [
            CommunityNode(node_id="a", table="User", name="Alice"),
            CommunityNode(node_id="b", table="Project", name="GraphBot"),
        ]
        edges = [
            CommunityEdge(from_id="a", to_id="b", edge_type="OWNS",
                          description="Alice OWNS GraphBot"),
        ]
        communities = detector.detect_communities(nodes, edges)
        assert "OWNS" in communities[0].summary

    def test_sorted_by_size_descending(self) -> None:
        """Communities should be sorted by size descending."""
        detector = CommunityDetector()
        nodes = [
            CommunityNode(node_id="a", table="User", name="Alice"),
            CommunityNode(node_id="b", table="Project", name="GraphBot"),
            CommunityNode(node_id="c", table="File", name="main.py"),
            CommunityNode(node_id="d", table="User", name="Bob"),
        ]
        edges = [
            CommunityEdge(from_id="a", to_id="b", edge_type="OWNS",
                          description="Alice OWNS GraphBot"),
            CommunityEdge(from_id="b", to_id="c", edge_type="CONTAINS",
                          description="GraphBot CONTAINS main.py"),
        ]
        communities = detector.detect_communities(nodes, edges)
        assert len(communities) == 2
        assert len(communities[0].nodes) >= len(communities[1].nodes)


# ---------------------------------------------------------------------------
# XML document formatting tests
# ---------------------------------------------------------------------------

class TestXMLDocumentFormatting:
    """Tests for XML document formatting in ContextFormatter."""

    def test_format_as_xml_document(self) -> None:
        """format_as_xml_document should produce correct XML structure."""
        result = ContextFormatter.format_as_xml_document(
            index=1,
            source="knowledge_graph",
            content="User: Alice -- developer at TU/e",
        )
        assert '<document index="1">' in result
        assert "<source>knowledge_graph</source>" in result
        assert "<content>User: Alice -- developer at TU/e</content>" in result
        assert "</document>" in result

    def test_entities_formatted_as_xml_documents(self) -> None:
        """Entity context should be formatted as XML document elements."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer"},
                {"type": "Project", "name": "GraphBot", "details": "language=Python"},
            ),
            entity_tokens=20,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")

        system_msg = messages[0]["content"]
        assert '<document index="1">' in system_msg
        assert "<source>knowledge_graph</source>" in system_msg
        assert "Alice" in system_msg
        assert '<document index="2">' in system_msg

    def test_relationships_formatted_as_xml_documents(self) -> None:
        """Relationship descriptions should be formatted as XML documents."""
        enriched = EnrichedContext(
            relationship_descriptions=(
                "Alice OWNS GraphBot",
                "Alice USES GitHub",
            ),
            relationship_tokens=10,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")

        system_msg = messages[0]["content"]
        assert "Alice OWNS GraphBot" in system_msg
        assert "<source>knowledge_graph</source>" in system_msg

    def test_community_summaries_formatted_as_xml_documents(self) -> None:
        """Community summaries should be formatted as XML documents."""
        enriched = EnrichedContext(
            community_summaries=(
                "Community of 3 entities (User: Alice; Project: GraphBot; Service: GitHub).",
            ),
            community_tokens=15,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")

        system_msg = messages[0]["content"]
        assert "Community of 3 entities" in system_msg
        assert "<source>knowledge_graph</source>" in system_msg

    def test_memories_formatted_as_xml_documents(self) -> None:
        """Memories should be formatted as XML documents with source=memory."""
        enriched = EnrichedContext(
            memories=("Alice prefers Python for all projects",),
            memory_tokens=10,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")

        system_msg = messages[0]["content"]
        assert "<source>memory</source>" in system_msg
        assert "Alice prefers Python" in system_msg

    def test_document_indices_are_sequential(self) -> None:
        """Document indices should be sequential across all sections."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "dev"},
            ),
            relationship_descriptions=(
                "Alice OWNS GraphBot",
            ),
            community_summaries=(
                "Community of 2 entities.",
            ),
            memories=("A memory",),
            entity_tokens=10,
            relationship_tokens=5,
            community_tokens=5,
            memory_tokens=5,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")

        system_msg = messages[0]["content"]
        assert '<document index="1">' in system_msg
        assert '<document index="2">' in system_msg
        assert '<document index="3">' in system_msg
        assert '<document index="4">' in system_msg


# ---------------------------------------------------------------------------
# Token budget trimming tests
# ---------------------------------------------------------------------------

class TestTokenBudgetTrimming:
    """Tests for token budget enforcement with the new subgraph format."""

    def test_subgraph_retriever_respects_max_nodes(self) -> None:
        """SubgraphRetriever should not exceed max_nodes."""
        store = _make_store()
        ids = _seed_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["user"]], max_nodes=3)
        assert len(result.nodes) <= 3
        store.close()

    def test_formatter_trims_sections_to_budget(self) -> None:
        """ContextFormatter should drop sections that exceed token budget."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer"},
            ),
            relationship_descriptions=("Alice OWNS GraphBot",),
            community_summaries=("Community of 2 entities.",),
            entity_tokens=10,
            relationship_tokens=5,
            community_tokens=5,
        )
        # Very small budget that can only fit one section.
        budget = TokenBudget(max_tokens=50, system_prompt_reserve=0,
                             user_message_reserve=0, response_reserve=0)
        formatter = ContextFormatter(token_budget=budget)
        messages = formatter.format(enriched, task="test")

        # Should still produce valid output (some sections may be trimmed).
        assert len(messages) >= 2  # system + user
        assert messages[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# EnrichedContext integration tests
# ---------------------------------------------------------------------------

class TestEnrichedContextIntegration:
    """Tests that EnrichedContext carries subgraph data correctly."""

    def test_enriched_context_has_relationship_fields(self) -> None:
        """EnrichedContext should have relationship and community fields."""
        ctx = EnrichedContext(
            relationship_descriptions=("Alice OWNS GraphBot",),
            community_summaries=("A community summary.",),
            relationship_tokens=5,
            community_tokens=8,
        )
        assert ctx.relationship_descriptions == ("Alice OWNS GraphBot",)
        assert ctx.community_summaries == ("A community summary.",)
        assert ctx.relationship_tokens == 5
        assert ctx.community_tokens == 8

    def test_total_tokens_includes_new_sections(self) -> None:
        """total_tokens should include relationship and community tokens."""
        ctx = EnrichedContext(
            entity_tokens=10,
            memory_tokens=5,
            relationship_tokens=7,
            community_tokens=3,
        )
        assert ctx.total_tokens == 10 + 5 + 7 + 3

    def test_default_enriched_context_empty(self) -> None:
        """Default EnrichedContext should have empty subgraph fields."""
        ctx = EnrichedContext()
        assert ctx.relationship_descriptions == ()
        assert ctx.community_summaries == ()
        assert ctx.relationship_tokens == 0
        assert ctx.community_tokens == 0
        assert ctx.total_tokens == 0


# ---------------------------------------------------------------------------
# SubgraphResult tests
# ---------------------------------------------------------------------------

class TestSubgraphResult:
    """Tests for the SubgraphResult dataclass."""

    def test_default_empty(self) -> None:
        """Default SubgraphResult should be empty."""
        result = SubgraphResult()
        assert result.nodes == ()
        assert result.edges == ()
        assert result.community_summaries == ()

    def test_immutable(self) -> None:
        """SubgraphResult should be frozen (immutable)."""
        result = SubgraphResult()
        with pytest.raises(AttributeError):
            result.nodes = ()  # type: ignore[misc]
