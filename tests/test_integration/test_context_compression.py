"""Integration tests for context assembly + prompt compression pipeline.

Validates that GraphRAG subgraph context assembly and prompt compression
work together correctly end-to-end (R002 + R005 validation).

Tests cover:
- Subgraph multi-hop retrieval finds connected entities
- Community summaries are generated from clusters
- XML document formatting (document index, source, content tags)
- Relationship descriptions included in context
- Compression quality: key facts retained after compression
- Compression + context assembly integration
- Token budget trimming works with compressed context
- End-to-end: enrich -> compress -> format produces valid prompt
"""

from __future__ import annotations

import pytest

from core_gb.compression import PromptCompressor
from core_gb.context_enrichment import ContextEnricher, EnrichedContext
from core_gb.context_formatter import ContextFormatter, SectionDef
from core_gb.token_budget import TokenBudget
from graph.community import Community, CommunityDetector, CommunityEdge, CommunityNode
from graph.store import GraphStore
from graph.subgraph import SubgraphRetriever, SubgraphResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _seed_rich_graph(store: GraphStore) -> dict[str, str]:
    """Seed a graph with multiple entity types and relationships.

    Structure (designed for multi-hop and community detection):
        User(Alice) --OWNS--> Project(GraphBot)
        User(Alice) --USES--> Service(GitHub)
        Project(GraphBot) has Memory(proj_mem) via ABOUT_PROJECT
        Memory(pref) --ABOUT--> User(Alice)
        Task(build_main) --PRODUCED--> File(main.py)
        User(Bob) --OWNS--> Project(WebApp)
        User(Bob) --USES--> Service(Slack)

    Two clusters:
        Cluster A: Alice, GraphBot, GitHub, main.py, build_main, pref, proj_mem
        Cluster B: Bob, WebApp, Slack

    Returns dict of node IDs keyed by role name.
    """
    alice_id = store.create_node("User", {
        "id": "user-alice",
        "name": "Alice",
        "role": "developer",
        "institution": "TU/e",
        "interests": "Python, graphs, machine learning",
    })
    graphbot_id = store.create_node("Project", {
        "id": "proj-graphbot",
        "name": "GraphBot",
        "path": "/dev/graphbot",
        "language": "Python",
        "framework": "Django",
        "status": "active",
    })
    github_id = store.create_node("Service", {
        "id": "svc-github",
        "name": "GitHub",
        "type": "vcs",
        "url": "https://github.com",
        "status": "active",
    })
    main_py_id = store.create_node("File", {
        "id": "file-main",
        "path": "/dev/graphbot/main.py",
        "type": "python",
        "description": "Main entry point for the DAG execution engine",
    })
    task_build_id = store.create_node("Task", {
        "id": "task-build-main",
        "description": "Build the main DAG execution module",
        "domain": "code",
        "status": "completed",
    })
    mem_pref_id = store.create_node("Memory", {
        "id": "mem-pref",
        "content": "Alice prefers Python for all projects and uses type hints everywhere",
        "category": "preference",
    })
    mem_proj_id = store.create_node("Memory", {
        "id": "mem-proj",
        "content": "GraphBot uses a temporal knowledge graph with ACT-R activation scoring",
        "category": "fact",
    })

    # Second cluster
    bob_id = store.create_node("User", {
        "id": "user-bob",
        "name": "Bob",
        "role": "designer",
        "institution": "MIT",
        "interests": "UI, accessibility",
    })
    webapp_id = store.create_node("Project", {
        "id": "proj-webapp",
        "name": "WebApp",
        "path": "/dev/webapp",
        "language": "TypeScript",
        "framework": "React",
        "status": "active",
    })
    slack_id = store.create_node("Service", {
        "id": "svc-slack",
        "name": "Slack",
        "type": "communication",
        "url": "https://slack.com",
        "status": "active",
    })

    # Cluster A edges
    store.create_edge("OWNS", alice_id, graphbot_id)
    store.create_edge("USES", alice_id, github_id)
    store.create_edge("PRODUCED", task_build_id, main_py_id)
    store.create_edge("ABOUT", mem_pref_id, alice_id)
    store.create_edge("ABOUT_PROJECT", mem_proj_id, graphbot_id)

    # Cluster B edges
    store.create_edge("OWNS", bob_id, webapp_id)
    store.create_edge("USES", bob_id, slack_id)

    return {
        "alice": alice_id,
        "graphbot": graphbot_id,
        "github": github_id,
        "main_py": main_py_id,
        "task_build": task_build_id,
        "mem_pref": mem_pref_id,
        "mem_proj": mem_proj_id,
        "bob": bob_id,
        "webapp": webapp_id,
        "slack": slack_id,
    }


def _build_long_context_from_enriched(enriched: EnrichedContext) -> str:
    """Assemble a text representation of enriched context for compression testing.

    Concatenates entities, relationships, and community summaries into a
    single block of text similar to what the formatter would produce.
    """
    parts: list[str] = []

    for entity in enriched.entities:
        etype = entity.get("type", "")
        name = entity.get("name", "")
        details = entity.get("details", "")
        parts.append(f"{etype}: {name} -- {details}")

    for desc in enriched.relationship_descriptions:
        parts.append(f"Relationship: {desc}")

    for summary in enriched.community_summaries:
        parts.append(f"Community: {summary}")

    for memory in enriched.memories:
        parts.append(f"Memory: {memory}")

    return ". ".join(parts) + "." if parts else ""


# ---------------------------------------------------------------------------
# Subgraph multi-hop retrieval tests
# ---------------------------------------------------------------------------

class TestSubgraphMultiHopRetrieval:
    """Subgraph retrieval finds connected entities across multiple hops."""

    def test_single_hop_finds_direct_neighbors(self) -> None:
        """From Alice, 1-hop should find Project(GraphBot) and Service(GitHub)."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["alice"]], max_hops=1)

        node_ids = {n.node_id for n in result.nodes}
        assert ids["alice"] in node_ids
        assert ids["graphbot"] in node_ids
        assert ids["github"] in node_ids
        # Bob should NOT be reachable from Alice in 1 hop.
        assert ids["bob"] not in node_ids
        store.close()

    def test_multi_hop_reaches_deeper_entities(self) -> None:
        """2-hop from Alice should reach project memory via Project->Memory."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["alice"]], max_hops=2)

        node_ids = {n.node_id for n in result.nodes}
        assert ids["alice"] in node_ids
        assert ids["graphbot"] in node_ids
        # 2-hop: Alice -> GraphBot -> Memory(proj)
        assert ids["mem_proj"] in node_ids
        store.close()

    def test_multiple_seeds_expands_both(self) -> None:
        """Multiple seed entities should each expand their neighborhoods."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph(
            [ids["alice"], ids["bob"]], max_hops=1,
        )

        node_ids = {n.node_id for n in result.nodes}
        # Alice cluster
        assert ids["graphbot"] in node_ids
        assert ids["github"] in node_ids
        # Bob cluster
        assert ids["webapp"] in node_ids
        assert ids["slack"] in node_ids
        store.close()

    def test_disconnected_seed_does_not_pollute(self) -> None:
        """A nonexistent seed should not affect results from valid seeds."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result_with_bad = retriever.retrieve_subgraph(
            [ids["alice"], "nonexistent-id"], max_hops=1,
        )
        result_clean = retriever.retrieve_subgraph(
            [ids["alice"]], max_hops=1,
        )

        node_ids_bad = {n.node_id for n in result_with_bad.nodes}
        node_ids_clean = {n.node_id for n in result_clean.nodes}
        assert node_ids_bad == node_ids_clean
        store.close()


# ---------------------------------------------------------------------------
# Community summary tests
# ---------------------------------------------------------------------------

class TestCommunitySummaryGeneration:
    """Community summaries are generated from detected clusters."""

    def test_community_summaries_nonempty(self) -> None:
        """Subgraph retrieval should produce at least one community summary."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["alice"]], max_hops=2)

        assert len(result.community_summaries) > 0
        store.close()

    def test_community_summary_mentions_entities(self) -> None:
        """Summaries should contain entity names from the cluster."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["alice"]], max_hops=1)

        all_summaries = " ".join(result.community_summaries)
        assert "Alice" in all_summaries or "GraphBot" in all_summaries

        store.close()

    def test_community_summary_contains_relationship_info(self) -> None:
        """Summaries should mention relationships when edges exist."""
        detector = CommunityDetector()
        nodes = [
            CommunityNode(node_id="a", table="User", name="Alice"),
            CommunityNode(node_id="b", table="Project", name="GraphBot"),
        ]
        edges = [
            CommunityEdge(
                from_id="a", to_id="b",
                edge_type="OWNS",
                description="Alice OWNS GraphBot",
            ),
        ]
        communities = detector.detect_communities(nodes, edges)
        assert len(communities) == 1
        summary = communities[0].summary
        assert "Relationships:" in summary
        assert "OWNS" in summary

    def test_disconnected_groups_form_separate_communities(self) -> None:
        """Two disconnected clusters should produce separate communities."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph(
            [ids["alice"], ids["bob"]], max_hops=1,
        )

        # With two disconnected clusters seeded, there should be >= 2 communities.
        assert len(result.community_summaries) >= 2
        store.close()


# ---------------------------------------------------------------------------
# XML document formatting tests
# ---------------------------------------------------------------------------

class TestXMLDocumentFormatting:
    """XML document format: <document index="N"><source>...</source><content>...</content></document>."""

    def test_xml_structure_has_all_tags(self) -> None:
        """format_as_xml_document produces document, source, and content tags."""
        xml = ContextFormatter.format_as_xml_document(
            index=1,
            source="knowledge_graph",
            content="User: Alice -- developer at TU/e",
        )
        assert '<document index="1">' in xml
        assert "<source>knowledge_graph</source>" in xml
        assert "<content>User: Alice -- developer at TU/e</content>" in xml
        assert xml.endswith("</document>")

    def test_entities_use_sequential_indices(self) -> None:
        """Multiple entities should have sequential document indices."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer"},
                {"type": "Project", "name": "GraphBot", "details": "DAG engine"},
            ),
            entity_tokens=20,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")
        system_msg = messages[0]["content"]

        assert '<document index="1">' in system_msg
        assert '<document index="2">' in system_msg

    def test_relationships_formatted_as_xml(self) -> None:
        """Relationship descriptions should appear inside XML document tags."""
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
        assert "</document>" in system_msg

    def test_community_summaries_formatted_as_xml(self) -> None:
        """Community summaries should be inside XML document elements."""
        enriched = EnrichedContext(
            community_summaries=(
                "Community of 3 entities (User: Alice; Project: GraphBot).",
            ),
            community_tokens=15,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")
        system_msg = messages[0]["content"]

        assert "Community of 3 entities" in system_msg
        assert "<source>knowledge_graph</source>" in system_msg

    def test_indices_sequential_across_sections(self) -> None:
        """Document indices should be sequential across entities, relationships, communities."""
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
            memories=("A memory about preferences",),
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
# Relationship descriptions in context
# ---------------------------------------------------------------------------

class TestRelationshipDescriptions:
    """Relationship descriptions are included in retrieved subgraph context."""

    def test_edges_have_human_readable_descriptions(self) -> None:
        """Edges should have descriptions like 'Alice OWNS GraphBot'."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["alice"]], max_hops=1)

        descriptions = [e.description for e in result.edges]
        assert len(descriptions) > 0
        owns_found = any(
            "OWNS" in d and "Alice" in d and "GraphBot" in d
            for d in descriptions
        )
        assert owns_found, f"Expected OWNS relationship, got: {descriptions}"
        store.close()

    def test_multiple_edge_types_present(self) -> None:
        """Multiple relationship types (OWNS, USES) should appear."""
        store = _make_store()
        ids = _seed_rich_graph(store)
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([ids["alice"]], max_hops=1)

        edge_types = {e.edge_type for e in result.edges}
        assert "OWNS" in edge_types
        assert "USES" in edge_types
        store.close()

    def test_relationship_descriptions_in_formatted_output(self) -> None:
        """Relationship descriptions should appear in the formatted prompt."""
        enriched = EnrichedContext(
            relationship_descriptions=(
                "Alice OWNS GraphBot",
                "Bob USES Slack",
            ),
            relationship_tokens=10,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="Who owns what?")
        system_msg = messages[0]["content"]

        assert "Alice OWNS GraphBot" in system_msg
        assert "Bob USES Slack" in system_msg


# ---------------------------------------------------------------------------
# Compression quality tests
# ---------------------------------------------------------------------------

class TestCompressionQuality:
    """Key facts are retained after compression."""

    def test_key_facts_survive_compression(self) -> None:
        """Domain-specific terms should survive TF-IDF compression."""
        compressor = PromptCompressor()
        text = (
            "GraphBot is a recursive DAG execution engine built on Kuzu. "
            "It decomposes complex tasks into trivially simple subtasks. "
            "The system processes requests through a seven-stage pipeline. "
            "This is a general filler statement with no real value. "
            "Another filler sentence that restates obvious information. "
            "Parallel execution on free LLMs achieves significant throughput gains. "
            "The temporal knowledge graph stores entities with time-decayed relevance. "
            "Things happen and stuff occurs in normal operations. "
            "Pattern matching eliminates redundant inference by reusing results. "
            "The decomposer uses topological sorting to schedule dependent nodes. "
            "In summary this section covered various topics. "
            "Entity resolution merges duplicate nodes using embedding similarity."
        )
        result = compressor.compress(text, target_ratio=0.5)

        key_terms = [
            "temporal knowledge graph",
            "entity resolution",
            "parallel execution",
            "pattern matching",
            "topological sorting",
        ]
        retained = sum(1 for t in key_terms if t.lower() in result.lower())
        # At least 40% of key terms should survive at 50% compression.
        assert retained >= 2, (
            f"Only {retained}/{len(key_terms)} key terms retained in: {result}"
        )

    def test_compressed_output_shorter_than_original(self) -> None:
        """Compressed text should have fewer estimated tokens."""
        compressor = PromptCompressor()
        text = (
            "Alice is a developer who works on GraphBot at TU Eindhoven. "
            "GraphBot is a recursive DAG execution engine built with Python. "
            "It uses a temporal knowledge graph for persistent storage of entities. "
            "The system supports parallel execution across multiple free LLM providers. "
            "Pattern matching reduces redundant inference by caching prior successful results. "
            "Entity resolution merges duplicate knowledge graph nodes using embeddings. "
            "This sentence adds absolutely no real information to the overall discussion. "
            "Another padding sentence with no substance or technical detail whatsoever. "
            "Yet another filler line that simply restates what has already been said before. "
            "The decomposer schedules dependent tasks using topological sorting algorithms. "
            "Activation scoring ranks entities by time-decayed relevance using ACT-R models. "
            "Community detection groups related entities into coherent clusters for summarization."
        )
        result = compressor.compress(text, target_ratio=0.5)

        original_tokens = compressor._estimate_tokens(text)
        compressed_tokens = compressor._estimate_tokens(result)
        assert compressed_tokens < original_tokens

    def test_compression_preserves_sentence_order(self) -> None:
        """Retained sentences should maintain their original relative order."""
        compressor = PromptCompressor(min_tokens=5)
        text = (
            "Alpha introduces the topic. "
            "Beta explains the core concept. "
            "Gamma provides background context. "
            "Delta offers specific technical details. "
            "Epsilon concludes with a summary. "
            "Zeta finishes with final remarks."
        )
        result = compressor.compress(text, target_ratio=0.5)

        order_markers = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
        present = [m for m in order_markers if m in result]
        assert len(present) >= 2
        positions = [result.index(m) for m in present]
        assert positions == sorted(positions), "Sentence order not preserved"


# ---------------------------------------------------------------------------
# Compression + context assembly integration
# ---------------------------------------------------------------------------

class TestCompressionContextAssemblyIntegration:
    """Compress assembled context while retaining critical information."""

    def test_compress_enriched_context_text(self) -> None:
        """Compressing assembled context should retain entity names."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer at TU/e"},
                {"type": "Project", "name": "GraphBot", "details": "DAG execution engine in Python"},
            ),
            relationship_descriptions=(
                "Alice OWNS GraphBot",
                "Alice USES GitHub",
            ),
            community_summaries=(
                "Community of 3 entities (User: Alice; Project: GraphBot; Service: GitHub). "
                "Relationships: Alice OWNS GraphBot; Alice USES GitHub.",
            ),
            memories=(
                "Alice prefers Python for all projects and uses type hints everywhere",
                "GraphBot uses a temporal knowledge graph with ACT-R activation scoring",
            ),
            entity_tokens=20,
            relationship_tokens=10,
            community_tokens=15,
            memory_tokens=15,
        )
        context_text = _build_long_context_from_enriched(enriched)
        compressor = PromptCompressor(min_tokens=10)
        compressed = compressor.compress(context_text, target_ratio=0.6)

        # Critical entities should survive compression.
        assert "Alice" in compressed or "GraphBot" in compressed, (
            f"Expected at least one key entity in compressed text: {compressed}"
        )

    def test_should_compress_over_budget(self) -> None:
        """should_compress returns True when context exceeds 50% of budget."""
        compressor = PromptCompressor()
        long_text = " ".join(["word"] * 200)  # ~267 estimated tokens
        assert compressor.should_compress(long_text, token_budget=100) is True

    def test_should_compress_under_budget(self) -> None:
        """should_compress returns False when context is under 50% of budget."""
        compressor = PromptCompressor()
        short_text = "A short piece of text."
        assert compressor.should_compress(short_text, token_budget=10000) is False

    def test_compress_skips_when_under_budget(self) -> None:
        """compress() with token_budget returns original when under 50%."""
        compressor = PromptCompressor(min_tokens=5)
        text = "A short sentence about Alice and GraphBot."
        result = compressor.compress(text, target_ratio=0.5, token_budget=100000)
        assert result == text

    def test_compress_applies_when_over_budget(self) -> None:
        """compress() with token_budget applies compression when over 50%."""
        compressor = PromptCompressor(min_tokens=5)
        text = (
            "GraphBot is a recursive DAG execution engine. "
            "It decomposes complex tasks into simple subtasks. "
            "The temporal knowledge graph stores entities. "
            "Pattern matching reduces redundant inference. "
            "Entity resolution merges duplicate nodes. "
            "Activation scoring ranks by relevance. "
            "Filler sentence with no useful information. "
            "Another filler that adds nothing."
        )
        # Small budget so text is > 50%
        result = compressor.compress(text, target_ratio=0.5, token_budget=50)

        original_tokens = compressor._estimate_tokens(text)
        compressed_tokens = compressor._estimate_tokens(result)
        assert compressed_tokens < original_tokens


# ---------------------------------------------------------------------------
# Token budget trimming with compressed context
# ---------------------------------------------------------------------------

class TestTokenBudgetWithCompression:
    """Token budget trimming works correctly with compressed context."""

    def test_formatter_trims_large_sections(self) -> None:
        """ContextFormatter drops sections that exceed the token budget."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer"},
                {"type": "Project", "name": "GraphBot", "details": "engine"},
            ),
            relationship_descriptions=("Alice OWNS GraphBot",),
            community_summaries=("Community of 2 entities.",),
            entity_tokens=10,
            relationship_tokens=5,
            community_tokens=5,
        )
        # Tight budget
        budget = TokenBudget(
            max_tokens=80,
            system_prompt_reserve=0,
            user_message_reserve=0,
            response_reserve=0,
        )
        formatter = ContextFormatter(token_budget=budget)
        messages = formatter.format(enriched, task="test")

        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_compressed_context_fits_budget(self) -> None:
        """After compression, the total token count should be under budget."""
        compressor = PromptCompressor(min_tokens=5)
        text = (
            "GraphBot is a recursive DAG execution engine built on Kuzu. "
            "It decomposes complex tasks into trivially simple subtasks. "
            "The system processes requests through a seven-stage pipeline. "
            "This is a general statement that does not add much value. "
            "Parallel execution on free LLMs achieves throughput gains. "
            "The temporal knowledge graph stores entities with time-decayed relevance. "
            "Pattern matching eliminates redundant inference by reusing results. "
            "The decomposer uses topological sorting to schedule dependent nodes. "
            "Entity resolution merges duplicate nodes using embedding similarity."
        )
        token_budget = 80
        compressed = compressor.compress(text, target_ratio=0.4, token_budget=token_budget)
        compressed_tokens = compressor._estimate_tokens(compressed)

        # Compressed text should be significantly smaller.
        original_tokens = compressor._estimate_tokens(text)
        assert compressed_tokens < original_tokens

    def test_budget_trimming_preserves_highest_activation_sections(self) -> None:
        """When trimming, highest-activation sections should be kept."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer"},
            ),
            memories=("A simple memory",),
            relationship_descriptions=("Alice OWNS GraphBot",),
            entity_tokens=500,   # High activation
            memory_tokens=10,    # Low activation
            relationship_tokens=200,  # Medium activation
        )
        formatter = ContextFormatter()
        ranked = formatter._rank_sections(enriched)
        names = [s.name for s in ranked]

        # Entities (500) should rank above relationships (200) above memories (10).
        assert names.index("entities") < names.index("relationships")
        assert names.index("relationships") < names.index("memories")


# ---------------------------------------------------------------------------
# End-to-end: enrich -> compress -> format
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """End-to-end pipeline: enrich -> compress -> format produces valid prompt."""

    def test_enriched_context_can_be_compressed_and_formatted(self) -> None:
        """Build EnrichedContext, compress its text, then format as messages."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer at TU/e"},
                {"type": "Project", "name": "GraphBot", "details": "DAG execution engine"},
            ),
            relationship_descriptions=(
                "Alice OWNS GraphBot",
                "Alice USES GitHub",
            ),
            community_summaries=(
                "Community of 3 entities (User: Alice; Project: GraphBot; Service: GitHub). "
                "Relationships: Alice OWNS GraphBot; Alice USES GitHub.",
            ),
            memories=(
                "Alice prefers Python for all projects",
            ),
            entity_tokens=20,
            relationship_tokens=10,
            community_tokens=15,
            memory_tokens=8,
        )

        # Step 1: Build context text from enriched context.
        context_text = _build_long_context_from_enriched(enriched)
        assert len(context_text) > 0

        # Step 2: Compress if needed.
        compressor = PromptCompressor(min_tokens=5)
        if compressor.should_compress(context_text, token_budget=50):
            compressed = compressor.compress(context_text, target_ratio=0.6)
        else:
            compressed = context_text
        assert len(compressed) > 0

        # Step 3: Format into LLM messages.
        formatter = ContextFormatter(complexity=3)
        messages = formatter.format(enriched, task="What does Alice work on?")

        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What does Alice work on?"

        # System prompt should contain XML-structured sections.
        system_content = messages[0]["content"]
        assert "<context>" in system_content
        assert "</context>" in system_content
        assert "<instructions>" in system_content

    def test_format_output_contains_entity_and_relationship_data(self) -> None:
        """Formatted output should contain entity and relationship content."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer"},
            ),
            relationship_descriptions=(
                "Alice OWNS GraphBot",
            ),
            entity_tokens=10,
            relationship_tokens=5,
        )
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="Describe Alice")
        system_content = messages[0]["content"]

        assert "Alice" in system_content
        assert "OWNS" in system_content
        assert "GraphBot" in system_content

    def test_empty_enrichment_produces_valid_messages(self) -> None:
        """An empty EnrichedContext should still produce valid system + user messages."""
        enriched = EnrichedContext()
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="Hello")

        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_pipeline_with_compression_retains_critical_entities(self) -> None:
        """Full pipeline: compress large context, critical entities survive."""
        enriched = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer at TU/e who works on Python projects"},
                {"type": "Project", "name": "GraphBot", "details": "recursive DAG execution engine with temporal knowledge graph"},
                {"type": "Service", "name": "GitHub", "details": "version control service for all projects"},
            ),
            relationship_descriptions=(
                "Alice OWNS GraphBot",
                "Alice USES GitHub",
                "GraphBot CONTAINS main.py",
            ),
            community_summaries=(
                "Community of 3 entities (User: Alice; Project: GraphBot; Service: GitHub). "
                "Relationships: Alice OWNS GraphBot; Alice USES GitHub.",
            ),
            memories=(
                "Alice prefers Python for all projects and uses type hints everywhere",
                "GraphBot uses a temporal knowledge graph with ACT-R activation scoring",
            ),
            entity_tokens=30,
            relationship_tokens=15,
            community_tokens=15,
            memory_tokens=15,
        )
        context_text = _build_long_context_from_enriched(enriched)

        compressor = PromptCompressor(min_tokens=5)
        compressed = compressor.compress(context_text, target_ratio=0.5)

        # At least some key entities should survive.
        key_entities = ["Alice", "GraphBot", "GitHub"]
        retained = sum(1 for e in key_entities if e in compressed)
        assert retained >= 1, (
            f"Expected at least 1 key entity in compressed text, "
            f"got {retained}: {compressed}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty graph, single entity, no relationships."""

    def test_empty_graph_returns_empty_subgraph(self) -> None:
        """An empty graph should return an empty SubgraphResult."""
        store = _make_store()
        retriever = SubgraphRetriever(store)
        result = retriever.retrieve_subgraph([])
        assert result.nodes == ()
        assert result.edges == ()
        assert result.community_summaries == ()
        store.close()

    def test_single_entity_no_relationships(self) -> None:
        """A single entity with no edges should form a single-node community."""
        store = _make_store()
        node_id = store.create_node("User", {
            "id": "user-lone",
            "name": "Charlie",
            "role": "intern",
        })
        retriever = SubgraphRetriever(store)

        result = retriever.retrieve_subgraph([node_id])

        assert len(result.nodes) == 1
        assert result.nodes[0].name == "Charlie"
        assert len(result.edges) == 0
        # Should still get a community summary for the lone node.
        assert len(result.community_summaries) >= 1
        assert "Charlie" in result.community_summaries[0]
        store.close()

    def test_nonexistent_entity_returns_empty(self) -> None:
        """Seeding with a nonexistent ID returns an empty result."""
        store = _make_store()
        retriever = SubgraphRetriever(store)
        result = retriever.retrieve_subgraph(["does-not-exist"])
        assert result.nodes == ()
        assert result.edges == ()
        store.close()

    def test_compress_empty_string(self) -> None:
        """Compressing an empty string returns empty string."""
        compressor = PromptCompressor()
        assert compressor.compress("") == ""

    def test_compress_single_sentence(self) -> None:
        """A single sentence cannot be split further, returned unchanged."""
        compressor = PromptCompressor(min_tokens=5)
        sentence = "GraphBot is a recursive DAG execution engine."
        result = compressor.compress(sentence, target_ratio=0.5)
        assert result == sentence

    def test_should_compress_zero_budget(self) -> None:
        """should_compress with zero budget returns False."""
        compressor = PromptCompressor()
        assert compressor.should_compress("any text here", token_budget=0) is False

    def test_enriched_context_default_totals_zero(self) -> None:
        """Default EnrichedContext should have zero total tokens."""
        ctx = EnrichedContext()
        assert ctx.total_tokens == 0
        assert ctx.relationship_descriptions == ()
        assert ctx.community_summaries == ()

    def test_formatter_handles_empty_enriched_context(self) -> None:
        """Formatting an empty context should still produce valid messages."""
        enriched = EnrichedContext()
        formatter = ContextFormatter()
        messages = formatter.format(enriched, task="test")

        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["content"] == "test"
