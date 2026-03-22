"""Tests for ConsolidationEngine -- dedup, merge, and summarize memory nodes."""

from __future__ import annotations

import pytest

from graph.consolidation import ConsolidationEngine
from graph.resolver import EntityResolver
from graph.store import GraphStore


@pytest.fixture
def store() -> GraphStore:
    """Provide an initialized in-memory GraphStore."""
    s = GraphStore(db_path=None)
    s.initialize()
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def engine(store: GraphStore) -> ConsolidationEngine:
    """Provide a ConsolidationEngine backed by the store."""
    return ConsolidationEngine(store)


class TestDetectDuplicates:
    """Duplicate detection via EntityResolver 3-layer matching."""

    def test_exact_name_duplicates_detected(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Two contacts with the same name are detected as duplicates."""
        store.create_node("Contact", {
            "id": "c-1",
            "name": "Alice Smith",
            "relationship": "classmate",
        })
        store.create_node("Contact", {
            "id": "c-2",
            "name": "Alice Smith",
            "relationship": "friend",
        })

        duplicates = engine.detect_duplicates()
        assert len(duplicates) >= 1
        # At least one group should contain both IDs
        found = False
        for group in duplicates:
            ids = {node_id for node_id, _table in group}
            if "c-1" in ids and "c-2" in ids:
                found = True
                break
        assert found, "Expected c-1 and c-2 to be in the same duplicate group"

    def test_fuzzy_name_duplicates_detected(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Two projects with near-identical names are detected as duplicates."""
        store.create_node("Project", {
            "id": "p-1",
            "name": "GraphBot",
            "language": "Python",
        })
        store.create_node("Project", {
            "id": "p-2",
            "name": "Graphbot",
            "language": "Python",
            "status": "active",
        })

        duplicates = engine.detect_duplicates()
        assert len(duplicates) >= 1
        found = False
        for group in duplicates:
            ids = {node_id for node_id, _table in group}
            if "p-1" in ids and "p-2" in ids:
                found = True
                break
        assert found, "Expected p-1 and p-2 to be in the same duplicate group"

    def test_no_duplicates_returns_empty(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Distinct entities produce no duplicate groups."""
        store.create_node("Contact", {"id": "c-1", "name": "Alice Smith"})
        store.create_node("Contact", {"id": "c-2", "name": "Bob Johnson"})
        store.create_node("Project", {"id": "p-1", "name": "GraphBot"})

        duplicates = engine.detect_duplicates()
        assert len(duplicates) == 0


class TestMergeDuplicates:
    """Merging duplicate entities: combine properties, redirect edges, delete old."""

    def test_merge_combines_properties(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Merge fills missing properties from the secondary node into the primary."""
        store.create_node("Contact", {
            "id": "c-1",
            "name": "Alice Smith",
            "relationship": "classmate",
        })
        store.create_node("Contact", {
            "id": "c-2",
            "name": "Alice Smith",
            "platform": "Discord",
        })

        engine.merge("Contact", "c-1", "c-2")

        primary = store.get_node("Contact", "c-1")
        assert primary is not None
        assert primary["name"] == "Alice Smith"
        assert primary["relationship"] == "classmate"
        assert primary["platform"] == "Discord"

        # Secondary node is deleted
        assert store.get_node("Contact", "c-2") is None

    def test_merge_primary_properties_take_precedence(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """When both nodes have a value for the same property, primary wins."""
        store.create_node("Contact", {
            "id": "c-1",
            "name": "Alice Smith",
            "relationship": "classmate",
        })
        store.create_node("Contact", {
            "id": "c-2",
            "name": "Alice Smyth",
            "relationship": "friend",
        })

        engine.merge("Contact", "c-1", "c-2")

        primary = store.get_node("Contact", "c-1")
        assert primary is not None
        # Primary's relationship should be preserved
        assert primary["relationship"] == "classmate"

    def test_merge_redirects_outgoing_edges(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Edges from the secondary node are redirected to the primary."""
        # Memory --ABOUT--> User
        store.create_node("User", {"id": "u-1", "name": "Lucas"})
        store.create_node("User", {"id": "u-2", "name": "lucas"})
        store.create_node("Memory", {"id": "m-1", "content": "Likes Python"})

        store.create_edge("ABOUT", "m-1", "u-2")

        engine.merge("User", "u-1", "u-2")

        # Edge should now point to u-1
        rows = store.query(
            "MATCH (m:Memory)-[:ABOUT]->(u:User) WHERE m.id = $mid RETURN u.id",
            {"mid": "m-1"},
        )
        assert len(rows) >= 1
        target_ids = [row["u.id"] for row in rows]
        assert "u-1" in target_ids

        # Old node should be gone
        assert store.get_node("User", "u-2") is None

    def test_merge_redirects_incoming_edges(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Edges going into the secondary node are redirected to the primary."""
        # User --OWNS--> Project
        store.create_node("User", {"id": "u-1", "name": "Lucas"})
        store.create_node("Project", {"id": "p-1", "name": "GraphBot"})
        store.create_node("Project", {"id": "p-2", "name": "Graphbot"})

        store.create_edge("OWNS", "u-1", "p-2")

        engine.merge("Project", "p-1", "p-2")

        # Edge should now point to p-1
        rows = store.query(
            "MATCH (u:User)-[:OWNS]->(p:Project) WHERE u.id = $uid RETURN p.id",
            {"uid": "u-1"},
        )
        assert len(rows) >= 1
        target_ids = [row["p.id"] for row in rows]
        assert "p-1" in target_ids

        # Old node should be gone
        assert store.get_node("Project", "p-2") is None


class TestSummarize:
    """Summary Memory node generation for clusters of related memories."""

    def test_summary_node_created(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Summarize creates a Memory node with combined content."""
        store.create_node("User", {"id": "u-1", "name": "Lucas"})

        m1_id = store.create_node("Memory", {
            "id": "m-1",
            "content": "User prefers dark mode.",
            "category": "preference",
        })
        m2_id = store.create_node("Memory", {
            "id": "m-2",
            "content": "User likes Python programming.",
            "category": "preference",
        })
        m3_id = store.create_node("Memory", {
            "id": "m-3",
            "content": "User uses VS Code editor.",
            "category": "preference",
        })

        # Connect memories to user
        store.create_edge("ABOUT", "m-1", "u-1")
        store.create_edge("ABOUT", "m-2", "u-1")
        store.create_edge("ABOUT", "m-3", "u-1")

        summary_id = engine.summarize(["m-1", "m-2", "m-3"])

        summary = store.get_node("Memory", summary_id)
        assert summary is not None
        assert summary["category"] == "summary"

        # Summary content should mention all original memory contents
        content = str(summary["content"])
        assert "dark mode" in content
        assert "Python" in content
        assert "VS Code" in content

    def test_summary_linked_to_user(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """Summary Memory node is linked to the same entity as the source memories."""
        store.create_node("User", {"id": "u-1", "name": "Lucas"})
        store.create_node("Memory", {"id": "m-1", "content": "Fact A"})
        store.create_node("Memory", {"id": "m-2", "content": "Fact B"})

        store.create_edge("ABOUT", "m-1", "u-1")
        store.create_edge("ABOUT", "m-2", "u-1")

        summary_id = engine.summarize(["m-1", "m-2"])

        # Summary should be linked to u-1 via ABOUT
        rows = store.query(
            "MATCH (m:Memory)-[:ABOUT]->(u:User) WHERE m.id = $mid RETURN u.id",
            {"mid": summary_id},
        )
        assert len(rows) >= 1
        assert rows[0]["u.id"] == "u-1"

    def test_summary_with_no_memories_returns_none(
        self, engine: ConsolidationEngine,
    ) -> None:
        """Summarize with an empty list returns None."""
        result = engine.summarize([])
        assert result is None

    def test_summary_with_single_memory_returns_none(
        self, store: GraphStore, engine: ConsolidationEngine,
    ) -> None:
        """Summarize with a single memory returns None -- nothing to consolidate."""
        store.create_node("Memory", {"id": "m-1", "content": "Only one"})
        result = engine.summarize(["m-1"])
        assert result is None


class TestRunConsolidation:
    """Full consolidation pass: dedup + merge + summarize."""

    def test_full_run_merges_duplicates(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """run() detects and merges duplicate entities."""
        store.create_node("Contact", {
            "id": "c-1",
            "name": "Alice Smith",
            "relationship": "classmate",
        })
        store.create_node("Contact", {
            "id": "c-2",
            "name": "Alice Smith",
            "platform": "Discord",
        })

        result = engine.run()

        assert result.merged_count >= 1
        # One of the two should remain with combined properties
        remaining = store.get_node("Contact", "c-1")
        assert remaining is not None

    def test_full_run_creates_summaries_for_related_memories(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """run() creates summary nodes for clusters of related memories."""
        store.create_node("User", {"id": "u-1", "name": "Lucas"})

        for i in range(5):
            mid = f"m-{i}"
            store.create_node("Memory", {
                "id": mid,
                "content": f"Preference fact number {i}.",
                "category": "preference",
            })
            store.create_edge("ABOUT", mid, "u-1")

        result = engine.run()

        assert result.summaries_created >= 1

    def test_run_edges_preserved_after_merge(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """After a full run, edges from merged nodes are preserved on the survivor."""
        store.create_node("User", {"id": "u-1", "name": "Lucas"})
        store.create_node("User", {"id": "u-2", "name": "Lucas"})
        store.create_node("Project", {"id": "p-1", "name": "GraphBot"})

        store.create_edge("OWNS", "u-2", "p-1")

        engine.run()

        # u-2 should be gone, u-1 should own p-1
        assert store.get_node("User", "u-2") is None
        rows = store.query(
            "MATCH (u:User)-[:OWNS]->(p:Project) WHERE u.id = $uid RETURN p.id",
            {"uid": "u-1"},
        )
        assert len(rows) >= 1
        assert rows[0]["p.id"] == "p-1"

    def test_run_returns_result_dataclass(
        self, store: GraphStore, engine: ConsolidationEngine
    ) -> None:
        """run() returns a ConsolidationResult with proper counts."""
        result = engine.run()

        assert hasattr(result, "merged_count")
        assert hasattr(result, "summaries_created")
        assert hasattr(result, "duplicates_found")
        assert isinstance(result.merged_count, int)
        assert isinstance(result.summaries_created, int)
        assert isinstance(result.duplicates_found, int)


class TestCallable:
    """ConsolidationEngine is callable (script or background)."""

    def test_engine_is_callable(self, engine: ConsolidationEngine) -> None:
        """ConsolidationEngine supports __call__ protocol."""
        result = engine()
        assert hasattr(result, "merged_count")
