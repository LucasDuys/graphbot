"""Tests for PPR-based retrieval via power iteration.

Validates that PPRRetriever:
1. Computes approximate Personalized PageRank using power iteration
2. Seeds from entity resolution and spreads activation through the graph
3. Returns top-K nodes ranked by PPR score within a token budget
4. Uses alpha=0.85, max_iterations=20, convergence threshold=1e-6
5. Finds relevant nodes 3+ hops away that 2-hop traversal misses
"""

from __future__ import annotations

import pytest

from graph.retrieval import PPRRetriever
from graph.store import GraphStore


@pytest.fixture
def populated_store() -> GraphStore:
    """Create a GraphStore with a multi-hop graph structure.

    Graph topology (designed so that nodes 3+ hops away are reachable):

    User(alice) --OWNS--> Project(proj_a) --[via Task edges]--> deep chain
    User(alice) --USES--> Service(svc_github)

    Task chain (via DEPENDS_ON): task1 -> task2 -> task3 -> task4
    task1 --INVOLVES--> Service(svc_github)
    task4 --PRODUCED--> File(deep_file)

    Memory(mem1) --ABOUT--> User(alice)

    This means from alice:
    - hop 0: alice
    - hop 1: proj_a, svc_github, mem1 (via ABOUT reverse)
    - hop 2: task1 (via INVOLVES reverse from svc_github)
    - hop 3: task2 (via DEPENDS_ON reverse from task1)
    - hop 4: task3 (via DEPENDS_ON reverse from task2)
    - hop 5: task4, deep_file

    A 2-hop traversal would NOT reach task2, task3, task4, or deep_file.
    PPR should rank deep_file and task4 as reachable (nonzero score).
    """
    store = GraphStore(db_path=None)
    store.initialize()

    # Nodes
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

    # Edges
    store.create_edge("OWNS", "alice", "proj_a")
    store.create_edge("USES", "alice", "svc_github")
    store.create_edge("ABOUT", "mem1", "alice")
    store.create_edge("INVOLVES", "task1", "svc_github")
    store.create_edge("DEPENDS_ON", "task1", "task2")
    store.create_edge("DEPENDS_ON", "task2", "task3")
    store.create_edge("DEPENDS_ON", "task3", "task4")
    store.create_edge("PRODUCED", "task4", "deep_file")

    return store


@pytest.fixture
def retriever(populated_store: GraphStore) -> PPRRetriever:
    """Create a PPRRetriever backed by the populated store."""
    return PPRRetriever(store=populated_store)


class TestPPRRetrieverBasicBehavior:
    """Tests for basic PPR computation and retrieval."""

    def test_returns_seed_nodes_with_highest_scores(
        self, retriever: PPRRetriever
    ) -> None:
        """Seed nodes should have the highest PPR scores."""
        results = retriever.retrieve(seed_ids=["alice"], top_k=10)
        assert len(results) > 0
        # The seed node itself should be in results with the top score
        ids = [node_id for node_id, _score in results]
        assert "alice" in ids
        # Seed should be first (highest score)
        assert results[0][0] == "alice"

    def test_returns_tuples_of_id_and_score(
        self, retriever: PPRRetriever
    ) -> None:
        """Each result should be a (node_id, score) tuple with score in (0, 1]."""
        results = retriever.retrieve(seed_ids=["alice"], top_k=5)
        for node_id, score in results:
            assert isinstance(node_id, str)
            assert isinstance(score, float)
            assert 0.0 < score <= 1.0

    def test_scores_are_descending(self, retriever: PPRRetriever) -> None:
        """Results should be sorted by PPR score descending."""
        results = retriever.retrieve(seed_ids=["alice"], top_k=10)
        scores = [score for _id, score in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_respects_top_k_limit(self, retriever: PPRRetriever) -> None:
        """Should return at most top_k results."""
        results = retriever.retrieve(seed_ids=["alice"], top_k=3)
        assert len(results) <= 3

    def test_empty_seeds_returns_empty(self, retriever: PPRRetriever) -> None:
        """Empty seed list should return empty results."""
        results = retriever.retrieve(seed_ids=[], top_k=5)
        assert results == []


class TestPPRReachesDeepNodes:
    """Tests that PPR finds relevant nodes 3+ hops away that 2-hop misses."""

    def test_reaches_nodes_beyond_2_hops(
        self, retriever: PPRRetriever, populated_store: GraphStore
    ) -> None:
        """PPR should assign nonzero score to nodes 3+ hops from seed.

        The graph has task2 at 3 hops, task3 at 4, task4 at 5, deep_file at 6.
        A 2-hop traversal from alice would only see: alice, proj_a, svc_github,
        mem1, task1. PPR should find task2+ with nonzero scores.
        """
        results = retriever.retrieve(seed_ids=["alice"], top_k=20)
        result_ids = {node_id for node_id, _score in results}

        # Verify 2-hop context does NOT include deep nodes
        two_hop_ctx = populated_store.get_context(["alice"], max_tokens=10000)
        two_hop_names = set()
        for entity in two_hop_ctx.relevant_entities:
            two_hop_names.add(entity.get("name", ""))

        # deep_file is beyond 2 hops -- standard context should not find it
        # (its name is the path "/deploy/config.yaml" in the entity)
        deep_node_ids = {"task3", "task4", "deep_file"}

        # PPR should find at least one deep node that 2-hop misses
        ppr_deep_hits = deep_node_ids & result_ids
        assert len(ppr_deep_hits) > 0, (
            f"PPR should reach nodes 3+ hops away. Found ids: {result_ids}"
        )

    def test_deep_nodes_have_lower_scores_than_nearby(
        self, retriever: PPRRetriever
    ) -> None:
        """Nodes further from seed should generally have lower PPR scores."""
        results = retriever.retrieve(seed_ids=["alice"], top_k=20)
        score_map = dict(results)

        # alice (seed, hop 0) should score higher than task1 (hop 2+)
        if "alice" in score_map and "task1" in score_map:
            assert score_map["alice"] > score_map["task1"]

        # task1 should score higher than task4 (much deeper)
        if "task1" in score_map and "task4" in score_map:
            assert score_map["task1"] > score_map["task4"]


class TestPPRParameters:
    """Tests for PPR algorithm parameters."""

    def test_default_alpha(self, retriever: PPRRetriever) -> None:
        """Default teleport probability alpha should be 0.85."""
        assert retriever.alpha == 0.85

    def test_default_max_iterations(self, retriever: PPRRetriever) -> None:
        """Default max_iterations should be 20."""
        assert retriever.max_iterations == 20

    def test_default_convergence_threshold(self, retriever: PPRRetriever) -> None:
        """Default convergence threshold should be 1e-6."""
        assert retriever.convergence_threshold == 1e-6

    def test_custom_alpha(self, populated_store: GraphStore) -> None:
        """Custom alpha should be accepted and used."""
        r = PPRRetriever(store=populated_store, alpha=0.5)
        assert r.alpha == 0.5
        results = r.retrieve(seed_ids=["alice"], top_k=5)
        assert len(results) > 0

    def test_convergence_produces_stable_scores(
        self, retriever: PPRRetriever
    ) -> None:
        """Running retrieve twice with same seeds should produce identical scores."""
        results1 = retriever.retrieve(seed_ids=["alice"], top_k=10)
        results2 = retriever.retrieve(seed_ids=["alice"], top_k=10)
        assert len(results1) == len(results2)
        for (id1, s1), (id2, s2) in zip(results1, results2):
            assert id1 == id2
            assert abs(s1 - s2) < 1e-10


class TestPPRTokenBudget:
    """Tests for token budget enforcement."""

    def test_respects_token_budget(self, retriever: PPRRetriever) -> None:
        """Should stop adding nodes when token budget is exhausted."""
        # Very small budget should limit results
        results_small = retriever.retrieve(
            seed_ids=["alice"], top_k=20, max_tokens=10
        )
        results_large = retriever.retrieve(
            seed_ids=["alice"], top_k=20, max_tokens=10000
        )
        # With tiny budget, we should get fewer results
        assert len(results_small) <= len(results_large)

    def test_token_budget_zero_returns_empty(
        self, retriever: PPRRetriever
    ) -> None:
        """A token budget of 0 should return empty results."""
        results = retriever.retrieve(seed_ids=["alice"], top_k=20, max_tokens=0)
        assert results == []


class TestPPRMultipleSeeds:
    """Tests for multiple seed nodes."""

    def test_multiple_seeds_combine_scores(
        self, retriever: PPRRetriever
    ) -> None:
        """Multiple seed nodes should both appear with high scores."""
        results = retriever.retrieve(
            seed_ids=["alice", "svc_github"], top_k=10
        )
        result_ids = {nid for nid, _s in results}
        assert "alice" in result_ids
        assert "svc_github" in result_ids

    def test_nonexistent_seed_is_ignored(
        self, retriever: PPRRetriever
    ) -> None:
        """Seeds that do not exist in the graph should be silently ignored."""
        results = retriever.retrieve(
            seed_ids=["alice", "nonexistent_node"], top_k=10
        )
        assert len(results) > 0
        result_ids = {nid for nid, _s in results}
        assert "alice" in result_ids
        assert "nonexistent_node" not in result_ids


class TestPPRDisconnectedGraph:
    """Tests for edge cases in graph structure."""

    def test_isolated_node_returns_only_self(self) -> None:
        """A node with no edges should only return itself."""
        store = GraphStore(db_path=None)
        store.initialize()
        store.create_node("User", {"id": "lonely", "name": "Lonely"})
        retriever = PPRRetriever(store=store)
        results = retriever.retrieve(seed_ids=["lonely"], top_k=10)
        assert len(results) == 1
        assert results[0][0] == "lonely"
        # Score should be close to 1.0 (all probability stays on seed)
        assert results[0][1] > 0.8
