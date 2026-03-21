"""Tests for GraphUpdater -- records task outcomes in the knowledge graph."""

from __future__ import annotations

from core_gb.types import Domain, ExecutionResult, FlowType, TaskNode, TaskStatus
from graph.store import GraphStore
from graph.updater import GraphUpdater


def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _multi_node_list() -> list[TaskNode]:
    """Return a realistic multi-node task tree (weather in 3 cities)."""
    return [
        TaskNode(
            id="root",
            description="Weather in Amsterdam, London, and Berlin",
            children=["leaf_ams", "leaf_lon", "leaf_ber"],
            domain=Domain.SYNTHESIS,
            complexity=2,
            flow_type=FlowType.PARALLEL,
        ),
        TaskNode(
            id="leaf_ams",
            description="Current weather in Amsterdam",
            parent_id="root",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            provides=["weather_amsterdam"],
        ),
        TaskNode(
            id="leaf_lon",
            description="Current weather in London",
            parent_id="root",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            provides=["weather_london"],
        ),
        TaskNode(
            id="leaf_ber",
            description="Current weather in Berlin",
            parent_id="root",
            is_atomic=True,
            domain=Domain.WEB,
            complexity=1,
            provides=["weather_berlin"],
        ),
    ]


def _single_node_list() -> list[TaskNode]:
    """Return a single atomic node."""
    return [
        TaskNode(
            id="single",
            description="What is 2 + 2?",
            is_atomic=True,
            domain=Domain.SYSTEM,
            complexity=1,
        ),
    ]


def _success_result(root_id: str = "root", total_nodes: int = 4) -> ExecutionResult:
    return ExecutionResult(
        root_id=root_id,
        output="Sunny in all cities",
        success=True,
        total_nodes=total_nodes,
        total_tokens=1200,
        total_latency_ms=350.0,
        total_cost=0.002,
    )


def _failure_result(root_id: str = "root") -> ExecutionResult:
    return ExecutionResult(
        root_id=root_id,
        output="",
        success=False,
        total_nodes=4,
        total_tokens=800,
        total_latency_ms=200.0,
        errors=("API timeout",),
    )


class TestGraphUpdater:
    """Graph update loop records task outcomes correctly."""

    def test_records_task_node(self) -> None:
        """update() creates a Task node with correct fields."""
        store = _make_store()
        updater = GraphUpdater(store)
        result = _success_result()

        updater.update(
            "Weather in Amsterdam, London, and Berlin",
            _multi_node_list(),
            result,
        )

        node = store.get_node("Task", result.root_id)
        assert node is not None
        assert node["id"] == "root"
        assert node["status"] == "completed"
        assert node["tokens_used"] == 1200
        assert node["latency_ms"] == 350.0
        assert node["domain"] == "synthesis"
        assert len(str(node["description"])) > 0
        store.close()

    def test_records_execution_tree(self) -> None:
        """update() creates an ExecutionTree node with correct fields."""
        store = _make_store()
        updater = GraphUpdater(store)
        result = _success_result()

        updater.update(
            "Weather in Amsterdam, London, and Berlin",
            _multi_node_list(),
            result,
        )

        tree = store.get_node("ExecutionTree", f"tree_{result.root_id}")
        assert tree is not None
        assert tree["root_task_id"] == "root"
        assert tree["total_nodes"] == 4
        assert tree["total_tokens"] == 1200
        assert tree["total_latency_ms"] == 350.0
        store.close()

    def test_links_tree_to_task(self) -> None:
        """update() creates a DERIVED_FROM edge from ExecutionTree to Task."""
        store = _make_store()
        updater = GraphUpdater(store)
        result = _success_result()

        updater.update(
            "Weather in Amsterdam, London, and Berlin",
            _multi_node_list(),
            result,
        )

        rows = store.query(
            "MATCH (t:ExecutionTree)-[:DERIVED_FROM]->(k:Task) "
            "WHERE t.id = $tid AND k.id = $kid RETURN t.id, k.id",
            {"tid": f"tree_{result.root_id}", "kid": result.root_id},
        )
        assert len(rows) == 1
        store.close()

    def test_extracts_pattern(self) -> None:
        """Multi-node successful execution stores a PatternNode in the graph."""
        store = _make_store()
        updater = GraphUpdater(store)
        result = _success_result()

        pattern_id = updater.update(
            "Weather in Amsterdam, London, and Berlin",
            _multi_node_list(),
            result,
        )

        assert pattern_id is not None
        pattern_node = store.get_node("PatternNode", pattern_id)
        assert pattern_node is not None
        assert pattern_node["success_count"] == 1
        store.close()

    def test_no_pattern_for_single_node(self) -> None:
        """Single-node execution does not extract a pattern."""
        store = _make_store()
        updater = GraphUpdater(store)
        result = _success_result(root_id="single", total_nodes=1)

        pattern_id = updater.update(
            "What is 2 + 2?",
            _single_node_list(),
            result,
        )

        assert pattern_id is None
        store.close()

    def test_no_pattern_for_failure(self) -> None:
        """Failed execution does not extract a pattern."""
        store = _make_store()
        updater = GraphUpdater(store)
        result = _failure_result()

        pattern_id = updater.update(
            "Weather in Amsterdam, London, and Berlin",
            _multi_node_list(),
            result,
        )

        assert pattern_id is None

        # Task node should still be recorded with failed status
        node = store.get_node("Task", result.root_id)
        assert node is not None
        assert node["status"] == "failed"
        store.close()

    def test_returns_pattern_id(self) -> None:
        """update() returns the pattern ID when a pattern is extracted."""
        store = _make_store()
        updater = GraphUpdater(store)
        result = _success_result()

        pattern_id = updater.update(
            "Weather in Amsterdam, London, and Berlin",
            _multi_node_list(),
            result,
        )

        assert pattern_id is not None
        assert isinstance(pattern_id, str)
        assert len(pattern_id) > 0

        # Verify it matches an actual PatternNode
        pattern_node = store.get_node("PatternNode", pattern_id)
        assert pattern_node is not None
        assert pattern_node["id"] == pattern_id
        store.close()
