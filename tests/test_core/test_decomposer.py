"""Tests for the Decomposer class (T016)."""

from __future__ import annotations

import json
import uuid

import pytest

from core_gb.decomposer import Decomposer, validate_decomposition, validate_tree
from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _weather_tree_json() -> dict:
    """Valid weather-3-cities decomposition tree."""
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Get weather for 3 cities",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 2,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["w1", "w2", "w3", "agg"],
            },
            {
                "id": "w1",
                "description": "Get Amsterdam weather",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_ams"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "w2",
                "description": "Get London weather",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_lon"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "w3",
                "description": "Get Berlin weather",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_ber"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "agg",
                "description": "Summarize weather",
                "domain": "synthesis",
                "task_type": "WRITE",
                "complexity": 1,
                "depends_on": ["w1", "w2", "w3"],
                "provides": ["summary"],
                "consumes": ["weather_ams", "weather_lon", "weather_ber"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


def _weather_tree_str() -> str:
    """Valid weather tree as JSON string."""
    return json.dumps(_weather_tree_json())


class MockRouter:
    """Mock ModelRouter that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_count = 0

    async def route(self, task: TaskNode, messages: list[dict]) -> CompletionResult:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return CompletionResult(
            content=self._responses[idx],
            model="mock",
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            cost=0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDecomposeValidResponse:
    """Test decompose() with a valid LLM response on the first try."""

    @pytest.mark.asyncio
    async def test_decompose_valid_response(self) -> None:
        router = MockRouter([_weather_tree_str()])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Weather in Amsterdam, London, Berlin")

        assert len(nodes) == 5
        assert all(isinstance(n, TaskNode) for n in nodes)

        # Verify parent/child relationships exist
        root = [n for n in nodes if n.parent_id is None]
        assert len(root) == 1
        assert root[0].description == "Get weather for 3 cities"
        assert len(root[0].children) == 4

        # All children reference root as parent
        children = [n for n in nodes if n.parent_id is not None]
        assert len(children) == 4
        for child in children:
            assert child.parent_id == root[0].id


class TestDecomposeInvalidThenRepaired:
    """Test that retry with json_repair works when first attempt is invalid."""

    @pytest.mark.asyncio
    async def test_decompose_invalid_then_repaired(self) -> None:
        invalid_json = "not valid json at all {{"
        valid_json = _weather_tree_str()
        router = MockRouter([invalid_json, valid_json])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Weather in 3 cities")

        assert len(nodes) == 5
        assert all(isinstance(n, TaskNode) for n in nodes)


class TestDecomposeBothFailFallback:
    """Test that two failures produce a single-node fallback."""

    @pytest.mark.asyncio
    async def test_decompose_both_fail_fallback(self) -> None:
        router = MockRouter(["garbage!!!", "more garbage!!!"])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Do something complex")

        assert len(nodes) == 1
        assert nodes[0].is_atomic is True
        assert nodes[0].description == "Do something complex"
        assert nodes[0].domain == Domain.SYNTHESIS
        assert nodes[0].status == TaskStatus.READY


class TestToTaskNodesMapsIds:
    """Test that original string IDs are mapped to UUIDs."""

    def test_to_task_nodes_maps_ids(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)

        raw_nodes = _weather_tree_json()["nodes"]
        task_nodes = decomposer._to_task_nodes(raw_nodes)

        # All IDs should be valid UUIDs (not the original string IDs)
        original_ids = {"root", "w1", "w2", "w3", "agg"}
        for node in task_nodes:
            assert node.id not in original_ids
            # Should be a valid UUID
            uuid.UUID(node.id)


class TestToTaskNodesSetsParent:
    """Test that parent_id is correctly set based on children references."""

    def test_to_task_nodes_sets_parent(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)

        raw_nodes = _weather_tree_json()["nodes"]
        task_nodes = decomposer._to_task_nodes(raw_nodes)

        # Build lookup
        by_desc = {n.description: n for n in task_nodes}
        root = by_desc["Get weather for 3 cities"]

        # Root has no parent
        assert root.parent_id is None

        # All leaf nodes have root as parent
        for desc in ["Get Amsterdam weather", "Get London weather",
                      "Get Berlin weather", "Summarize weather"]:
            node = by_desc[desc]
            assert node.parent_id == root.id


class TestToTaskNodesMAPSDomain:
    """Test that domain strings are mapped to Domain enum values."""

    def test_to_task_nodes_maps_domain(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)

        raw_nodes = _weather_tree_json()["nodes"]
        task_nodes = decomposer._to_task_nodes(raw_nodes)

        by_desc = {n.description: n for n in task_nodes}

        assert by_desc["Get weather for 3 cities"].domain == Domain.SYNTHESIS
        assert by_desc["Get Amsterdam weather"].domain == Domain.WEB
        assert by_desc["Summarize weather"].domain == Domain.SYNTHESIS

    def test_invalid_domain_falls_back_to_synthesis(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)

        raw_nodes = [
            {
                "id": "root",
                "description": "Test",
                "domain": "nonexistent_domain",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            }
        ]
        task_nodes = decomposer._to_task_nodes(raw_nodes)
        assert task_nodes[0].domain == Domain.SYNTHESIS


class TestFallbackSingleNode:
    """Test the fallback mechanism produces a single atomic node."""

    def test_fallback_single_node(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)

        nodes = decomposer._fallback_single_node("Write a poem about cats")

        assert len(nodes) == 1
        node = nodes[0]
        assert node.is_atomic is True
        assert node.description == "Write a poem about cats"
        assert node.domain == Domain.SYNTHESIS
        assert node.complexity == 1
        assert node.status == TaskStatus.READY
        # ID should be a valid UUID
        uuid.UUID(node.id)


class TestDecomposeParallelTree:
    """Test full decomposition of the weather-3-cities tree."""

    @pytest.mark.asyncio
    async def test_decompose_parallel_tree(self) -> None:
        router = MockRouter([_weather_tree_str()])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Weather in Amsterdam, London, Berlin")

        # Should produce 5 TaskNodes (root + 3 weather + aggregator)
        assert len(nodes) == 5

        by_desc = {n.description: n for n in nodes}
        root = by_desc["Get weather for 3 cities"]

        # Root is not atomic, has 4 children
        assert root.is_atomic is False
        assert len(root.children) == 4

        # Weather nodes are atomic with web domain
        for city in ["Get Amsterdam weather", "Get London weather", "Get Berlin weather"]:
            node = by_desc[city]
            assert node.is_atomic is True
            assert node.domain == Domain.WEB
            assert node.parent_id == root.id

        # Aggregator depends on 3 weather nodes and is atomic
        agg = by_desc["Summarize weather"]
        assert agg.is_atomic is True
        assert agg.domain == Domain.SYNTHESIS
        assert len(agg.requires) == 3
        assert len(agg.consumes) == 3

        # All requires point to valid node IDs in the tree
        all_ids = {n.id for n in nodes}
        for req_id in agg.requires:
            assert req_id in all_ids
