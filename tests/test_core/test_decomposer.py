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
        self._kwargs_log: list[dict] = []

    async def route(
        self, task: TaskNode, messages: list[dict], **kwargs: object
    ) -> CompletionResult:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        self._kwargs_log.append(kwargs)
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


class TestDecomposeRepairsFirstResponse:
    """Test that json_repair is tried on the first response before a second LLM call."""

    @pytest.mark.asyncio
    async def test_decompose_repairs_first_response(self) -> None:
        # Almost-valid JSON: trailing comma after last node, missing closing bracket
        tree = _weather_tree_json()
        almost_valid = json.dumps(tree)[:-2] + ",]}"  # inject trailing comma
        router = MockRouter([almost_valid])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Weather in 3 cities")

        # json_repair should fix the trailing comma and produce valid nodes
        assert len(nodes) == 5
        assert all(isinstance(n, TaskNode) for n in nodes)
        # Only one LLM call should have been made (repair worked on first response)
        assert router._call_count == 1


class TestDecomposeJsonModePassedThrough:
    """Test that response_format is passed through to the router."""

    @pytest.mark.asyncio
    async def test_decompose_json_mode_passed(self) -> None:
        router = MockRouter([_weather_tree_str()])
        decomposer = Decomposer(router)

        await decomposer.decompose("Weather in 3 cities")

        # The first (and only) call should include response_format
        assert len(router._kwargs_log) == 1
        assert router._kwargs_log[0].get("response_format") == {"type": "json_object"}


def _weather_tree_with_template_json() -> dict:
    """Valid weather tree with output_template."""
    tree = _weather_tree_json()
    tree["output_template"] = {
        "aggregation_type": "template_fill",
        "template": "## Weather Comparison\n\n### Amsterdam\n{weather_ams}\n\n### London\n{weather_lon}\n\n### Berlin\n{weather_ber}",
        "slot_definitions": {
            "weather_ams": "Current weather in Amsterdam",
            "weather_lon": "Current weather in London",
            "weather_ber": "Current weather in Berlin",
        },
    }
    return tree


def _weather_tree_with_template_str() -> str:
    """Valid weather tree with output_template as JSON string."""
    return json.dumps(_weather_tree_with_template_json())


class TestLastTemplateSet:
    """Test that Decomposer.last_template is set when valid template in response."""

    @pytest.mark.asyncio
    async def test_last_template_set_on_valid_response(self) -> None:
        router = MockRouter([_weather_tree_with_template_str()])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Weather in Amsterdam, London, Berlin")

        assert len(nodes) == 5
        assert decomposer.last_template is not None
        assert decomposer.last_template["aggregation_type"] == "template_fill"
        assert "weather_ams" in decomposer.last_template["template"]
        assert "weather_ams" in decomposer.last_template["slot_definitions"]

    @pytest.mark.asyncio
    async def test_last_template_none_when_no_template(self) -> None:
        router = MockRouter([_weather_tree_str()])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Weather in Amsterdam, London, Berlin")

        assert len(nodes) == 5
        assert decomposer.last_template is None

    @pytest.mark.asyncio
    async def test_last_template_reset_between_calls(self) -> None:
        router = MockRouter([
            _weather_tree_with_template_str(),
            _weather_tree_str(),
        ])
        decomposer = Decomposer(router)

        await decomposer.decompose("First call")
        assert decomposer.last_template is not None

        await decomposer.decompose("Second call")
        assert decomposer.last_template is None

    @pytest.mark.asyncio
    async def test_last_template_none_on_fallback(self) -> None:
        router = MockRouter(["garbage!!!", "more garbage!!!"])
        decomposer = Decomposer(router)

        nodes = await decomposer.decompose("Do something")

        assert len(nodes) == 1
        assert decomposer.last_template is None

    def test_last_template_initial_value(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)
        assert decomposer.last_template is None


class TestFixMissingFields:
    """Test _fix_missing_fields fills in sensible defaults."""

    def test_fills_defaults(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)

        incomplete_nodes = [
            {"id": "root", "description": "Test", "children": ["a"]},
            {"description": "Child task"},
        ]
        fixed = decomposer._fix_missing_fields(incomplete_nodes)

        assert len(fixed) == 2
        # First node: has children so is_atomic should be False
        assert fixed[0]["is_atomic"] is False
        assert fixed[0]["domain"] == "synthesis"
        assert fixed[0]["task_type"] == "THINK"
        assert fixed[0]["complexity"] == 1
        assert fixed[0]["depends_on"] == []
        assert fixed[0]["provides"] == []
        assert fixed[0]["consumes"] == []

        # Second node: no children so is_atomic should be True, auto-generated ID
        assert fixed[1]["id"] == "node_1"
        assert fixed[1]["is_atomic"] is True
        assert fixed[1]["children"] == []

    def test_preserves_existing_fields(self) -> None:
        router = MockRouter([])
        decomposer = Decomposer(router)

        nodes = [
            {
                "id": "custom_id",
                "description": "Already complete",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 3,
                "depends_on": ["other"],
                "provides": ["data"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            }
        ]
        fixed = decomposer._fix_missing_fields(nodes)

        assert fixed[0]["id"] == "custom_id"
        assert fixed[0]["domain"] == "web"
        assert fixed[0]["task_type"] == "RETRIEVE"
        assert fixed[0]["complexity"] == 3
        assert fixed[0]["is_atomic"] is True
