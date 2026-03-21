"""Tests for decomposition schema validation."""

from __future__ import annotations

import jsonschema

from core_gb.decomposer import DECOMPOSITION_SCHEMA, validate_decomposition


# ---------------------------------------------------------------------------
# Valid trees
# ---------------------------------------------------------------------------

def _parallel_weather_tree() -> dict:
    """Parallel: weather in 3 cities (root + 3 independent leaves + aggregator)."""
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
                "description": "Get weather for Amsterdam",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_amsterdam"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "w2",
                "description": "Get weather for Berlin",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_berlin"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "w3",
                "description": "Get weather for Paris",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["weather_paris"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "agg",
                "description": "Aggregate weather results",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": ["w1", "w2", "w3"],
                "provides": ["weather_summary"],
                "consumes": ["weather_amsterdam", "weather_berlin", "weather_paris"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


def _sequential_file_tree() -> dict:
    """Sequential: read file -> parse -> format (root + 3 sequential nodes)."""
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Read and format file contents",
                "domain": "file",
                "task_type": "THINK",
                "complexity": 2,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["read", "parse", "format"],
            },
            {
                "id": "read",
                "description": "Read the input file",
                "domain": "file",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["raw_content"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "parse",
                "description": "Parse raw content into structured data",
                "domain": "code",
                "task_type": "CODE",
                "complexity": 2,
                "depends_on": ["read"],
                "provides": ["parsed_data"],
                "consumes": ["raw_content"],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "format",
                "description": "Format parsed data for output",
                "domain": "synthesis",
                "task_type": "WRITE",
                "complexity": 1,
                "depends_on": ["parse"],
                "provides": ["formatted_output"],
                "consumes": ["parsed_data"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


def _mixed_research_tree() -> dict:
    """Mixed: parallel data gathering + sequential synthesis."""
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Research topic and produce report",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 3,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["gather", "synthesize"],
            },
            {
                "id": "gather",
                "description": "Gather data from multiple sources",
                "domain": "web",
                "task_type": "THINK",
                "complexity": 2,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["src_a", "src_b"],
            },
            {
                "id": "src_a",
                "description": "Search academic papers",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["papers"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "src_b",
                "description": "Search news articles",
                "domain": "web",
                "task_type": "RETRIEVE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["articles"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "synthesize",
                "description": "Write research summary",
                "domain": "synthesis",
                "task_type": "WRITE",
                "complexity": 2,
                "depends_on": ["gather"],
                "provides": ["report"],
                "consumes": ["papers", "articles"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


def _minimal_valid_tree() -> dict:
    """Minimal valid tree: single root node with no children."""
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Simple atomic task",
                "domain": "system",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": ["result"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            }
        ]
    }


def _two_level_tree() -> dict:
    """Two-level tree: root with two children, both atomic."""
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Coordinate two subtasks",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 2,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["a", "b"],
            },
            {
                "id": "a",
                "description": "First subtask",
                "domain": "code",
                "task_type": "CODE",
                "complexity": 1,
                "depends_on": [],
                "provides": ["data_a"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
            {
                "id": "b",
                "description": "Second subtask",
                "domain": "code",
                "task_type": "CODE",
                "complexity": 1,
                "depends_on": ["a"],
                "provides": ["data_b"],
                "consumes": ["data_a"],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Invalid trees
# ---------------------------------------------------------------------------

def _missing_required_field() -> dict:
    """Missing 'description' on a node."""
    return {
        "nodes": [
            {
                "id": "root",
                # description is missing
                "domain": "system",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            }
        ]
    }


def _invalid_domain() -> dict:
    """Invalid domain value 'magic'."""
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Bad domain",
                "domain": "magic",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            }
        ]
    }


def _too_deep() -> dict:
    """Tree with depth 4 (exceeds max_depth=3).

    Depth counting: root=1, level_1=2, level_2=3, level_3=4.
    """
    return {
        "nodes": [
            {
                "id": "root",
                "description": "Level 0",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["l1"],
            },
            {
                "id": "l1",
                "description": "Level 1",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["l2"],
            },
            {
                "id": "l2",
                "description": "Level 2",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": ["l3"],
            },
            {
                "id": "l3",
                "description": "Level 3",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestSchemaIsValid:
    """The DECOMPOSITION_SCHEMA itself must be a valid JSON Schema."""

    def test_schema_is_valid_json_schema(self) -> None:
        jsonschema.Draft7Validator.check_schema(DECOMPOSITION_SCHEMA)


class TestValidTrees:
    """Five valid trees must all validate without errors."""

    def test_parallel_weather(self) -> None:
        errors = validate_decomposition(_parallel_weather_tree())
        assert errors == [], f"Unexpected errors: {errors}"

    def test_sequential_file(self) -> None:
        errors = validate_decomposition(_sequential_file_tree())
        assert errors == [], f"Unexpected errors: {errors}"

    def test_mixed_research(self) -> None:
        errors = validate_decomposition(_mixed_research_tree())
        assert errors == [], f"Unexpected errors: {errors}"

    def test_minimal_single_node(self) -> None:
        errors = validate_decomposition(_minimal_valid_tree())
        assert errors == [], f"Unexpected errors: {errors}"

    def test_two_level_tree(self) -> None:
        errors = validate_decomposition(_two_level_tree())
        assert errors == [], f"Unexpected errors: {errors}"


class TestInvalidTrees:
    """Invalid trees must produce clear error messages."""

    def test_missing_required_field(self) -> None:
        errors = validate_decomposition(_missing_required_field())
        assert len(errors) > 0
        assert any("description" in e.lower() or "required" in e.lower() for e in errors), (
            f"Expected error about missing 'description', got: {errors}"
        )

    def test_invalid_domain(self) -> None:
        errors = validate_decomposition(_invalid_domain())
        assert len(errors) > 0
        assert any("magic" in e.lower() or "domain" in e.lower() or "enum" in e.lower() for e in errors), (
            f"Expected error about invalid domain, got: {errors}"
        )

    def test_too_deep(self) -> None:
        errors = validate_decomposition(_too_deep())
        assert len(errors) > 0
        assert any("depth" in e.lower() for e in errors), (
            f"Expected error about depth, got: {errors}"
        )


class TestMaxChildrenConstraint:
    """Nodes must not exceed 5 children."""

    def test_too_many_children(self) -> None:
        tree = {
            "nodes": [
                {
                    "id": "root",
                    "description": "Too many kids",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["c1", "c2", "c3", "c4", "c5", "c6"],
                },
                *[
                    {
                        "id": f"c{i}",
                        "description": f"Child {i}",
                        "domain": "system",
                        "task_type": "THINK",
                        "complexity": 1,
                        "depends_on": [],
                        "provides": [],
                        "consumes": [],
                        "is_atomic": True,
                        "children": [],
                    }
                    for i in range(1, 7)
                ],
            ]
        }
        errors = validate_decomposition(tree)
        assert len(errors) > 0
        assert any("children" in e.lower() or "5" in e for e in errors), (
            f"Expected error about max children, got: {errors}"
        )


class TestConfigurableMaxDepth:
    """max_depth parameter should be respected."""

    def test_custom_max_depth_allows_deeper(self) -> None:
        tree = _too_deep()
        errors = validate_decomposition(tree, max_depth=4)
        assert not any("depth" in e.lower() for e in errors)

    def test_custom_max_depth_restricts_shallower(self) -> None:
        tree = _mixed_research_tree()  # depth 3
        errors = validate_decomposition(tree, max_depth=2)
        assert any("depth" in e.lower() for e in errors)
