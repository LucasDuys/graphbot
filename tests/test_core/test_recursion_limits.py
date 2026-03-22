"""Tests for recursion hard limits in Decomposer.

Validates that decomposition depth is capped at 5 and total node count at 50.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core_gb.decomposer import validate_decomposition, _compute_depth


# ---------------------------------------------------------------------------
# Depth limit tests
# ---------------------------------------------------------------------------


class TestMaxDepthLimit:
    """Decomposition depth must not exceed 5 levels."""

    def _make_tree(self, depth: int) -> dict[str, Any]:
        """Build a decomposition tree with exactly `depth` levels.

        Creates a linear chain: root -> c1 -> c2 -> ... -> leaf.
        """
        nodes: list[dict[str, Any]] = []
        ids = [f"n{i}" for i in range(depth)]

        for i, node_id in enumerate(ids):
            is_leaf = i == depth - 1
            children = [ids[i + 1]] if not is_leaf else []
            nodes.append({
                "id": node_id,
                "description": f"Level {i + 1} node",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [f"data_{i}"] if is_leaf else [],
                "consumes": [],
                "is_atomic": is_leaf,
                "children": children,
            })

        return {"nodes": nodes}

    def test_depth_5_accepted(self) -> None:
        """A tree with depth 5 should be accepted."""
        tree = self._make_tree(5)
        errors = validate_decomposition(tree, max_depth=5)
        assert not errors, f"Expected no errors, got: {errors}"

    def test_depth_6_rejected(self) -> None:
        """A tree with depth 6 should be rejected."""
        tree = self._make_tree(6)
        errors = validate_decomposition(tree, max_depth=5)
        assert any("depth" in e.lower() for e in errors), (
            f"Expected depth error, got: {errors}"
        )

    def test_depth_10_rejected(self) -> None:
        """A deeply nested tree (depth 10) should be rejected."""
        tree = self._make_tree(10)
        errors = validate_decomposition(tree, max_depth=5)
        assert any("depth" in e.lower() for e in errors)

    def test_depth_1_accepted(self) -> None:
        """A flat tree (depth 1) should always be accepted."""
        tree = self._make_tree(1)
        errors = validate_decomposition(tree, max_depth=5)
        assert not errors

    def test_depth_3_accepted_under_default(self) -> None:
        """Default max_depth of 3 accepts depth 3."""
        tree = self._make_tree(3)
        errors = validate_decomposition(tree, max_depth=3)
        assert not errors

    def test_compute_depth_empty(self) -> None:
        """_compute_depth returns 0 for empty list."""
        assert _compute_depth([]) == 0

    def test_compute_depth_linear_chain(self) -> None:
        """_compute_depth computes correct depth for a linear chain."""
        nodes = [
            {"id": "a", "children": ["b"]},
            {"id": "b", "children": ["c"]},
            {"id": "c", "children": []},
        ]
        assert _compute_depth(nodes) == 3


# ---------------------------------------------------------------------------
# Node count limit tests
# ---------------------------------------------------------------------------


class TestMaxNodeCountLimit:
    """Decomposition must not exceed 50 total nodes."""

    def _make_wide_tree(self, node_count: int) -> dict[str, Any]:
        """Build a flat tree with one root and N-1 leaf children."""
        child_ids = [f"leaf_{i}" for i in range(node_count - 1)]
        nodes: list[dict[str, Any]] = [
            {
                "id": "root",
                "description": "Root node",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 2,
                "depends_on": [],
                "provides": [],
                "consumes": [],
                "is_atomic": False,
                "children": child_ids[:5],  # Schema limits children to 5
            }
        ]
        for i, cid in enumerate(child_ids):
            nodes.append({
                "id": cid,
                "description": f"Leaf task {i}",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [f"data_{i}"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            })
        return {"nodes": nodes}

    def test_50_nodes_accepted(self) -> None:
        """Exactly 50 nodes should be accepted."""
        tree = self._make_wide_tree(50)
        errors = validate_decomposition(tree, max_depth=5)
        # Filter out tree structure errors (children constraint) -- we only care about node count
        node_errors = [e for e in errors if "node" in e.lower() and "exceed" in e.lower()]
        assert not node_errors, f"Expected no node count errors, got: {node_errors}"

    def test_51_nodes_rejected(self) -> None:
        """51 nodes should be rejected."""
        tree = self._make_wide_tree(51)
        errors = validate_decomposition(tree, max_depth=5)
        node_errors = [e for e in errors if "node" in e.lower() and "exceed" in e.lower()]
        assert node_errors, f"Expected node count error for 51 nodes"

    def test_100_nodes_rejected(self) -> None:
        """100 nodes should be rejected."""
        tree = self._make_wide_tree(100)
        errors = validate_decomposition(tree, max_depth=5)
        node_errors = [e for e in errors if "node" in e.lower() and "exceed" in e.lower()]
        assert node_errors

    def test_10_nodes_accepted(self) -> None:
        """10 nodes is well within limits."""
        tree = self._make_wide_tree(10)
        errors = validate_decomposition(tree, max_depth=5)
        node_errors = [e for e in errors if "node" in e.lower() and "exceed" in e.lower()]
        assert not node_errors


# ---------------------------------------------------------------------------
# Decomposer enforces limits
# ---------------------------------------------------------------------------


class TestDecomposerEnforcesLimits:
    """The Decomposer class rejects plans that exceed recursion/node limits."""

    async def test_decomposer_rejects_deep_tree(self) -> None:
        """Decomposer falls back to single node when tree exceeds depth 5."""
        from core_gb.decomposer import Decomposer
        from core_gb.types import CompletionResult
        from models.base import ModelProvider
        from models.router import ModelRouter

        # Build a tree with depth 7 (exceeds limit of 5)
        nodes_raw: list[dict[str, Any]] = []
        for i in range(7):
            is_leaf = i == 6
            children = [f"n{i + 1}"] if not is_leaf else []
            nodes_raw.append({
                "id": f"n{i}",
                "description": f"Level {i} node",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [f"data_{i}"] if is_leaf else [],
                "consumes": [],
                "is_atomic": is_leaf,
                "children": children,
            })

        deep_tree = json.dumps({"nodes": nodes_raw})

        class MockProvider(ModelProvider):
            @property
            def name(self) -> str:
                return "mock"

            async def complete(
                self, messages: list[dict], model: str, **kwargs: object
            ) -> CompletionResult:
                return CompletionResult(
                    content=deep_tree,
                    model="mock",
                    tokens_in=10,
                    tokens_out=50,
                    latency_ms=10.0,
                    cost=0.001,
                )

        provider = MockProvider()
        router = ModelRouter(provider=provider)
        decomposer = Decomposer(router)

        result = await decomposer.decompose("Some complex task", max_depth=5)

        # Decomposer should fall back to single node because depth 7 > 5
        assert len(result) == 1
        assert result[0].is_atomic is True

    async def test_decomposer_rejects_oversized_dag(self) -> None:
        """Decomposer falls back when total nodes exceed 50."""
        from core_gb.decomposer import Decomposer
        from core_gb.types import CompletionResult
        from models.base import ModelProvider
        from models.router import ModelRouter

        # Build a flat tree with 55 nodes (exceeds limit of 50)
        nodes_raw: list[dict[str, Any]] = []
        leaf_ids = [f"leaf_{i}" for i in range(54)]
        nodes_raw.append({
            "id": "root",
            "description": "Root node",
            "domain": "synthesis",
            "task_type": "THINK",
            "complexity": 2,
            "depends_on": [],
            "provides": [],
            "consumes": [],
            "is_atomic": False,
            "children": leaf_ids[:5],
        })
        for i, lid in enumerate(leaf_ids):
            nodes_raw.append({
                "id": lid,
                "description": f"Leaf {i}",
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [f"data_{i}"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            })

        big_tree = json.dumps({"nodes": nodes_raw})

        class MockProvider(ModelProvider):
            @property
            def name(self) -> str:
                return "mock"

            async def complete(
                self, messages: list[dict], model: str, **kwargs: object
            ) -> CompletionResult:
                return CompletionResult(
                    content=big_tree,
                    model="mock",
                    tokens_in=10,
                    tokens_out=50,
                    latency_ms=10.0,
                    cost=0.001,
                )

        provider = MockProvider()
        router = ModelRouter(provider=provider)
        decomposer = Decomposer(router)

        result = await decomposer.decompose("Some task with many steps", max_depth=5)

        # Should fall back to single node because 55 > 50
        assert len(result) == 1
        assert result[0].is_atomic is True
