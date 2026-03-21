"""Tests for structural tree validation of decomposition output."""

from __future__ import annotations

from core_gb.decomposer import validate_tree


# ---------------------------------------------------------------------------
# Helper: minimal node builder
# ---------------------------------------------------------------------------

def _node(
    id: str,
    *,
    children: list[str] | None = None,
    depends_on: list[str] | None = None,
    provides: list[str] | None = None,
    consumes: list[str] | None = None,
    is_atomic: bool | None = None,
    description: str = "",
    domain: str = "synthesis",
    task_type: str = "THINK",
    complexity: int = 1,
) -> dict:
    is_leaf = (children is None or len(children) == 0)
    return {
        "id": id,
        "description": description or f"Node {id}",
        "domain": domain,
        "task_type": task_type,
        "complexity": complexity,
        "depends_on": depends_on or [],
        "provides": provides or [],
        "consumes": consumes or [],
        "is_atomic": is_atomic if is_atomic is not None else is_leaf,
        "children": children or [],
    }


# ---------------------------------------------------------------------------
# Valid trees
# ---------------------------------------------------------------------------

class TestValidTrees:
    """Valid tree structures must produce zero errors."""

    def test_valid_parallel_tree(self) -> None:
        """Three independent leaves under a root."""
        nodes = [
            _node("root", children=["a", "b", "c"], is_atomic=False),
            _node("a", provides=["data_a"]),
            _node("b", provides=["data_b"]),
            _node("c", provides=["data_c"]),
        ]
        errors = validate_tree(nodes)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_valid_sequential_tree(self) -> None:
        """A -> B -> C chain under a root."""
        nodes = [
            _node("root", children=["a", "b", "c"], is_atomic=False),
            _node("a", provides=["x"]),
            _node("b", depends_on=["a"], provides=["y"], consumes=["x"]),
            _node("c", depends_on=["b"], provides=["z"], consumes=["y"]),
        ]
        errors = validate_tree(nodes)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_valid_mixed_tree(self) -> None:
        """Parallel gathering + sequential synthesis."""
        nodes = [
            _node("root", children=["gather", "synth"], is_atomic=False),
            _node("gather", children=["g1", "g2"], is_atomic=False),
            _node("g1", provides=["papers"]),
            _node("g2", provides=["articles"]),
            _node("synth", depends_on=["gather"], consumes=["papers", "articles"], provides=["report"]),
        ]
        errors = validate_tree(nodes)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_satisfied_data_contract(self) -> None:
        """Node consumes data_x, and a dependency provides it."""
        nodes = [
            _node("root", children=["producer", "consumer"], is_atomic=False),
            _node("producer", provides=["data_x"]),
            _node("consumer", depends_on=["producer"], consumes=["data_x"]),
        ]
        errors = validate_tree(nodes)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_single_atomic_root(self) -> None:
        """A single atomic root node is valid."""
        nodes = [_node("root", is_atomic=True)]
        errors = validate_tree(nodes)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_transitive_data_contract(self) -> None:
        """Node C consumes key provided by A, where C depends on B which depends on A."""
        nodes = [
            _node("root", children=["a", "b", "c"], is_atomic=False),
            _node("a", provides=["data_x"]),
            _node("b", depends_on=["a"], provides=["data_y"], consumes=["data_x"]),
            _node("c", depends_on=["b"], consumes=["data_y"]),
        ]
        errors = validate_tree(nodes)
        assert errors == [], f"Unexpected errors: {errors}"


# ---------------------------------------------------------------------------
# Invalid trees
# ---------------------------------------------------------------------------

class TestCircularDependency:

    def test_circular_dependency(self) -> None:
        """A depends on B, B depends on A."""
        nodes = [
            _node("root", children=["a", "b"], is_atomic=False),
            _node("a", depends_on=["b"]),
            _node("b", depends_on=["a"]),
        ]
        errors = validate_tree(nodes)
        assert len(errors) > 0
        assert any("circular" in e.lower() or "cycle" in e.lower() for e in errors), (
            f"Expected circular dependency error, got: {errors}"
        )


class TestMissingDependency:

    def test_missing_dependency(self) -> None:
        """Node depends on non-existent ID."""
        nodes = [
            _node("root", children=["a"], is_atomic=False),
            _node("a", depends_on=["nonexistent"]),
        ]
        errors = validate_tree(nodes)
        assert len(errors) > 0
        assert any("nonexistent" in e for e in errors), (
            f"Expected error mentioning 'nonexistent', got: {errors}"
        )


class TestNonAtomicLeaf:

    def test_non_atomic_leaf(self) -> None:
        """Leaf node with is_atomic: false."""
        nodes = [
            _node("root", children=["leaf"], is_atomic=False),
            _node("leaf", is_atomic=False, children=[]),
        ]
        errors = validate_tree(nodes)
        assert len(errors) > 0
        assert any("atomic" in e.lower() or "leaf" in e.lower() for e in errors), (
            f"Expected non-atomic leaf error, got: {errors}"
        )


class TestMultipleRoots:

    def test_multiple_roots(self) -> None:
        """Two nodes not referenced as children by any other node."""
        nodes = [
            _node("root1", children=["child1"], is_atomic=False),
            _node("child1"),
            _node("root2"),
        ]
        errors = validate_tree(nodes)
        assert len(errors) > 0
        assert any("root" in e.lower() for e in errors), (
            f"Expected multiple roots error, got: {errors}"
        )


class TestOrphanNode:

    def test_orphan_node(self) -> None:
        """Node not referenced as child and not root -- effectively an extra root / orphan."""
        nodes = [
            _node("root", children=["a"], is_atomic=False),
            _node("a"),
            _node("orphan"),
        ]
        errors = validate_tree(nodes)
        assert len(errors) > 0
        assert any("root" in e.lower() or "orphan" in e.lower() for e in errors), (
            f"Expected orphan/multiple-root error, got: {errors}"
        )


class TestUnsatisfiedDataContract:

    def test_unsatisfied_data_contract(self) -> None:
        """Node consumes data_x but no dependency provides it."""
        nodes = [
            _node("root", children=["a", "b"], is_atomic=False),
            _node("a", provides=["data_y"]),
            _node("b", depends_on=["a"], consumes=["data_x"]),
        ]
        errors = validate_tree(nodes)
        assert len(errors) > 0
        assert any("data_x" in e for e in errors), (
            f"Expected error mentioning 'data_x', got: {errors}"
        )
