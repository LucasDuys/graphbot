"""Tests for core GraphBot data structures."""

from core_gb.types import (
    Domain,
    ExecutionResult,
    FlowType,
    GraphContext,
    Pattern,
    TaskNode,
    TaskStatus,
)


class TestTaskNode:
    def test_create_simple_task(self, simple_task: TaskNode) -> None:
        assert simple_task.id == "task_001"
        assert simple_task.is_atomic is True
        assert simple_task.domain == Domain.SYSTEM
        assert simple_task.complexity == 1
        assert simple_task.status == TaskStatus.READY

    def test_default_values(self) -> None:
        node = TaskNode(id="t1", description="test")
        assert node.parent_id is None
        assert node.children == []
        assert node.requires == []
        assert node.provides == []
        assert node.consumes == []
        assert node.is_atomic is False
        assert node.status == TaskStatus.CREATED
        assert node.input_data == {}
        assert node.output_data == {}
        assert node.tokens_used == 0

    def test_dependency_contracts(self) -> None:
        node = TaskNode(
            id="t1",
            description="Parse TODOs",
            requires=["read_file"],
            consumes=["file_content"],
            provides=["todo_list"],
        )
        assert "read_file" in node.requires
        assert "file_content" in node.consumes
        assert "todo_list" in node.provides


class TestParallelTree:
    def test_tree_structure(self, parallel_tree: dict[str, TaskNode]) -> None:
        root = parallel_tree["root"]
        assert len(root.children) == 3
        assert root.flow_type == FlowType.PARALLEL

    def test_leaves_are_atomic(self, parallel_tree: dict[str, TaskNode]) -> None:
        for node_id in ["leaf_ams", "leaf_lon", "leaf_ber"]:
            assert parallel_tree[node_id].is_atomic is True
            assert parallel_tree[node_id].domain == Domain.WEB

    def test_leaves_provide_data(self, parallel_tree: dict[str, TaskNode]) -> None:
        assert "weather_amsterdam" in parallel_tree["leaf_ams"].provides
        assert "weather_london" in parallel_tree["leaf_lon"].provides
        assert "weather_berlin" in parallel_tree["leaf_ber"].provides


class TestDependentTree:
    def test_sequential_flow(self, dependent_tree: dict[str, TaskNode]) -> None:
        root = dependent_tree["root"]
        assert root.flow_type == FlowType.SEQUENCE

    def test_dependency_chain(self, dependent_tree: dict[str, TaskNode]) -> None:
        parse = dependent_tree["parse"]
        assert "read" in parse.requires
        assert "file_content" in parse.consumes

        fmt = dependent_tree["format"]
        assert "parse" in fmt.requires
        assert "todo_list" in fmt.consumes

    def test_data_flow_contracts(self, dependent_tree: dict[str, TaskNode]) -> None:
        read = dependent_tree["read"]
        parse = dependent_tree["parse"]
        assert "file_content" in read.provides
        assert "file_content" in parse.consumes


class TestExecutionResult:
    def test_create_result(self) -> None:
        result = ExecutionResult(
            root_id="root",
            output="Test output",
            success=True,
            total_nodes=5,
            total_tokens=500,
        )
        assert result.success is True
        assert result.total_nodes == 5


class TestGraphContext:
    def test_format_context(self) -> None:
        ctx = GraphContext(
            user_summary="Lucas | CSE at TU/e",
            relevant_entities=(
                {"type": "PROJECT", "name": "graphbot", "details": "Python 3.11"},
            ),
            active_memories=("Extended bachelor 3.5yr+ (since 2025-09)",),
        )
        formatted = ctx.format()
        assert "Lucas" in formatted
        assert "graphbot" in formatted
        assert "Extended bachelor" in formatted

    def test_empty_context(self) -> None:
        ctx = GraphContext()
        assert ctx.format() == ""
