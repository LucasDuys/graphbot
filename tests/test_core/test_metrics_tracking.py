"""Tests for tool vs LLM usage tracking in ExecutionResult."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.types import ExecutionResult, TaskNode, TaskStatus, Domain


class TestExecutionResultFields:
    """Verify tools_used and llm_calls fields exist with correct defaults."""

    def test_execution_result_has_tool_fields(self) -> None:
        result = ExecutionResult(root_id="r1", output="ok", success=True)
        assert result.tools_used == 0
        assert result.llm_calls == 0

    def test_execution_result_accepts_tool_counts(self) -> None:
        result = ExecutionResult(
            root_id="r1",
            output="ok",
            success=True,
            tools_used=3,
            llm_calls=2,
        )
        assert result.tools_used == 3
        assert result.llm_calls == 2


class TestDAGCountsToolNodes:
    """Verify _aggregate_results counts tool vs LLM nodes correctly."""

    async def test_dag_counts_tool_nodes(self) -> None:
        from core_gb.dag_executor import DAGExecutor

        mock_executor = MagicMock()
        dag = DAGExecutor(mock_executor)

        # 2 tool nodes + 1 LLM node
        leaves = [
            TaskNode(id="n1", description="tool1", is_atomic=True, domain=Domain.FILE),
            TaskNode(id="n2", description="tool2", is_atomic=True, domain=Domain.WEB),
            TaskNode(id="n3", description="llm1", is_atomic=True, domain=Domain.SYNTHESIS),
        ]

        results = {
            "n1": ExecutionResult(
                root_id="n1", output="file read", success=True,
                total_tokens=0, model_used="tool:file_read",
            ),
            "n2": ExecutionResult(
                root_id="n2", output="web fetch", success=True,
                total_tokens=0, model_used="tool:web_fetch",
            ),
            "n3": ExecutionResult(
                root_id="n3", output="synthesis", success=True,
                total_tokens=150, model_used="gpt-4o-mini",
            ),
        }

        import time
        node_by_id = {n.id: n for n in leaves}
        start = time.perf_counter()

        aggregated = await dag._aggregate_results("root", leaves, results, node_by_id, start)

        assert aggregated.tools_used == 2
        assert aggregated.llm_calls == 1

    async def test_dag_counts_zero_when_no_results(self) -> None:
        from core_gb.dag_executor import DAGExecutor

        mock_executor = MagicMock()
        dag = DAGExecutor(mock_executor)

        import time
        start = time.perf_counter()

        aggregated = await dag._aggregate_results("root", [], {}, {}, start)

        assert aggregated.tools_used == 0
        assert aggregated.llm_calls == 0


class TestSimpleTaskMetrics:
    """Verify single LLM call path sets tools_used=0, llm_calls=1."""

    @pytest.mark.asyncio
    async def test_simple_task_metrics(self) -> None:
        from core_gb.executor import SimpleExecutor
        from core_gb.types import CompletionResult

        mock_store = MagicMock()
        mock_store.get_context.return_value = MagicMock(
            format=MagicMock(return_value=""),
            total_tokens=0,
        )

        mock_completion = CompletionResult(
            content="Hello world",
            model="gpt-4o-mini",
            tokens_in=10,
            tokens_out=20,
            latency_ms=100.0,
            cost=0.001,
        )

        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_completion)

        executor = SimpleExecutor(mock_store, mock_router, tool_registry=None)
        result = await executor.execute("Say hello", complexity=1)

        assert result.success is True
        assert result.tools_used == 0
        assert result.llm_calls == 1
        assert result.total_tokens == 30

    @pytest.mark.asyncio
    async def test_tool_task_metrics(self) -> None:
        from core_gb.executor import SimpleExecutor

        mock_store = MagicMock()
        mock_router = MagicMock()

        tool_result = ExecutionResult(
            root_id="t1",
            output="file contents",
            success=True,
            model_used="tool:file_read",
        )

        mock_registry = MagicMock()
        mock_registry.has_tool.return_value = True
        mock_registry.execute = AsyncMock(return_value=tool_result)

        # Patch infer_domain_from_description to return a domain with a tool
        import core_gb.executor as executor_mod
        original_fn = None
        try:
            from core_gb.decomposer import infer_domain_from_description
            original_fn = infer_domain_from_description
        except ImportError:
            pass

        executor = SimpleExecutor(mock_store, mock_router, tool_registry=mock_registry)

        # Directly test: when tool_registry returns a result, tools_used=1
        # We need infer_domain to return something
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "core_gb.executor.SimpleExecutor._extract_mentions",
                staticmethod(lambda text: []),
            )
            # Patch the domain inference to return FILE
            mp.setattr(
                "core_gb.decomposer.infer_domain_from_description",
                lambda text: Domain.FILE,
            )
            result = await executor.execute("Read file.txt", complexity=1)

        assert result.success is True
        assert result.tools_used == 1
        assert result.llm_calls == 0
