"""Tests for tool failure recovery with retry + LLM fallback in DAGExecutor."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import Domain, ExecutionResult, TaskNode, TaskStatus


def _make_leaf(
    node_id: str,
    description: str,
    domain: Domain = Domain.FILE,
) -> TaskNode:
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=domain,
        complexity=1,
        status=TaskStatus.READY,
    )


class _MockToolRegistry:
    """Tool registry that fails a configurable number of times then succeeds."""

    def __init__(self, fail_count: int = 0) -> None:
        self._fail_count = fail_count
        self._call_count = 0

    def has_tool(self, domain: Domain) -> bool:
        return domain == Domain.FILE

    async def execute(self, node: TaskNode) -> ExecutionResult:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return ExecutionResult(
                root_id=node.id,
                output="",
                success=False,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=0.0,
                total_cost=0.0,
                errors=(f"Tool error attempt {self._call_count}",),
            )
        return ExecutionResult(
            root_id=node.id,
            output="Tool success",
            success=True,
            total_nodes=1,
            total_tokens=0,
            total_latency_ms=0.0,
            total_cost=0.0,
        )

    @property
    def call_count(self) -> int:
        return self._call_count


class _MockLLMExecutor:
    """Mock LLM executor that tracks calls."""

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self, task: str, complexity: int = 1, provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        self.call_count += 1
        return ExecutionResult(
            root_id=str(uuid.uuid4()),
            output=f"LLM fallback: {task}",
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=5.0,
            total_cost=0.001,
        )


class TestToolRetry:
    async def test_retry_succeeds_second_attempt(self) -> None:
        """Tool fails once, succeeds on retry -- no LLM fallback."""
        tool_reg = _MockToolRegistry(fail_count=1)
        llm = _MockLLMExecutor()
        dag = DAGExecutor(executor=llm, max_concurrency=10, tool_registry=tool_reg)

        nodes = [_make_leaf("a", "Read file.txt")]
        result = await dag.execute(nodes)

        assert result.success is True
        assert "Tool success" in result.output
        assert tool_reg.call_count == 2  # first fail + retry success
        assert llm.call_count == 0  # no LLM fallback needed

    async def test_retry_falls_back_to_llm(self) -> None:
        """Tool fails twice -- falls back to LLM executor."""
        tool_reg = _MockToolRegistry(fail_count=5)  # always fails
        llm = _MockLLMExecutor()
        dag = DAGExecutor(executor=llm, max_concurrency=10, tool_registry=tool_reg)

        nodes = [_make_leaf("a", "Read file.txt")]
        result = await dag.execute(nodes)

        assert result.success is True
        assert "LLM fallback" in result.output
        assert tool_reg.call_count == 2  # two tool attempts
        assert llm.call_count == 1  # then LLM

    async def test_no_retry_on_success(self) -> None:
        """Tool succeeds first try -- no retry, no LLM."""
        tool_reg = _MockToolRegistry(fail_count=0)
        llm = _MockLLMExecutor()
        dag = DAGExecutor(executor=llm, max_concurrency=10, tool_registry=tool_reg)

        nodes = [_make_leaf("a", "Read file.txt")]
        result = await dag.execute(nodes)

        assert result.success is True
        assert "Tool success" in result.output
        assert tool_reg.call_count == 1
        assert llm.call_count == 0
