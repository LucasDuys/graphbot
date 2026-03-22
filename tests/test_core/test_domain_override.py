"""Tests for domain override and tool-aware execution (T072/T073/T074)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.decomposer import Decomposer, infer_domain_from_description
from core_gb.executor import SimpleExecutor
from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# T072: infer_domain_from_description
# ---------------------------------------------------------------------------


class TestInferDomainFromDescription:

    def test_infer_file_domain(self) -> None:
        assert infer_domain_from_description("Read README.md") == Domain.FILE

    def test_infer_file_domain_list(self) -> None:
        assert infer_domain_from_description("List files in the project") == Domain.FILE

    def test_infer_file_domain_extension(self) -> None:
        assert infer_domain_from_description("Open config.json") == Domain.FILE

    def test_infer_web_domain(self) -> None:
        assert infer_domain_from_description("Search the web for Python tutorials") == Domain.WEB

    def test_infer_web_domain_url(self) -> None:
        assert infer_domain_from_description("Fetch https://example.com") == Domain.WEB

    def test_infer_code_domain(self) -> None:
        assert infer_domain_from_description("Run git log --oneline") == Domain.CODE

    def test_infer_code_domain_pytest(self) -> None:
        assert infer_domain_from_description("Execute pytest tests/") == Domain.CODE

    def test_infer_no_override(self) -> None:
        assert infer_domain_from_description("What is 2+2?") is None

    def test_infer_no_override_reasoning(self) -> None:
        assert infer_domain_from_description("Explain quantum computing") is None


# ---------------------------------------------------------------------------
# T072: Domain override applied during decomposition
# ---------------------------------------------------------------------------


class TestDomainOverrideInDecompose:

    async def test_override_applied_in_decompose(self) -> None:
        """Mock decomposer returns node with domain=SYSTEM but description='Read file.py',
        verify domain becomes FILE after post-processing."""
        mock_router = MagicMock()

        # Build a valid decomposition JSON where an atomic node has wrong domain
        tree_json = json.dumps({
            "nodes": [
                {
                    "id": "root",
                    "description": "Read file.py and summarize",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 2,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["read", "sum"],
                },
                {
                    "id": "read",
                    "description": "Read file.py",
                    "domain": "system",  # WRONG -- should be file
                    "task_type": "RETRIEVE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["content"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "sum",
                    "description": "Summarize the content",
                    "domain": "synthesis",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": ["read"],
                    "provides": ["summary"],
                    "consumes": ["content"],
                    "is_atomic": True,
                    "children": [],
                },
            ]
        })

        mock_completion = CompletionResult(
            content=tree_json,
            model="mock",
            tokens_in=10,
            tokens_out=20,
            latency_ms=50.0,
            cost=0.001,
        )
        mock_router.route = AsyncMock(return_value=mock_completion)

        decomposer = Decomposer(mock_router)
        nodes = await decomposer.decompose("Read file.py and summarize")

        # Find the atomic node that reads file.py
        read_nodes = [n for n in nodes if "file.py" in n.description.lower() and n.is_atomic]
        assert len(read_nodes) == 1
        assert read_nodes[0].domain == Domain.FILE, (
            f"Expected FILE but got {read_nodes[0].domain}"
        )

        # The synthesis node should remain unchanged (no file keywords in "Summarize the content")
        synth_nodes = [n for n in nodes if "summarize" in n.description.lower() and n.is_atomic]
        assert len(synth_nodes) == 1
        assert synth_nodes[0].domain == Domain.SYNTHESIS


# ---------------------------------------------------------------------------
# T074: SimpleExecutor uses tool when tool_registry is provided
# ---------------------------------------------------------------------------


class TestSimpleExecutorToolAwareness:

    async def test_simple_executor_uses_tool(self) -> None:
        """SimpleExecutor with tool registry, task 'list files in .' -> uses tool (0 tokens)."""
        from graph.store import GraphStore

        store = GraphStore(db_path=None)
        store.initialize()

        mock_router = MagicMock()

        # Create a mock tool registry
        mock_tool_registry = MagicMock()
        mock_tool_registry.has_tool.return_value = True
        mock_tool_registry.execute = AsyncMock(return_value=ExecutionResult(
            root_id="test",
            output="file1.py\nfile2.py",
            success=True,
            total_nodes=1,
            total_tokens=0,
            total_latency_ms=5.0,
            total_cost=0.0,
            model_used="tool:file",
            errors=(),
        ))

        executor = SimpleExecutor(store, mock_router, tool_registry=mock_tool_registry)
        result = await executor.execute("list files in .")

        assert result.success is True
        assert result.total_tokens == 0
        assert "file1.py" in result.output
        assert result.model_used == "tool:file"
        # LLM router should NOT have been called
        mock_router.route.assert_not_called()

    async def test_simple_executor_no_tool_for_reasoning(self) -> None:
        """SimpleExecutor should fall through to LLM for reasoning tasks."""
        from graph.store import GraphStore
        from models.router import ModelRouter

        store = GraphStore(db_path=None)
        store.initialize()

        mock_completion = CompletionResult(
            content="4",
            model="mock-model",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50.0,
            cost=0.001,
        )
        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.complete = AsyncMock(return_value=mock_completion)

        router = ModelRouter(provider=mock_provider)

        mock_tool_registry = MagicMock()
        mock_tool_registry.has_tool.return_value = False

        executor = SimpleExecutor(store, router, tool_registry=mock_tool_registry)
        result = await executor.execute("What is 2+2?")

        # Should have used the LLM, not a tool
        assert result.success is True
        assert result.total_tokens > 0
