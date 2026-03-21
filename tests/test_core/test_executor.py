"""Tests for the SimpleExecutor -- minimal single-task execution."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core_gb.types import CompletionResult, ExecutionResult
from models.base import ModelProvider
from models.errors import ProviderError
from models.router import ModelRouter
from graph.store import GraphStore


class MockProvider(ModelProvider):
    """Mock provider that returns a configurable CompletionResult."""

    def __init__(self, result: CompletionResult | None = None) -> None:
        self._result = result or CompletionResult(
            content="9386",
            model="mock-model",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50.0,
            cost=0.001,
        )
        self._mock_complete = AsyncMock(side_effect=self._do_complete)
        self.captured_messages: list[list[dict]] = []

    @property
    def name(self) -> str:
        return "mock"

    async def _do_complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        self.captured_messages.append(messages)
        return self._result

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


class TestExecuteSimpleTask:
    async def test_execute_simple_task(self) -> None:
        """Execute a simple math question with empty graph, verify success."""
        store = _make_store()
        provider = MockProvider()
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute("What is 247 * 38?")

        assert result.success is True
        assert result.output == "9386"
        assert result.total_nodes == 1
        assert isinstance(result, ExecutionResult)

        store.close()


class TestExecuteWithGraphContext:
    async def test_execute_with_graph_context(self) -> None:
        """Seed graph with a User entity, verify context injection."""
        store = _make_store()
        provider = MockProvider()
        router = ModelRouter(provider=provider)

        # Seed the graph with a User entity
        store.create_node("User", {
            "id": "user_1",
            "name": "Lucas",
            "role": "student",
            "institution": "TU/e",
            "interests": "AI",
        })

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute("Tell me about Lucas")

        assert result.success is True
        # Verify context was injected: system message should contain <context>
        assert len(provider.captured_messages) == 1
        messages = provider.captured_messages[0]
        system_msg = messages[0]["content"]
        assert "<context>" in system_msg
        assert "Lucas" in system_msg
        assert result.context_tokens > 0

        store.close()


class TestExecuteEmptyGraph:
    async def test_execute_empty_graph(self) -> None:
        """Execute with no graph data -- should still work without context."""
        store = _make_store()
        provider = MockProvider()
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute("Hello world")

        assert result.success is True
        assert result.output == "9386"
        # System message should NOT contain <context> tag
        messages = provider.captured_messages[0]
        system_msg = messages[0]["content"]
        assert "<context>" not in system_msg
        assert system_msg == "You are a helpful assistant."

        store.close()


class TestExecuteProviderError:
    async def test_execute_provider_error(self) -> None:
        """Provider raises ProviderError -- result.success should be False."""
        store = _make_store()
        provider = MockProvider()
        provider._mock_complete.side_effect = ProviderError(
            "connection failed", provider="mock", model="mock-model"
        )
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute("What is 1+1?")

        assert result.success is False
        assert len(result.errors) > 0
        assert "connection failed" in result.errors[0]
        assert result.output == ""

        store.close()


class TestExecuteReturnsMetrics:
    async def test_execute_returns_metrics(self) -> None:
        """Verify total_tokens, latency_ms, model_used are populated."""
        store = _make_store()
        mock_result = CompletionResult(
            content="answer",
            model="test-model-v1",
            tokens_in=25,
            tokens_out=15,
            latency_ms=120.0,
            cost=0.005,
        )
        provider = MockProvider(result=mock_result)
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute("Tell me something", complexity=2)

        assert result.success is True
        assert result.total_tokens == 25 + 15
        assert result.total_latency_ms > 0
        assert result.model_used == "test-model-v1"
        assert result.total_cost == 0.005
        assert result.root_id  # should be a non-empty UUID string

        store.close()
