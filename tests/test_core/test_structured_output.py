"""Tests for structured JSON output when provides_keys is set."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from core_gb.types import CompletionResult, ExecutionResult
from models.base import ModelProvider
from models.router import ModelRouter
from graph.store import GraphStore


class MockProvider(ModelProvider):
    """Mock provider that returns a configurable CompletionResult."""

    def __init__(self, result: CompletionResult | None = None) -> None:
        self._result = result or CompletionResult(
            content='{"summary": "hello", "details": "world"}',
            model="mock-model",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50.0,
            cost=0.001,
        )
        self._mock_complete = AsyncMock(side_effect=self._do_complete)
        self.captured_messages: list[list[dict]] = []
        self.captured_kwargs: list[dict] = []

    @property
    def name(self) -> str:
        return "mock"

    async def _do_complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        self.captured_messages.append(messages)
        self.captured_kwargs.append(kwargs)
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


class TestProvidesKeysTriggersJsonInstruction:
    async def test_provides_keys_triggers_json_instruction(self) -> None:
        """When provides_keys is set, the system message contains JSON instruction."""
        store = _make_store()
        provider = MockProvider()
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        await executor.execute("Do something", provides_keys=["summary", "details"])

        assert len(provider.captured_messages) == 1
        system_msg = provider.captured_messages[0][0]["content"]
        assert "JSON object" in system_msg
        assert '"summary"' in system_msg
        assert '"details"' in system_msg

        store.close()


class TestStructuredOutputParsed:
    async def test_structured_output_parsed(self) -> None:
        """When provider returns valid JSON, ExecutionResult.output is valid JSON."""
        store = _make_store()
        json_content = '{"summary": "hello", "details": "world"}'
        provider = MockProvider(
            result=CompletionResult(
                content=json_content,
                model="mock-model",
                tokens_in=10,
                tokens_out=5,
                latency_ms=50.0,
                cost=0.001,
            )
        )
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute(
            "Do something", provides_keys=["summary", "details"]
        )

        assert result.success is True
        parsed = json.loads(result.output)
        assert parsed["summary"] == "hello"
        assert parsed["details"] == "world"

        store.close()


class TestStructuredOutputFallback:
    async def test_structured_output_fallback(self) -> None:
        """When provider returns plain text, it is wrapped under the first provides key."""
        store = _make_store()
        provider = MockProvider(
            result=CompletionResult(
                content="This is plain text, not JSON.",
                model="mock-model",
                tokens_in=10,
                tokens_out=5,
                latency_ms=50.0,
                cost=0.001,
            )
        )
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute(
            "Do something", provides_keys=["summary", "details"]
        )

        assert result.success is True
        parsed = json.loads(result.output)
        assert parsed["summary"] == "This is plain text, not JSON."

        store.close()


class TestNoProvidesKeysUnchanged:
    async def test_no_provides_keys_unchanged(self) -> None:
        """Without provides_keys, behavior is unchanged (plain text output)."""
        store = _make_store()
        provider = MockProvider(
            result=CompletionResult(
                content="plain answer",
                model="mock-model",
                tokens_in=10,
                tokens_out=5,
                latency_ms=50.0,
                cost=0.001,
            )
        )
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        result = await executor.execute("Do something")

        assert result.success is True
        assert result.output == "plain answer"

        # System message should NOT contain JSON instruction
        system_msg = provider.captured_messages[0][0]["content"]
        assert "JSON object" not in system_msg

        store.close()


class TestResponseFormatPassed:
    async def test_response_format_passed(self) -> None:
        """Verify response_format=json_object is passed to router when provides_keys set."""
        store = _make_store()
        provider = MockProvider()
        router = ModelRouter(provider=provider)

        from core_gb.executor import SimpleExecutor

        executor = SimpleExecutor(store=store, router=router)
        await executor.execute("Do something", provides_keys=["summary"])

        assert len(provider.captured_kwargs) == 1
        kwargs = provider.captured_kwargs[0]
        assert "response_format" in kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

        store.close()
