"""Tests for Orchestrator conversation memory integration.

Verifies that:
- process() accepts optional chat_id parameter
- Follow-up messages use previous conversation context
- Different chat_ids have isolated histories
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from core_gb.conversation import ConversationMemory
from core_gb.orchestrator import Orchestrator
from core_gb.types import CompletionResult, ExecutionResult
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _simple_completion(content: str, tokens: int = 10, cost: float = 0.001) -> CompletionResult:
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens,
        tokens_out=tokens,
        latency_ms=50.0,
        cost=cost,
    )


class CapturingMockProvider(ModelProvider):
    """Mock provider that captures messages sent to it and returns canned responses."""

    def __init__(self, response_content: str = "Mock response") -> None:
        self._response_content = response_content
        self.captured_messages: list[list[dict]] = []

    @property
    def name(self) -> str:
        return "mock"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        self.captured_messages.append(list(messages))
        return _simple_completion(self._response_content)


# ---------------------------------------------------------------------------
# Tests: chat_id parameter
# ---------------------------------------------------------------------------

class TestProcessAcceptsChatId:
    """Orchestrator.process() accepts optional chat_id parameter."""

    @pytest.mark.asyncio
    async def test_process_works_without_chat_id(self) -> None:
        """Backward compatibility: process() works without chat_id."""
        store = _make_store()
        provider = CapturingMockProvider("Hello!")
        router = ModelRouter(provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Hello")

        assert result.success
        store.close()

    @pytest.mark.asyncio
    async def test_process_works_with_chat_id(self) -> None:
        """process() accepts chat_id and returns a result."""
        store = _make_store()
        provider = CapturingMockProvider("Hello!")
        router = ModelRouter(provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Hello", chat_id="user-123")

        assert result.success
        store.close()


class TestConversationContextInjection:
    """Follow-up messages include previous conversation context."""

    @pytest.mark.asyncio
    async def test_followup_includes_previous_context(self) -> None:
        """Second message in same chat includes first message in LLM context."""
        store = _make_store()
        provider = CapturingMockProvider("The answer is 4")
        router = ModelRouter(provider)
        orch = Orchestrator(store, router)

        # First message
        await orch.process("What is 2+2?", chat_id="chat-A")

        # Second message (follow-up)
        await orch.process("And what about 3+3?", chat_id="chat-A")

        # The second LLM call should contain conversation history.
        assert len(provider.captured_messages) >= 2
        second_call_messages = provider.captured_messages[-1]
        # Look for the previous user message in the conversation context.
        all_content = " ".join(m.get("content", "") for m in second_call_messages)
        assert "2+2" in all_content, (
            "Follow-up LLM call should contain previous conversation context"
        )

    @pytest.mark.asyncio
    async def test_assistant_response_stored_in_history(self) -> None:
        """The assistant's response is also stored in conversation history."""
        store = _make_store()
        provider = CapturingMockProvider("The capital is Paris")
        router = ModelRouter(provider)
        orch = Orchestrator(store, router)

        await orch.process("What is the capital of France?", chat_id="chat-B")
        await orch.process("Tell me more about it", chat_id="chat-B")

        # The second call should contain the assistant's previous response.
        second_call_messages = provider.captured_messages[-1]
        all_content = " ".join(m.get("content", "") for m in second_call_messages)
        assert "Paris" in all_content, (
            "Follow-up should include the assistant's previous response"
        )


class TestConversationIsolation:
    """Different chat_ids maintain separate conversation histories."""

    @pytest.mark.asyncio
    async def test_different_chats_isolated(self) -> None:
        """Messages from chat-A do not appear in chat-B's context."""
        store = _make_store()
        provider = CapturingMockProvider("Response")
        router = ModelRouter(provider)
        orch = Orchestrator(store, router)

        # Chat A: talk about Python
        await orch.process("Tell me about Python", chat_id="chat-A")

        # Chat B: talk about JavaScript
        await orch.process("Tell me about JavaScript", chat_id="chat-B")

        # Chat B follow-up: should NOT contain Python context from chat-A
        await orch.process("What frameworks does it have?", chat_id="chat-B")

        third_call_messages = provider.captured_messages[-1]
        all_content = " ".join(m.get("content", "") for m in third_call_messages)
        assert "Python" not in all_content, (
            "Chat B should not contain chat A's conversation history"
        )
        assert "JavaScript" in all_content, (
            "Chat B follow-up should contain chat B's prior context"
        )
