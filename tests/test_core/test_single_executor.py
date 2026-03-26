"""Tests for SingleCallExecutor -- one enriched LLM call with graph context."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core_gb.single_executor import SingleCallExecutor, DEFAULT_HISTORY_LIMIT
from core_gb.types import (
    CompletionResult,
    ExecutionResult,
    GraphContext,
    Pattern,
)
from models.base import ModelProvider
from models.errors import AllProvidersExhaustedError, ProviderError
from models.router import ModelRouter


class MockProvider(ModelProvider):
    """Mock provider that returns a configurable CompletionResult."""

    def __init__(self, result: CompletionResult | None = None) -> None:
        self._result = result or CompletionResult(
            content="mock response",
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


class FailingProvider(ModelProvider):
    """Mock provider that raises ProviderError."""

    @property
    def name(self) -> str:
        return "failing"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        raise ProviderError("provider down", provider="failing", model=model)


def _empty_context() -> GraphContext:
    return GraphContext()


def _rich_context() -> GraphContext:
    return GraphContext(
        user_summary="Lucas, CS student",
        relevant_entities=(
            {"type": "Person", "name": "Lucas", "details": "studies CS at TU/e"},
        ),
        active_memories=("prefers Python",),
        total_tokens=42,
    )


class TestSingleCallExecutorBasic:
    """Basic execution: task + empty context -> ExecutionResult."""

    async def test_returns_execution_result(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("What is 2+2?", _empty_context())

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output == "mock response"
        assert result.total_nodes == 1
        assert result.llm_calls == 1
        assert result.tools_used == 0
        assert result.total_tokens == 15  # 10 in + 5 out
        assert result.total_cost == 0.001
        assert result.model_used == "mock-model"
        assert len(result.nodes) == 1
        assert result.errors == ()

    async def test_root_id_is_uuid(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("hello", _empty_context())

        # root_id should be a valid UUID string (36 chars with dashes)
        assert len(result.root_id) == 36
        assert result.root_id.count("-") == 4

    async def test_latency_is_positive(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("test", _empty_context())

        assert result.total_latency_ms > 0


class TestPromptAssembly:
    """Verify the prompt structure: system, context, history, user."""

    async def test_system_message_present(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        await executor.execute("What is AI?", _empty_context(), complexity=4)

        messages = provider.captured_messages[0]
        assert messages[0]["role"] == "system"
        # XML-structured prompt at high complexity: domain role and instruction sections
        system_content = messages[0]["content"]
        assert "<instructions>" in system_content
        assert "<examples>" in system_content
        assert "<output_format>" in system_content

    async def test_user_message_is_last(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        await executor.execute("Explain GraphBot", _empty_context())

        messages = provider.captured_messages[0]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Explain GraphBot"

    async def test_context_injected_in_system(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        ctx = _rich_context()
        await executor.execute("Who is Lucas?", ctx, complexity=3)

        system_content = provider.captured_messages[0][0]["content"]
        assert "<context>" in system_content
        assert "Lucas" in system_content
        assert "CS student" in system_content
        assert "prefers Python" in system_content

    async def test_empty_context_no_context_tag(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        await executor.execute("hello", _empty_context())

        system_content = provider.captured_messages[0][0]["content"]
        assert "<context>" not in system_content

    async def test_context_tokens_tracked(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        ctx = _rich_context()
        result = await executor.execute("test", ctx)

        assert result.context_tokens == 42


class TestConversationHistory:
    """Verify conversation history injection."""

    async def test_history_injected_between_system_and_user(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        await executor.execute("Follow up", _empty_context(), conversation_history=history)

        messages = provider.captured_messages[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hello!"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Follow up"

    async def test_history_trimmed_to_limit(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router, history_limit=2)

        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "msg3"},
        ]

        await executor.execute("msg4", _empty_context(), conversation_history=history)

        messages = provider.captured_messages[0]
        # system + 2 history messages + user = 4
        assert len(messages) == 4
        # Only the last 2 history messages should be included
        assert messages[1]["content"] == "resp2"
        assert messages[2]["content"] == "msg3"

    async def test_no_history_only_system_and_user(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        await executor.execute("hello", _empty_context())

        messages = provider.captured_messages[0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestPatternHints:
    """Verify pattern hints are formatted into the prompt."""

    async def test_pattern_hints_in_system_message(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        patterns = [
            Pattern(
                id="p1",
                trigger="summarize document",
                description="Extract key points then synthesize",
                success_count=5,
            ),
        ]

        await executor.execute(
            "Summarize this paper", _empty_context(), pattern_hints=patterns,
        )

        system_content = provider.captured_messages[0][0]["content"]
        assert "Similar tasks have been answered like this:" in system_content
        assert "summarize document" in system_content
        assert "Extract key points" in system_content
        assert "5 successes" in system_content

    async def test_multiple_pattern_hints(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        patterns = [
            Pattern(id="p1", trigger="code review", description="Analyze code quality"),
            Pattern(id="p2", trigger="bug fix", description="Identify root cause"),
        ]

        await executor.execute("Review my code", _empty_context(), pattern_hints=patterns)

        system_content = provider.captured_messages[0][0]["content"]
        assert "code review" in system_content
        assert "bug fix" in system_content

    async def test_no_patterns_no_hint_section(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        await executor.execute("hello", _empty_context(), pattern_hints=[])

        system_content = provider.captured_messages[0][0]["content"]
        assert "Similar tasks" not in system_content


class TestErrorHandling:
    """Verify graceful error handling on provider failures."""

    async def test_provider_error_returns_failed_result(self) -> None:
        provider = FailingProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("will fail", _empty_context())

        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert result.output == ""
        assert len(result.errors) > 0
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.model_used == ""
        assert result.total_latency_ms > 0

    async def test_all_providers_exhausted(self) -> None:
        provider = FailingProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute("will fail", _empty_context())

        assert result.success is False
        assert len(result.errors) >= 1


class TestComplexityRouting:
    """Verify complexity is forwarded to the model router."""

    async def test_complexity_forwarded_to_task_node(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute(
            "complex task", _empty_context(), complexity=4,
        )

        assert result.success is True
        # The router received the call -- it uses complexity to pick model.
        # We verify indirectly: the call succeeded.
        provider._mock_complete.assert_called_once()


class TestExactlyOneCall:
    """Verify exactly one LLM call is made, no more."""

    async def test_single_llm_call(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        result = await executor.execute(
            "answer this",
            _rich_context(),
            conversation_history=[
                {"role": "user", "content": "prior"},
                {"role": "assistant", "content": "ok"},
            ],
            pattern_hints=[
                Pattern(id="p1", trigger="answer", description="direct answer"),
            ],
            complexity=3,
        )

        assert result.llm_calls == 1
        assert provider._mock_complete.call_count == 1


class TestFullPromptIntegration:
    """Integration test: all sections present in one prompt."""

    async def test_all_sections_assembled(self) -> None:
        provider = MockProvider()
        router = ModelRouter(provider=provider)
        executor = SingleCallExecutor(router)

        ctx = GraphContext(
            user_summary="Test user",
            relevant_entities=(
                {"type": "Project", "name": "GraphBot", "details": "DAG engine"},
            ),
            active_memories=("uses Python 3.12",),
            total_tokens=30,
        )
        history = [
            {"role": "user", "content": "What is GraphBot?"},
            {"role": "assistant", "content": "A DAG execution engine."},
        ]
        patterns = [
            Pattern(id="p1", trigger="explain project", description="overview then details"),
        ]

        result = await executor.execute(
            "Tell me more about GraphBot",
            ctx,
            conversation_history=history,
            pattern_hints=patterns,
            complexity=4,
        )

        assert result.success is True

        messages = provider.captured_messages[0]

        # System message: domain role + XML-structured context + pattern hints
        system = messages[0]
        assert system["role"] == "system"
        assert "<instructions>" in system["content"]
        assert "<context>" in system["content"]
        assert "Test user" in system["content"]
        assert "GraphBot" in system["content"]
        assert "uses Python 3.12" in system["content"]
        assert "explain project" in system["content"]

        # Conversation history
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is GraphBot?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "A DAG execution engine."

        # User message: actual task
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Tell me more about GraphBot"
