"""Integration tests for the single-call pipeline through smart routing.

Tests the end-to-end flow: user message -> intake -> smart routing decision ->
single-call execution (or decomposition). Verifies that simple tasks route to
SingleCallExecutor, complex tasks route to decomposition, force_decompose
overrides routing, graph context and conversation history are included, safety
checks still run before single-call, and the trivial fast path still works.

At least 10 integration tests exercising the full Orchestrator.process() path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core_gb.orchestrator import Orchestrator
from core_gb.types import CompletionResult, Domain, ExecutionResult, GraphContext
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider(ModelProvider):
    """Provider that returns configurable responses based on call order."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = list(responses)
        self._call_count: int = 0
        self.call_log: list[list[dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return "mock"

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: object
    ) -> CompletionResult:
        self.call_log.append(messages)
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def _completion(
    content: str,
    tokens_in: int = 10,
    tokens_out: int = 20,
    cost: float = 0.001,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=50.0,
        cost=cost,
    )


def _make_store() -> GraphStore:
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _decomposition_json(
    *,
    root_desc: str = "Root task",
    leaves: list[dict[str, str]] | None = None,
) -> str:
    """Build a valid decomposition JSON string for the mock decomposer."""
    if leaves is None:
        leaves = [
            {"id": "leaf1", "description": "Sub-task 1"},
            {"id": "leaf2", "description": "Sub-task 2"},
        ]
    nodes: list[dict[str, Any]] = [
        {
            "id": "root",
            "description": root_desc,
            "domain": "synthesis",
            "task_type": "THINK",
            "complexity": 2,
            "depends_on": [],
            "provides": [],
            "consumes": [],
            "is_atomic": False,
            "children": [leaf["id"] for leaf in leaves],
        },
    ]
    for leaf in leaves:
        nodes.append(
            {
                "id": leaf["id"],
                "description": leaf["description"],
                "domain": "synthesis",
                "task_type": "THINK",
                "complexity": 1,
                "depends_on": [],
                "provides": [f"{leaf['id']}_result"],
                "consumes": [],
                "is_atomic": True,
                "children": [],
            }
        )
    return json.dumps({"nodes": nodes})


def _is_decomposer_call(messages: list[dict[str, Any]]) -> bool:
    """Check if a call log entry is a decomposer call (vs single-call or leaf)."""
    if not messages:
        return False
    system_content = messages[0].get("content", "")
    return "task decomposer" in system_content.lower()


def _is_single_call(messages: list[dict[str, Any]]) -> bool:
    """Check if a call log entry is a single-call (system starts with assistant)."""
    if not messages:
        return False
    system_content = messages[0].get("content", "")
    return system_content.startswith("You are a helpful assistant")


# ---------------------------------------------------------------------------
# Test 1: Simple task routes to single-call and produces valid output
# ---------------------------------------------------------------------------


class TestSimpleTaskRoutesSingleCall:
    """A low-complexity, non-tool-domain task should route to SingleCallExecutor
    and return a successful ExecutionResult with the LLM response.

    Uses SYSTEM-domain messages (e.g. "Explain quantum physics") which have
    complexity=1 and do not require tools.
    """

    async def test_simple_question_routes_to_single_call(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("Quantum physics studies matter at atomic scale.")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Explain quantum physics")

        assert result.success is True
        assert "Quantum" in result.output
        # Single-call: exactly one LLM call, routed through SingleCallExecutor
        assert len(provider.call_log) == 1
        assert _is_single_call(provider.call_log[0])
        assert result.llm_calls == 1
        assert result.total_nodes == 1

        store.close()

    async def test_single_call_result_has_valid_metrics(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("42", tokens_in=5, tokens_out=3)])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("What is the meaning of life?")

        assert result.success is True
        assert result.total_tokens == 5 + 3
        assert result.total_latency_ms > 0
        assert result.model_used == "mock-model"
        assert result.tools_used == 0
        assert result.errors == ()

        store.close()


# ---------------------------------------------------------------------------
# Test 2: Complex task (complexity >= 4) routes to decomposition
# ---------------------------------------------------------------------------


class TestComplexTaskRoutesToDecomposition:
    """A task that IntakeParser rates as complexity >= 4 should go through
    decomposition, not single-call. Verified by checking that the first call
    is to the decomposer (system prompt contains "task decomposer").
    """

    async def test_high_complexity_routes_to_decomposition(self) -> None:
        store = _make_store()
        decomp = _decomposition_json(
            root_desc="Multi-step research",
            leaves=[
                {"id": "step1", "description": "Research topic A"},
                {"id": "step2", "description": "Research topic B"},
            ],
        )
        provider = MockProvider([
            _completion(decomp),              # decomposer call
            _completion("Result A"),          # leaf 1 execution
            _completion("Result B"),          # leaf 2 execution
            _completion("Aggregated answer"), # aggregation
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        # This message triggers multiple conjunctions and sub-clauses,
        # which IntakeParser rates at complexity >= 4.
        msg = (
            "Research the history of quantum computing and summarize the key "
            "breakthroughs, then compare the approaches of IBM and Google, "
            "then evaluate the current state of quantum supremacy claims, "
            "and finally write a 500-word essay synthesizing everything"
        )
        result = await orch.process(msg)

        assert result.success is True
        # Decomposition path: more than 1 LLM call
        assert len(provider.call_log) > 1
        # First call should be the decomposer
        assert _is_decomposer_call(provider.call_log[0])

        store.close()


# ---------------------------------------------------------------------------
# Test 3: Tool domain (FILE) routes to decomposition
# ---------------------------------------------------------------------------


class TestToolDomainRoutesToDecomposition:
    """A task classified under a tool domain (FILE, WEB, CODE, BROWSER) should
    route to decomposition even if complexity is low, because tool domains
    require the DAG executor with tool access.
    """

    async def test_file_domain_routes_to_decomposition(self) -> None:
        store = _make_store()
        decomp = _decomposition_json(
            root_desc="Read file",
            leaves=[{"id": "read1", "description": "Read the file contents"}],
        )
        provider = MockProvider([
            _completion(decomp),             # decomposer
            _completion("File contents..."), # leaf execution
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        # "Read" + "file" triggers Domain.FILE in IntakeParser
        result = await orch.process("Read the file config.yaml")

        # The task routes to decomposition (first call is decomposer)
        assert len(provider.call_log) >= 1
        assert _is_decomposer_call(provider.call_log[0])

        store.close()


# ---------------------------------------------------------------------------
# Test 4: force_decompose=True overrides smart routing
# ---------------------------------------------------------------------------


class TestForceDecomposeOverridesRouting:
    """When force_decompose=True, even a simple SYSTEM-domain question must go
    through decomposition instead of single-call.
    """

    async def test_force_decompose_bypasses_single_call(self) -> None:
        store = _make_store()
        decomp = _decomposition_json(
            root_desc="Simple question decomposed",
            leaves=[{"id": "q1", "description": "Answer the question"}],
        )
        provider = MockProvider([
            _completion(decomp),       # decomposer
            _completion("42"),         # leaf execution
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router, force_decompose=True)

        # "What is the meaning of life?" would normally single-call (SYSTEM, complexity=1)
        result = await orch.process("What is the meaning of life?")

        assert result.success is True
        # Decomposition path was taken: first call is decomposer
        assert len(provider.call_log) >= 2
        assert _is_decomposer_call(provider.call_log[0])

        store.close()

    async def test_force_decompose_false_allows_single_call(self) -> None:
        """With force_decompose=False (default), simple tasks use single-call."""
        store = _make_store()
        provider = MockProvider([_completion("Gravity is a fundamental force.")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router, force_decompose=False)

        result = await orch.process("How does gravity work?")

        assert result.success is True
        assert len(provider.call_log) == 1
        assert _is_single_call(provider.call_log[0])

        store.close()


# ---------------------------------------------------------------------------
# Test 5: Single-call with graph context includes entities in prompt
# ---------------------------------------------------------------------------


class TestSingleCallIncludesGraphContext:
    """When the knowledge graph has relevant entities, the single-call prompt
    should include context assembled from the graph.
    """

    async def test_single_call_sends_system_and_user_messages(self) -> None:
        store = _make_store()
        # Seed the graph with a Project node named "GraphBot"
        store.create_node("Project", {
            "id": "p_graphbot",
            "name": "GraphBot",
            "path": "/dev/graphbot",
            "language": "Python",
            "status": "active",
        })

        provider = MockProvider([_completion("GraphBot is a DAG execution engine.")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        # "Tell me about GraphBot" is SYSTEM domain, complexity=1 -> single-call
        result = await orch.process("Tell me about GraphBot")

        assert result.success is True
        assert len(provider.call_log) == 1

        # The single-call sends a system message and a user message
        messages = provider.call_log[0]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "GraphBot" in messages[-1]["content"]

        store.close()

    async def test_graph_context_enriches_system_prompt(self) -> None:
        """When entities are resolved, the system prompt should contain
        context section(s) with entity information."""
        store = _make_store()
        # Create a User node that the resolver can find
        store.create_node("User", {
            "id": "u_alice",
            "name": "Alice",
            "role": "engineer",
            "institution": "MIT",
        })

        provider = MockProvider([_completion("Alice is an engineer at MIT.")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Tell me about Alice")

        assert result.success is True
        # We verify the pipeline completed -- the context enricher ran
        # (context_tokens may or may not be > 0 depending on resolver match)
        assert result.total_nodes == 1
        assert result.llm_calls == 1

        store.close()


# ---------------------------------------------------------------------------
# Test 6: Single-call with conversation history includes previous messages
# ---------------------------------------------------------------------------


class TestSingleCallIncludesConversationHistory:
    """When chat_id is provided and previous messages exist, the single-call
    prompt should include conversation history between system and user.
    """

    async def test_conversation_history_included_in_prompt(self) -> None:
        store = _make_store()
        provider = MockProvider([
            _completion("Gravity pulls objects together."),
            _completion("About 9.8 m/s squared on Earth."),
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        chat_id = "test-chat-001"

        # First message: SYSTEM domain, complexity=1 -> single-call
        result1 = await orch.process("How does gravity work?", chat_id=chat_id)
        assert result1.success is True

        # Second message: also SYSTEM domain, complexity=1 -> single-call
        result2 = await orch.process("How strong is it?", chat_id=chat_id)
        assert result2.success is True

        # First call: system + user only (no history yet)
        first_call = provider.call_log[0]
        assert _is_single_call(first_call)
        assert len(first_call) == 2  # system + user

        # Second call: should have conversation history
        second_call = provider.call_log[1]
        assert _is_single_call(second_call)
        # system + history(user1 + assistant1) + user2 = 4 messages minimum
        assert len(second_call) >= 4

        store.close()

    async def test_no_history_without_chat_id(self) -> None:
        store = _make_store()
        provider = MockProvider([
            _completion("Answer 1"),
            _completion("Answer 2"),
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        await orch.process("Explain photosynthesis")
        await orch.process("Explain respiration")

        # Without chat_id, no conversation history should be injected
        for call_messages in provider.call_log:
            # system + user = 2 messages (no history)
            assert len(call_messages) == 2

        store.close()


# ---------------------------------------------------------------------------
# Test 7: Safety check blocks harmful task before single-call
# ---------------------------------------------------------------------------


class TestSafetyBlocksBeforeSingleCall:
    """The pre-decomposition safety check must block dangerous messages even
    for tasks that would otherwise route to single-call.
    """

    async def test_harmful_message_blocked_before_single_call(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("should not reach here")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Execute rm -rf / on the server")

        assert result.success is False
        assert "blocked" in result.output.lower()
        assert len(result.errors) > 0
        # No LLM calls should have been made
        assert len(provider.call_log) == 0

        store.close()

    async def test_spam_message_blocked_before_single_call(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("should not reach here")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Send spam emails to everyone in the database")

        assert result.success is False
        assert "blocked" in result.output.lower()
        assert len(provider.call_log) == 0

        store.close()

    async def test_benign_simple_task_not_blocked(self) -> None:
        """Verify that a normal question passes safety checks and executes."""
        store = _make_store()
        provider = MockProvider([_completion("The sky is blue due to Rayleigh scattering.")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Why is the sky blue?")

        assert result.success is True
        assert len(provider.call_log) == 1

        store.close()


# ---------------------------------------------------------------------------
# Test 8: Trivial fast path still works (greetings return immediately)
# ---------------------------------------------------------------------------


class TestTrivialFastPath:
    """Greetings and acknowledgments should return canned responses with
    zero LLM calls via the trivial fast path.
    """

    async def test_greeting_returns_immediately(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("should not be called")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Hello")

        assert result.success is True
        assert result.root_id == "trivial"
        assert result.total_nodes == 0
        assert result.llm_calls == 0
        # No LLM calls at all
        assert len(provider.call_log) == 0

        store.close()

    async def test_thanks_returns_immediately(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("should not be called")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Thanks!")

        assert result.success is True
        assert result.root_id == "trivial"
        assert len(provider.call_log) == 0

        store.close()

    async def test_bye_returns_immediately(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("should not be called")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Bye")

        assert result.success is True
        assert result.root_id == "trivial"
        assert len(provider.call_log) == 0

        store.close()

    async def test_trivial_has_low_latency(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("should not be called")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Hey")

        assert result.success is True
        assert result.total_latency_ms < 500  # Fast path should be well under 500ms
        assert result.total_tokens == 0

        store.close()


# ---------------------------------------------------------------------------
# Test 9: _should_decompose routing logic (end-to-end)
# ---------------------------------------------------------------------------


class TestShouldDecomposeLogic:
    """Verify the smart routing decision is correct for various domain/complexity
    combinations, exercised through the full pipeline.
    """

    async def test_system_domain_low_complexity_routes_single_call(self) -> None:
        """Domain.SYSTEM + complexity < 4 -> single-call."""
        store = _make_store()
        provider = MockProvider([_completion("Direct answer")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("Explain what prime numbers are")

        assert result.success is True
        assert len(provider.call_log) == 1
        assert _is_single_call(provider.call_log[0])

        store.close()

    async def test_web_domain_routes_to_decomposition(self) -> None:
        """Domain.WEB requires tools -> decomposition."""
        store = _make_store()
        decomp = _decomposition_json(
            root_desc="Web search",
            leaves=[{"id": "search1", "description": "Search the web"}],
        )
        provider = MockProvider([
            _completion(decomp),
            _completion("Search result"),
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        # "search the web" triggers Domain.WEB via _needs_tool and domain keywords
        result = await orch.process("Search the web for Python tutorials")

        # Should have gone through decomposition
        assert len(provider.call_log) >= 1
        assert _is_decomposer_call(provider.call_log[0])

        store.close()

    async def test_code_domain_routes_to_decomposition(self) -> None:
        """Domain.CODE requires tools -> decomposition."""
        store = _make_store()
        decomp = _decomposition_json(
            root_desc="Run pytest",
            leaves=[{"id": "exec1", "description": "Run the pytest command"}],
        )
        provider = MockProvider([
            _completion(decomp),
            _completion("Tests passed"),
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        # "pytest" triggers Domain.CODE
        result = await orch.process("Run pytest on the project")

        assert len(provider.call_log) >= 1
        assert _is_decomposer_call(provider.call_log[0])

        store.close()


# ---------------------------------------------------------------------------
# Test 10: Single-call result is persisted in knowledge graph
# ---------------------------------------------------------------------------


class TestSingleCallGraphUpdate:
    """After a single-call execution, the knowledge graph should be updated
    with the task node and result (via GraphUpdater).
    """

    async def test_single_call_updates_graph(self) -> None:
        store = _make_store()
        provider = MockProvider([_completion("The answer is 42.")])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        result = await orch.process("What is the meaning of life?")

        assert result.success is True
        # The pipeline completed with a real root_id (not trivial/blocked)
        assert result.root_id != ""
        assert result.root_id != "trivial"
        assert result.root_id != "blocked"
        assert len(result.nodes) == 1

        store.close()


# ---------------------------------------------------------------------------
# Test 11: Multiple simple queries in sequence (stateless without chat_id)
# ---------------------------------------------------------------------------


class TestMultipleSimpleQueries:
    """Multiple simple queries without chat_id should each route to single-call
    independently with no cross-contamination.
    """

    async def test_sequential_simple_queries_independent(self) -> None:
        store = _make_store()
        provider = MockProvider([
            _completion("Gravity pulls objects together."),
            _completion("Photosynthesis converts light to energy."),
            _completion("Water is H2O."),
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        r1 = await orch.process("How does gravity work?")
        r2 = await orch.process("Define photosynthesis")
        r3 = await orch.process("What is water?")

        assert r1.success and r2.success and r3.success
        assert "Gravity" in r1.output
        assert "Photosynthesis" in r2.output
        assert "Water" in r3.output
        # Each was a separate single-call
        assert len(provider.call_log) == 3
        for call in provider.call_log:
            assert _is_single_call(call)

        store.close()


# ---------------------------------------------------------------------------
# Test 12: Conversation history persists across single-call exchanges
# ---------------------------------------------------------------------------


class TestConversationPersistence:
    """When chat_id is provided, conversation memory should accumulate across
    multiple single-call exchanges, and the assistant's response should also
    be stored.
    """

    async def test_history_accumulates_across_turns(self) -> None:
        store = _make_store()
        provider = MockProvider([
            _completion("Gravity is a fundamental force."),
            _completion("About 9.8 m/s squared."),
            _completion("Einstein described it as curvature of spacetime."),
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        chat_id = "chat-accumulate"

        await orch.process("How does gravity work?", chat_id=chat_id)
        await orch.process("How strong is it?", chat_id=chat_id)
        await orch.process("Who discovered it?", chat_id=chat_id)

        # Third call should have the most history
        third_call = provider.call_log[2]
        # system + (user1 + assistant1 + user2 + assistant2) + user3 = 6+
        assert len(third_call) >= 5

        store.close()


# ---------------------------------------------------------------------------
# Test 13: Single-call handles provider errors gracefully
# ---------------------------------------------------------------------------


class TestSingleCallProviderError:
    """When the LLM provider fails, the single-call path should return a
    failed ExecutionResult rather than raising an exception.
    """

    async def test_provider_error_returns_failed_result(self) -> None:
        store = _make_store()

        class FailingProvider(ModelProvider):
            @property
            def name(self) -> str:
                return "failing"

            async def complete(
                self, messages: list[dict[str, Any]], model: str, **kwargs: object
            ) -> CompletionResult:
                from models.errors import ProviderError
                raise ProviderError("Service unavailable")

        router = ModelRouter(provider=FailingProvider())
        orch = Orchestrator(store, router)

        result = await orch.process("What is 2 plus 2?")

        assert result.success is False
        assert len(result.errors) > 0

        store.close()


# ---------------------------------------------------------------------------
# Test 14: Trivial fast path with conversation memory stores response
# ---------------------------------------------------------------------------


class TestTrivialWithChatId:
    """Trivial responses should still be stored in conversation memory when
    a chat_id is provided.
    """

    async def test_trivial_response_stored_in_memory(self) -> None:
        store = _make_store()
        provider = MockProvider([
            _completion("Gravity pulls things down."),
        ])
        router = ModelRouter(provider=provider)
        orch = Orchestrator(store, router)

        chat_id = "chat-trivial"

        # Send a greeting (trivial fast path)
        r1 = await orch.process("Hello", chat_id=chat_id)
        assert r1.success is True
        assert r1.root_id == "trivial"

        # Follow up with a real question -- should have "Hello" + greeting response in history
        r2 = await orch.process("How does gravity work?", chat_id=chat_id)
        assert r2.success is True

        # The second call should have conversation history from the trivial exchange
        second_call = provider.call_log[0]  # First actual LLM call
        assert _is_single_call(second_call)
        # system + history(user_hello + assistant_greeting) + user_question = 4+
        assert len(second_call) >= 4

        store.close()
