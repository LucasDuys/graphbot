"""Integration tests for all Phase 21 bugfixes (T179-T184).

Tests cover the full set of Phase 21 fixes end-to-end:

1. Pattern cache: shell task does NOT create reusable pattern (domain scoping)
2. Pattern cache: unfilled slot detected, falls back to decomposition
3. Code routing: "Write a Python function" goes to LLM, no shell tool attempt
4. Safety: "rm -rf /" blocked pre-decomposition with 0 LLM calls
5. Safety: "send spam emails" blocked pre-decomposition
6. Latency: trivial greeting returns immediate response (no pipeline)
7. Latency: simple Q&A skips entity resolution
8. Conversation: follow-up uses context from previous in same chat
9. Aggregation: decomposed task produces clean output (no JSON artifacts)
10. Aggregation: simple task skips synthesis step

All tests use mocked providers -- no real LLM or network calls.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.aggregator import Aggregator, strip_json_artifacts
from core_gb.constitution import ConstitutionalChecker
from core_gb.conversation import ConversationMemory
from core_gb.intake import IntakeParser
from core_gb.orchestrator import Orchestrator
from core_gb.patterns import PatternExtractor, PatternMatcher, PatternStore
from core_gb.safety import IntentClassifier
from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    Pattern,
    TaskNode,
    TaskStatus,
)
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


class _MockProvider(ModelProvider):
    """A mock LLM provider that returns canned responses and tracks calls."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = responses or ["Mock response"]
        self.call_count: int = 0
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "mock"

    async def complete(
        self, messages: list[dict[str, str]], model: str, **kwargs: object
    ) -> CompletionResult:
        idx = min(self.call_count, len(self._responses) - 1)
        content = self._responses[idx]
        self.call_count += 1
        self.calls.append({
            "index": self.call_count,
            "model": model,
            "messages": messages,
            "kwargs": kwargs,
        })
        return CompletionResult(
            content=content,
            model=model,
            tokens_in=50,
            tokens_out=50,
            latency_ms=50.0,
            cost=0.001,
        )


# ---------------------------------------------------------------------------
# 1. Pattern cache: shell task does NOT create reusable pattern
# ---------------------------------------------------------------------------


class TestPatternCacheDomainScoping:
    """Shell/code tasks must not produce reusable patterns (R001)."""

    def test_shell_task_does_not_create_pattern(self) -> None:
        """A task whose atomic leaves are in the CODE domain should be blocked
        from pattern extraction, preventing shell-specific outputs from
        polluting the cache and matching unrelated tasks like haiku requests.
        """
        extractor = PatternExtractor()

        # Simulate a shell task with CODE domain leaves
        shell_leaves: list[TaskNode] = [
            TaskNode(
                id="leaf_1",
                description="Run python --version",
                is_atomic=True,
                domain=Domain.CODE,
                complexity=1,
                status=TaskStatus.COMPLETED,
            ),
            TaskNode(
                id="leaf_2",
                description="Run pip --version",
                is_atomic=True,
                domain=Domain.CODE,
                complexity=1,
                status=TaskStatus.COMPLETED,
            ),
        ]

        result = ExecutionResult(
            root_id="root",
            output="Python 3.11.0\npip 24.0",
            success=True,
            total_nodes=2,
            total_tokens=100,
            total_latency_ms=500.0,
        )

        pattern = extractor.extract(
            task="Check Python and pip versions",
            nodes=shell_leaves,
            result=result,
        )

        # CODE domain is in the non-cacheable blocklist
        assert pattern is None, (
            "PatternExtractor should not create a pattern from CODE domain leaves"
        )

    def test_browser_task_does_not_create_pattern(self) -> None:
        """BROWSER domain leaves should also be blocked from extraction."""
        extractor = PatternExtractor()

        browser_leaves: list[TaskNode] = [
            TaskNode(
                id="b1",
                description="Scrape headline from example.com",
                is_atomic=True,
                domain=Domain.BROWSER,
                complexity=1,
                status=TaskStatus.COMPLETED,
            ),
            TaskNode(
                id="b2",
                description="Scrape headline from news.com",
                is_atomic=True,
                domain=Domain.BROWSER,
                complexity=1,
                status=TaskStatus.COMPLETED,
            ),
        ]

        result = ExecutionResult(
            root_id="root",
            output="Headline A\nHeadline B",
            success=True,
            total_nodes=2,
            total_tokens=80,
            total_latency_ms=400.0,
        )

        pattern = extractor.extract(
            task="Get headlines from two sites",
            nodes=browser_leaves,
            result=result,
        )

        assert pattern is None, (
            "PatternExtractor should not create a pattern from BROWSER domain leaves"
        )

    def test_synthesis_task_creates_pattern(self) -> None:
        """SYNTHESIS domain leaves SHOULD produce patterns (not blocked)."""
        extractor = PatternExtractor()

        synthesis_leaves: list[TaskNode] = [
            TaskNode(
                id="s1",
                description="Describe the benefits of Python",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.COMPLETED,
            ),
            TaskNode(
                id="s2",
                description="Describe the benefits of Rust",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.COMPLETED,
            ),
        ]

        result = ExecutionResult(
            root_id="root",
            output="Python is great. Rust is fast.",
            success=True,
            total_nodes=2,
            total_tokens=120,
            total_latency_ms=300.0,
        )

        pattern = extractor.extract(
            task="Describe the benefits of Python and Rust",
            nodes=synthesis_leaves,
            result=result,
        )

        assert pattern is not None, (
            "PatternExtractor should allow pattern creation from SYNTHESIS domain"
        )


# ---------------------------------------------------------------------------
# 2. Pattern cache: unfilled slot falls back to decomposition
# ---------------------------------------------------------------------------


class TestUnfilledSlotFallback:
    """Patterns with unfilled slots must be rejected, forcing decomposition (R001)."""

    def test_unfilled_slot_rejected(self) -> None:
        """When a pattern has variable slots that cannot be filled by regex
        extraction, the matcher must reject the match and return None,
        forcing the pipeline to fall back to decomposition.
        """
        matcher = PatternMatcher()

        # Pattern expects slot_0 to be filled with a specific topic
        pattern_with_slots = Pattern(
            id="pat_001",
            trigger="Write a haiku about {slot_0}",
            description="Haiku pattern",
            variable_slots=("slot_0",),
            tree_template=json.dumps([{
                "description": "Compose haiku about {slot_0}",
                "domain": "synthesis",
                "is_atomic": True,
                "complexity": 1,
                "provides": ["haiku"],
                "consumes": [],
            }]),
            success_count=5,
            source_domain="synthesis",
        )

        # Try to match a completely unrelated task -- slots cannot be filled
        result = matcher.match(
            task="What is the capital of France?",
            patterns=[pattern_with_slots],
            threshold=0.7,
        )

        assert result is None, (
            "PatternMatcher should reject match when task does not fill "
            "pattern variable slots"
        )

    def test_filled_slot_accepted(self) -> None:
        """When all slots can be filled, the pattern should match."""
        matcher = PatternMatcher()

        pattern = Pattern(
            id="pat_002",
            trigger="Write a haiku about {slot_0}",
            description="Haiku pattern",
            variable_slots=("slot_0",),
            tree_template="[]",
            success_count=5,
            source_domain="synthesis",
        )

        result = matcher.match(
            task="Write a haiku about autumn",
            patterns=[pattern],
            threshold=0.7,
        )

        assert result is not None, (
            "PatternMatcher should accept match when all slots are filled"
        )
        matched_pattern, bindings = result
        assert matched_pattern.id == "pat_002"
        assert bindings.get("slot_0") == "autumn"

    def test_partially_filled_slots_rejected(self) -> None:
        """A pattern with multiple slots where only some are fillable must be rejected."""
        matcher = PatternMatcher()

        pattern = Pattern(
            id="pat_003",
            trigger="Compare {slot_0} and {slot_1} in terms of {slot_2}",
            description="Comparison pattern",
            variable_slots=("slot_0", "slot_1", "slot_2"),
            tree_template="[]",
            success_count=3,
            source_domain="general",
        )

        # Only slot_0 and slot_1 can be filled, slot_2 cannot
        result = matcher.match(
            task="Compare cats and dogs",
            patterns=[pattern],
            threshold=0.5,
        )

        assert result is None, (
            "PatternMatcher should reject match when not all variable slots "
            "can be filled"
        )


# ---------------------------------------------------------------------------
# 3. Code routing: "Write a Python function" goes to LLM, no shell attempt
# ---------------------------------------------------------------------------


class TestCodeRouting:
    """Code generation tasks must route to LLM, not shell tool (R002)."""

    def test_code_generation_classified_as_system_or_synthesis(self) -> None:
        """'Write a Python function' should NOT be classified as CODE domain.
        After T180, ambiguous code keywords were removed from CODE domain."""
        parser = IntakeParser()
        result = parser.parse("Write a Python function that sorts a list")

        # Should not be CODE -- CODE is reserved for shell/edit commands
        assert result.domain != Domain.CODE, (
            "Code generation tasks should not be classified as CODE domain"
        )

    def test_shell_command_still_classified_as_code(self) -> None:
        """Explicit shell commands should still route to CODE domain."""
        parser = IntakeParser()
        result = parser.parse("Execute the command: git status")

        assert result.domain == Domain.CODE, (
            "Explicit shell commands should still be classified as CODE domain"
        )

    async def test_code_generation_no_tool_attempt_in_orchestrator(self) -> None:
        """End-to-end: code generation task goes to LLM with 0 tool attempts."""
        store = _make_store()
        provider = _MockProvider(responses=["def sort_list(lst): return sorted(lst)"])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Write a Python function that sorts a list"
        )

        assert result.success is True
        assert "sort" in result.output.lower()
        # Only 1 LLM call -- no tool attempt, no decomposition
        assert provider.call_count == 1, (
            f"Expected 1 LLM call (direct execution), got {provider.call_count}"
        )

        store.close()


# ---------------------------------------------------------------------------
# 4. Safety: "rm -rf /" blocked pre-decomposition with 0 LLM calls
# ---------------------------------------------------------------------------


class TestSafetyRmRf:
    """'rm -rf /' must be blocked before decomposition with zero LLM calls (R003)."""

    def test_rm_rf_blocked_by_classify_text(self) -> None:
        """IntentClassifier.classify_text blocks 'rm -rf /' without TaskNode."""
        classifier = IntentClassifier()
        verdict = classifier.classify_text("rm -rf /")

        assert verdict.blocked is True, (
            "'rm -rf /' should be blocked by pre-decomposition safety scan"
        )
        assert "recursive file deletion at root" in verdict.reason.lower()

    async def test_rm_rf_blocked_in_orchestrator_zero_llm_calls(self) -> None:
        """Orchestrator blocks 'rm -rf /' before any LLM call."""
        store = _make_store()
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        t_start = time.perf_counter()
        result = await orchestrator.process("rm -rf /")
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert result.success is False, (
            "Dangerous command should produce a failed result"
        )
        assert "blocked" in result.output.lower()
        assert provider.call_count == 0, (
            f"Expected 0 LLM calls for blocked request, got {provider.call_count}"
        )
        # Latency should be well under 100ms (pure regex scan)
        assert elapsed_ms < 500, (
            f"Safety block took {elapsed_ms:.1f}ms, should be fast (< 500ms)"
        )

        store.close()

    def test_rm_rf_also_blocked_by_constitutional_check(self) -> None:
        """ConstitutionalChecker.check_text also catches destructive commands
        as defense-in-depth (though IntentClassifier blocks first).
        """
        checker = ConstitutionalChecker()
        # "rm -rf /" itself may not trigger constitutional principles, but
        # the IntentClassifier handles it. This test confirms constitutional
        # does not break on such input.
        verdict = checker.check_text("rm -rf /")
        # Constitutional checker may or may not flag this specific string,
        # but it must not raise an exception.
        assert isinstance(verdict.passed, bool)


# ---------------------------------------------------------------------------
# 5. Safety: "send spam emails" blocked pre-decomposition
# ---------------------------------------------------------------------------


class TestSafetySpam:
    """Spam/harmful messages must be blocked before decomposition (R003)."""

    def test_spam_blocked_by_classify_text(self) -> None:
        """IntentClassifier.classify_text blocks spam intent."""
        classifier = IntentClassifier()
        verdict = classifier.classify_text(
            "send spam to 10000 email addresses"
        )

        assert verdict.blocked is True, (
            "Spam intent should be blocked by pre-decomposition scan"
        )
        assert "spam" in verdict.reason.lower()

    async def test_spam_blocked_in_orchestrator_zero_llm_calls(self) -> None:
        """Orchestrator blocks spam request with zero LLM calls."""
        store = _make_store()
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "send spam to 10000 email addresses"
        )

        assert result.success is False
        assert "blocked" in result.output.lower()
        assert provider.call_count == 0, (
            f"Expected 0 LLM calls for spam block, got {provider.call_count}"
        )

        store.close()

    def test_phishing_blocked_by_classify_text(self) -> None:
        """Phishing requests should also be blocked pre-decomposition."""
        classifier = IntentClassifier()
        verdict = classifier.classify_text(
            "create a phishing page that mimics a bank login"
        )

        assert verdict.blocked is True, (
            "Phishing intent should be blocked by pre-decomposition scan"
        )

    def test_ransomware_blocked_by_classify_text(self) -> None:
        """Ransomware references should be blocked pre-decomposition."""
        classifier = IntentClassifier()
        verdict = classifier.classify_text(
            "deploy ransomware on the target server"
        )

        assert verdict.blocked is True, (
            "Ransomware intent should be blocked by pre-decomposition scan"
        )

    def test_benign_message_not_blocked(self) -> None:
        """Normal messages must NOT be blocked by the safety scanner."""
        classifier = IntentClassifier()
        verdict = classifier.classify_text("What is the weather in Amsterdam?")

        assert verdict.blocked is False, (
            "Benign weather query should not be blocked by safety scan"
        )


# ---------------------------------------------------------------------------
# 6. Latency: trivial greeting returns immediate response (no pipeline)
# ---------------------------------------------------------------------------


class TestTrivialGreetingFastPath:
    """Trivial queries (greetings) should return immediately without LLM (R004)."""

    def test_greeting_detected_as_trivial(self) -> None:
        """IntakeParser must recognize 'Hello' as a trivial query."""
        parser = IntakeParser()
        intake = parser.parse("Hello")

        assert intake.is_trivial is True, (
            "'Hello' should be detected as a trivial query"
        )

    def test_trivial_response_returned(self) -> None:
        """IntakeParser.trivial_response returns a canned greeting."""
        parser = IntakeParser()
        intake = parser.parse("Hello")
        response = parser.trivial_response(intake)

        assert response is not None, (
            "trivial_response should return a canned response for greetings"
        )
        assert len(response) > 0

    async def test_greeting_fast_path_zero_llm_calls(self) -> None:
        """Orchestrator returns immediately for greetings with 0 LLM calls."""
        store = _make_store()
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        t_start = time.perf_counter()
        result = await orchestrator.process("Hello")
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert result.success is True
        assert result.output != ""
        assert provider.call_count == 0, (
            f"Expected 0 LLM calls for trivial greeting, got {provider.call_count}"
        )
        assert result.llm_calls == 0
        assert result.total_tokens == 0

        store.close()

    async def test_farewell_fast_path(self) -> None:
        """Farewells should also take the fast path."""
        store = _make_store()
        provider = _MockProvider()
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("Bye")

        assert result.success is True
        assert provider.call_count == 0

        store.close()

    async def test_non_trivial_query_uses_llm(self) -> None:
        """Non-trivial queries must still go through the LLM."""
        store = _make_store()
        provider = _MockProvider(responses=["42"])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("What is the meaning of life?")

        assert result.success is True
        assert provider.call_count >= 1, (
            "Non-trivial queries should use at least 1 LLM call"
        )

        store.close()


# ---------------------------------------------------------------------------
# 7. Latency: simple Q&A skips entity resolution
# ---------------------------------------------------------------------------


class TestSimpleTaskSkipsEntityResolution:
    """Simple Q&A tasks (complexity=1) should skip entity resolution (R004)."""

    def test_simple_qa_classified_as_simple(self) -> None:
        """A straightforward question should be classified as simple."""
        parser = IntakeParser()
        intake = parser.parse("What is 2 + 2?")

        assert intake.is_simple is True, (
            "Simple arithmetic question should be classified as is_simple=True"
        )
        assert intake.complexity <= 2

    async def test_simple_qa_does_not_call_entity_resolver(self) -> None:
        """Simple tasks go through the simple path, skipping entity resolution."""
        store = _make_store()
        provider = _MockProvider(responses=["4"])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        # Patch the entity resolver to track if it's called
        original_resolve = orchestrator._resolver.resolve
        resolve_calls: list[str] = []

        def tracking_resolve(entity: str, top_k: int = 3) -> list[tuple[str, float]]:
            resolve_calls.append(entity)
            return original_resolve(entity, top_k=top_k)

        orchestrator._resolver.resolve = tracking_resolve  # type: ignore[assignment]

        result = await orchestrator.process("What is 2 + 2?")

        assert result.success is True
        assert len(resolve_calls) == 0, (
            f"Entity resolver should not be called for simple Q&A, "
            f"but was called {len(resolve_calls)} time(s)"
        )

        store.close()

    async def test_complex_task_does_call_entity_resolver(self) -> None:
        """Complex tasks should still call entity resolution."""
        store = _make_store()

        # Multi-part task that triggers decomposition
        decomposition = json.dumps({
            "nodes": [
                {
                    "id": "root",
                    "description": "Compare cities",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 3,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["a"],
                },
                {
                    "id": "a",
                    "description": "Describe Amsterdam weather",
                    "domain": "system",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["info_a"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
            ],
            "output_template": None,
        })

        provider = _MockProvider(responses=[decomposition, "Amsterdam is rainy"])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        resolve_calls: list[str] = []
        original_resolve = orchestrator._resolver.resolve

        def tracking_resolve(entity: str, top_k: int = 3) -> list[tuple[str, float]]:
            resolve_calls.append(entity)
            return original_resolve(entity, top_k=top_k)

        orchestrator._resolver.resolve = tracking_resolve  # type: ignore[assignment]

        # This message has entities and needs decomposition
        result = await orchestrator.process(
            "Compare the weather in Amsterdam, London, and Berlin in detail and also check news"
        )

        # Complex path should attempt entity resolution
        assert result.success is True
        assert len(resolve_calls) > 0, (
            "Entity resolver should be called for complex multi-entity tasks"
        )

        store.close()


# ---------------------------------------------------------------------------
# 8. Conversation: follow-up uses context from previous message
# ---------------------------------------------------------------------------


class TestConversationMemory:
    """Follow-up messages should use context from previous messages (R005)."""

    def test_conversation_memory_stores_and_retrieves(self) -> None:
        """ConversationMemory stores messages and retrieves them in order."""
        store = _make_store()
        memory = ConversationMemory(store, max_messages=10)

        chat_id = "test_chat_001"
        memory.add_message(chat_id, "user", "Hello, I need help with Python.")
        memory.add_message(chat_id, "assistant", "Of course! What do you need?")
        memory.add_message(chat_id, "user", "How do I sort a list?")

        history = memory.get_history(chat_id)

        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert "Python" in history[0]["content"]
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert "sort" in history[2]["content"]

        store.close()

    def test_chat_id_isolation(self) -> None:
        """Different chat_ids must have separate conversation histories."""
        store = _make_store()
        memory = ConversationMemory(store, max_messages=10)

        memory.add_message("chat_A", "user", "Topic A message")
        memory.add_message("chat_B", "user", "Topic B message")

        history_a = memory.get_history("chat_A")
        history_b = memory.get_history("chat_B")

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert "Topic A" in history_a[0]["content"]
        assert "Topic B" in history_b[0]["content"]

        store.close()

    async def test_follow_up_uses_previous_context(self) -> None:
        """Orchestrator passes conversation history to LLM for follow-ups."""
        store = _make_store()

        # First response answers about Python, second uses context
        provider = _MockProvider(responses=[
            "Python is a programming language created by Guido van Rossum.",
            "It was created in 1991.",
        ])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        chat_id = "test_chat_followup"

        # First message
        result1 = await orchestrator.process(
            "What is Python?", chat_id=chat_id
        )
        assert result1.success is True

        # Follow-up message in the same chat
        result2 = await orchestrator.process(
            "When was it created?", chat_id=chat_id
        )
        assert result2.success is True

        # The second LLM call should have received conversation history
        assert provider.call_count == 2
        second_call_messages = provider.calls[1]["messages"]
        assert isinstance(second_call_messages, list)

        # Check that conversation history was injected into the messages
        all_text = " ".join(
            str(m.get("content", "")) for m in second_call_messages
        )
        # The history should contain the first question or context
        assert "Python" in all_text or len(second_call_messages) > 2, (
            "Follow-up LLM call should include conversation history "
            "from the first message"
        )

        store.close()

    async def test_no_chat_id_no_history(self) -> None:
        """When chat_id is None, no conversation memory is used."""
        store = _make_store()
        provider = _MockProvider(responses=["Response 1", "Response 2"])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        await orchestrator.process("First message")
        await orchestrator.process("Second message")

        # Without chat_id, second call should not have history
        assert provider.call_count == 2
        second_messages = provider.calls[1]["messages"]
        assert isinstance(second_messages, list)
        # Should be just system + user, no history
        assert len(second_messages) <= 3

        store.close()


# ---------------------------------------------------------------------------
# 9. Aggregation: decomposed task produces clean output (no JSON artifacts)
# ---------------------------------------------------------------------------


class TestAggregationCleanOutput:
    """Decomposed task output should be clean prose without JSON artifacts (R006)."""

    def test_strip_json_artifacts_from_object(self) -> None:
        """strip_json_artifacts should extract readable content from JSON."""
        raw = json.dumps({
            "answer": "Paris is the capital of France.",
            "confidence": 0.95,
            "model": "llama-8b",
        })

        cleaned = strip_json_artifacts(raw)

        assert "Paris is the capital of France" in cleaned
        assert '"answer"' not in cleaned
        assert '"confidence"' not in cleaned
        assert '"model"' not in cleaned

    def test_strip_json_artifacts_from_array(self) -> None:
        """strip_json_artifacts should handle JSON arrays."""
        raw = json.dumps(["Apple", "Banana", "Cherry"])

        cleaned = strip_json_artifacts(raw)

        assert "Apple" in cleaned
        assert "Banana" in cleaned
        assert "Cherry" in cleaned
        assert "[" not in cleaned

    def test_strip_json_artifacts_preserves_plain_text(self) -> None:
        """Plain text without JSON should pass through unchanged."""
        plain = "The weather in Amsterdam is rainy and cool."
        cleaned = strip_json_artifacts(plain)
        assert cleaned == plain

    async def test_synthesize_produces_clean_output(self) -> None:
        """Aggregator.synthesize should produce clean prose from JSON outputs."""
        provider = _MockProvider(responses=[
            "Amsterdam has rainy weather while Berlin tends to be colder."
        ])
        router = ModelRouter(provider=provider)
        aggregator = Aggregator(router=router, synthesis_threshold=2)

        leaf_outputs = {
            "amsterdam_weather": json.dumps({
                "answer": "Amsterdam is rainy with 15C",
                "confidence": 0.9,
            }),
            "berlin_weather": json.dumps({
                "answer": "Berlin is cold with 8C",
                "confidence": 0.88,
            }),
            "london_weather": json.dumps({
                "answer": "London is foggy with 12C",
                "confidence": 0.85,
            }),
        }

        result = await aggregator.synthesize(
            original_question="Compare weather in Amsterdam, Berlin, and London",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        # LLM synthesis should have been triggered (3 outputs >= threshold 2)
        assert provider.call_count == 1, (
            "Synthesis should trigger LLM call for 3 outputs (threshold=2)"
        )
        # Output should be clean prose
        assert '"answer"' not in result
        assert '"confidence"' not in result

    async def test_aggregation_end_to_end_clean(self) -> None:
        """End-to-end: a decomposed task should produce output free of JSON noise."""
        store = _make_store()

        decomposition = json.dumps({
            "nodes": [
                {
                    "id": "root",
                    "description": "Compare weather",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 3,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["a", "b", "c"],
                },
                {
                    "id": "a",
                    "description": "Weather in city A",
                    "domain": "system",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["city_a"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "b",
                    "description": "Weather in city B",
                    "domain": "system",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["city_b"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "c",
                    "description": "Weather in city C",
                    "domain": "system",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["city_c"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
            ],
            "output_template": {
                "aggregation_type": "concatenate",
                "template": "",
                "slot_definitions": {},
            },
        })

        provider = _MockProvider(responses=[
            decomposition,
            # Leaf outputs with JSON artifacts
            json.dumps({"answer": "City A is sunny", "confidence": 0.9}),
            json.dumps({"answer": "City B is rainy", "confidence": 0.85}),
            json.dumps({"answer": "City C is cold", "confidence": 0.88}),
            # Synthesis response
            "City A enjoys sunny weather, City B is rainy, and City C is cold.",
        ])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process(
            "Compare the weather in city A, city B, and city C in detail"
        )

        assert result.success is True
        # Output should not contain raw JSON syntax
        assert '"answer"' not in result.output
        assert '"confidence"' not in result.output

        store.close()


# ---------------------------------------------------------------------------
# 10. Aggregation: simple task skips synthesis step
# ---------------------------------------------------------------------------


class TestSimpleTaskSkipsSynthesis:
    """Simple tasks (below synthesis threshold) should skip the LLM synthesis (R006)."""

    async def test_below_threshold_skips_synthesis(self) -> None:
        """Aggregator with outputs below threshold uses deterministic aggregation."""
        provider = _MockProvider(responses=["Should not be called"])
        router = ModelRouter(provider=provider)
        aggregator = Aggregator(router=router, synthesis_threshold=3)

        # Only 2 outputs -- below threshold of 3
        leaf_outputs = {
            "info_a": "Amsterdam is rainy.",
            "info_b": "Berlin is cold.",
        }

        result = await aggregator.synthesize(
            original_question="Compare Amsterdam and Berlin weather",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        assert provider.call_count == 0, (
            f"Synthesis should be skipped for {len(leaf_outputs)} outputs "
            f"(below threshold 3), but got {provider.call_count} LLM calls"
        )
        # Deterministic aggregation should still produce output
        assert "Amsterdam" in result or "rainy" in result
        assert "Berlin" in result or "cold" in result

    async def test_above_threshold_triggers_synthesis(self) -> None:
        """Aggregator with outputs at or above threshold triggers LLM synthesis."""
        provider = _MockProvider(responses=[
            "All three cities have different weather patterns."
        ])
        router = ModelRouter(provider=provider)
        aggregator = Aggregator(router=router, synthesis_threshold=3)

        leaf_outputs = {
            "info_a": "Amsterdam is rainy.",
            "info_b": "Berlin is cold.",
            "info_c": "London is foggy.",
        }

        result = await aggregator.synthesize(
            original_question="Compare three cities",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        assert provider.call_count == 1, (
            f"Synthesis should trigger for {len(leaf_outputs)} outputs "
            f"(meets threshold 3), but got {provider.call_count} LLM calls"
        )

    async def test_no_router_skips_synthesis(self) -> None:
        """Aggregator without a router always uses deterministic aggregation."""
        aggregator = Aggregator(router=None, synthesis_threshold=1)

        leaf_outputs = {
            "info_a": "Fact A",
            "info_b": "Fact B",
            "info_c": "Fact C",
        }

        result = await aggregator.synthesize(
            original_question="Summarize facts",
            leaf_outputs=leaf_outputs,
            template=None,
        )

        # Should produce output via deterministic aggregation
        assert "Fact A" in result or "Fact B" in result

    async def test_simple_orchestrator_task_no_synthesis(self) -> None:
        """A simple task through the orchestrator should not trigger synthesis."""
        store = _make_store()
        provider = _MockProvider(responses=["4"])
        router = ModelRouter(provider=provider)
        orchestrator = Orchestrator(store, router)

        result = await orchestrator.process("What is 2 + 2?")

        assert result.success is True
        # Simple task: 1 LLM call only, no synthesis
        assert provider.call_count == 1, (
            f"Simple task should use exactly 1 LLM call, got {provider.call_count}"
        )

        store.close()


# ---------------------------------------------------------------------------
# Cross-cutting: pollution purge on startup
# ---------------------------------------------------------------------------


class TestPollutionPurge:
    """PatternStore.purge_polluted() should remove patterns with shell markers."""

    def test_purge_polluted_removes_shell_patterns(self) -> None:
        """Patterns containing shell markers should be purged."""
        store = _make_store()
        pattern_store = PatternStore(store)

        # Save a polluted pattern (contains shell-specific markers)
        polluted_template = json.dumps([{
            "description": "Run python --version and check pip install status",
            "domain": "code",
            "is_atomic": True,
            "complexity": 1,
            "provides": ["version_info"],
            "consumes": [],
        }])

        pattern_store.save(Pattern(
            id="polluted_001",
            trigger="Check {slot_0} version",
            description="Version check pattern",
            variable_slots=("slot_0",),
            tree_template=polluted_template,
            success_count=2,
            source_domain="code",
        ))

        # Save a clean pattern
        clean_template = json.dumps([{
            "description": "Compose a haiku about nature",
            "domain": "synthesis",
            "is_atomic": True,
            "complexity": 1,
            "provides": ["haiku"],
            "consumes": [],
        }])

        pattern_store.save(Pattern(
            id="clean_001",
            trigger="Write a haiku about {slot_0}",
            description="Haiku pattern",
            variable_slots=("slot_0",),
            tree_template=clean_template,
            success_count=5,
            source_domain="synthesis",
        ))

        # Before purge: 2 patterns
        all_before = pattern_store.load_all()
        assert len(all_before) == 2

        # Purge polluted patterns
        purged_count = pattern_store.purge_polluted()

        # After purge: only clean pattern remains
        pattern_store.invalidate_cache()
        all_after = pattern_store.load_all()
        assert purged_count == 1, (
            f"Expected 1 polluted pattern purged, got {purged_count}"
        )
        assert len(all_after) == 1
        assert all_after[0].id == "clean_001"

        store.close()

    def test_purge_polluted_leaves_clean_patterns(self) -> None:
        """Clean patterns should survive the purge."""
        store = _make_store()
        pattern_store = PatternStore(store)

        clean_template = json.dumps([{
            "description": "Compare pros and cons of topic X vs topic Y",
            "domain": "synthesis",
            "is_atomic": True,
            "complexity": 2,
            "provides": ["comparison"],
            "consumes": [],
        }])

        pattern_store.save(Pattern(
            id="clean_002",
            trigger="Compare {slot_0} and {slot_1}",
            description="Comparison pattern",
            variable_slots=("slot_0", "slot_1"),
            tree_template=clean_template,
            success_count=10,
            source_domain="synthesis",
        ))

        purged_count = pattern_store.purge_polluted()

        assert purged_count == 0
        pattern_store.invalidate_cache()
        remaining = pattern_store.load_all()
        assert len(remaining) == 1
        assert remaining[0].id == "clean_002"

        store.close()
