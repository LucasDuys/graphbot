"""Tests for simple task latency optimization (T182).

Validates:
1. Trivial query fast path returns immediate response without pipeline
2. Pattern cache is lazy-loaded (not loaded on every request)
3. Entity resolution is skipped for complexity=1 tasks
4. Layer 1 verification is skipped for complexity=1 by default
5. Pipeline stage latency is logged
6. Simple task latency stays under budget
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.intake import IntakeParser, IntakeResult, TaskType
from core_gb.types import Domain, ExecutionResult, Pattern
from core_gb.verification import VerificationConfig


def _make_safe_classifier() -> MagicMock:
    """Create a mock IntentClassifier that does NOT block anything."""
    cls = MagicMock()
    cls.classify_text.return_value = MagicMock(blocked=False)
    cls.classify_dag.return_value = MagicMock(blocked=False)
    return cls


def _make_safe_constitutional() -> MagicMock:
    """Create a mock ConstitutionalChecker that passes everything."""
    checker = MagicMock()
    checker.check_text.return_value = MagicMock(passed=True, violations=[])
    checker.check_plan.return_value = MagicMock(passed=True, violations=[])
    return checker


# ---------------------------------------------------------------------------
# 1. Trivial Query Fast Path Tests
# ---------------------------------------------------------------------------


class TestTrivialQueryFastPath:
    """Test that trivial queries (greetings, acknowledgments) are intercepted
    and return an immediate response without entering the pipeline."""

    def test_intake_parser_detects_trivial_greeting(self) -> None:
        """IntakeParser.parse should flag 'hello' as trivial."""
        parser = IntakeParser()
        result = parser.parse("hello")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_hi(self) -> None:
        parser = IntakeParser()
        result = parser.parse("hi")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_thanks(self) -> None:
        parser = IntakeParser()
        result = parser.parse("thanks")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_thank_you(self) -> None:
        parser = IntakeParser()
        result = parser.parse("thank you")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_hey(self) -> None:
        parser = IntakeParser()
        result = parser.parse("hey there")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_ok(self) -> None:
        parser = IntakeParser()
        result = parser.parse("ok")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_yes(self) -> None:
        parser = IntakeParser()
        result = parser.parse("yes")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_no(self) -> None:
        parser = IntakeParser()
        result = parser.parse("no")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_goodbye(self) -> None:
        parser = IntakeParser()
        result = parser.parse("goodbye")
        assert result.is_trivial is True

    def test_intake_parser_detects_trivial_good_morning(self) -> None:
        parser = IntakeParser()
        result = parser.parse("good morning")
        assert result.is_trivial is True

    def test_intake_parser_not_trivial_for_real_question(self) -> None:
        """A real question should NOT be flagged as trivial."""
        parser = IntakeParser()
        result = parser.parse("What is the capital of France?")
        assert result.is_trivial is False

    def test_intake_parser_not_trivial_for_complex_task(self) -> None:
        parser = IntakeParser()
        result = parser.parse("Compare Python and JavaScript performance for web development")
        assert result.is_trivial is False

    def test_intake_parser_not_trivial_for_tool_request(self) -> None:
        parser = IntakeParser()
        result = parser.parse("Read the file README.md")
        assert result.is_trivial is False

    def test_trivial_response_content_is_nonempty(self) -> None:
        """IntakeParser.trivial_response should return a non-empty string."""
        parser = IntakeParser()
        result = parser.parse("hello")
        assert result.is_trivial is True
        response = parser.trivial_response(result)
        assert response is not None
        assert len(response) > 0

    def test_trivial_response_for_greeting_is_greeting(self) -> None:
        """Trivial response for a greeting should be a greeting back."""
        parser = IntakeParser()
        result = parser.parse("hello")
        response = parser.trivial_response(result)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_trivial_response_for_thanks_is_welcome(self) -> None:
        """Trivial response for 'thanks' should acknowledge."""
        parser = IntakeParser()
        result = parser.parse("thanks")
        response = parser.trivial_response(result)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_trivial_response_returns_none_for_nontrivial(self) -> None:
        """trivial_response should return None for non-trivial queries."""
        parser = IntakeParser()
        result = parser.parse("What is the capital of France?")
        response = parser.trivial_response(result)
        assert response is None


# ---------------------------------------------------------------------------
# 2. Lazy Pattern Cache Tests
# ---------------------------------------------------------------------------


class TestLazyPatternCache:
    """Test that PatternStore caches results and only reloads when needed."""

    def test_pattern_store_caches_load_all(self) -> None:
        """Second call to load_all should return cached result without query."""
        from core_gb.patterns import PatternStore

        mock_store = MagicMock()
        mock_store.query.return_value = []
        ps = PatternStore(mock_store)

        # First call: should hit the store
        result1 = ps.load_all()
        assert mock_store.query.call_count == 1

        # Second call: should return cached, no new query
        result2 = ps.load_all()
        assert mock_store.query.call_count == 1
        assert result1 == result2

    def test_pattern_store_cache_invalidates_after_save(self) -> None:
        """Saving a new pattern should invalidate the cache."""
        from core_gb.patterns import PatternStore

        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_store.create_node.return_value = "new-id"
        ps = PatternStore(mock_store)

        # Load to populate cache
        ps.load_all()
        assert mock_store.query.call_count == 1

        # Save invalidates
        pattern = Pattern(
            id="test-1",
            trigger="test trigger",
            description="test",
        )
        ps.save(pattern)

        # Next load_all should re-query
        ps.load_all()
        assert mock_store.query.call_count == 2

    def test_pattern_store_invalidate_cache_method(self) -> None:
        """invalidate_cache should force re-query on next load_all."""
        from core_gb.patterns import PatternStore

        mock_store = MagicMock()
        mock_store.query.return_value = []
        ps = PatternStore(mock_store)

        ps.load_all()
        assert mock_store.query.call_count == 1

        ps.invalidate_cache()

        ps.load_all()
        assert mock_store.query.call_count == 2


# ---------------------------------------------------------------------------
# 3. Skip Entity Resolution for complexity=1
# ---------------------------------------------------------------------------


class TestSkipEntityResolution:
    """Test that entity resolution is skipped for complexity=1 simple tasks."""

    @pytest.mark.asyncio
    async def test_simple_task_skips_entity_resolution(self) -> None:
        """When intake says is_simple=True and complexity=1,
        entity resolution should NOT be called."""
        from core_gb.orchestrator import Orchestrator

        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_router = MagicMock()

        with patch.object(Orchestrator, "__init__", lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        # Use a message that IntakeParser classifies as is_simple=True.
        # "Tell me a joke" -> complexity=1, no entities, single domain.
        orch._intake = IntakeParser()
        orch._store = mock_store
        orch._router = mock_router
        orch._pattern_store = MagicMock()
        orch._pattern_store.load_all.return_value = []
        orch._pattern_matcher = MagicMock()
        orch._resolver = MagicMock()
        orch._graph_updater = MagicMock()
        orch._executor = AsyncMock()
        orch._executor.execute.return_value = ExecutionResult(
            root_id="test", output="Why did the chicken cross the road?", success=True,
        )
        orch._dag_executor = MagicMock()
        orch._decomposer = AsyncMock()
        orch._intent_classifier = _make_safe_classifier()
        orch._constitutional_checker = _make_safe_constitutional()
        orch._verification_config = VerificationConfig()
        orch._enable_replan = False

        result = await orch.process("Tell me a joke")
        assert result.success is True
        # Entity resolution should NOT have been called for this simple query
        orch._resolver.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_complex_task_does_entity_resolution(self) -> None:
        """Complex tasks (is_simple=False) should still do entity resolution."""
        from core_gb.orchestrator import Orchestrator

        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_store.get_context.return_value = None
        mock_router = MagicMock()

        with patch.object(Orchestrator, "__init__", lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        orch._intake = MagicMock()
        orch._intake.parse.return_value = IntakeResult(
            domain=Domain.SYNTHESIS,
            complexity=4,
            entities=("Python", "JavaScript"),
            is_simple=False,
            raw_message="Compare Python and JavaScript",
            task_type=TaskType.SEQUENTIAL,
            is_trivial=False,
        )
        orch._intake.trivial_response.return_value = None
        orch._store = mock_store
        orch._router = mock_router
        orch._pattern_store = MagicMock()
        orch._pattern_store.load_all.return_value = []
        orch._pattern_matcher = MagicMock()
        orch._pattern_matcher.match.return_value = None
        orch._resolver = MagicMock()
        orch._resolver.resolve.return_value = [("eid-1", 0.9)]
        orch._graph_updater = MagicMock()
        orch._executor = AsyncMock()
        orch._executor.execute.return_value = ExecutionResult(
            root_id="test", output="Comparison result", success=True,
        )
        orch._dag_executor = MagicMock()
        orch._decomposer = AsyncMock()
        orch._decomposer.decompose.return_value = []
        orch._intent_classifier = _make_safe_classifier()
        orch._constitutional_checker = _make_safe_constitutional()
        orch._verification_config = VerificationConfig()
        orch._enable_replan = False

        await orch.process("Compare Python and JavaScript")
        # Entity resolution SHOULD have been called for complex task
        assert orch._resolver.resolve.call_count > 0


# ---------------------------------------------------------------------------
# 4. Skip Verification for complexity=1
# ---------------------------------------------------------------------------


class TestSkipVerificationForSimple:
    """Test that Layer 1 verification is skipped for complexity=1."""

    def test_verification_config_skip_for_simple(self) -> None:
        """VerificationConfig should have a skip_layer1_for_simple flag."""
        config = VerificationConfig()
        assert hasattr(config, "skip_layer1_for_simple")
        assert config.skip_layer1_for_simple is True  # default: skip for simple

    def test_verification_config_skip_disabled(self) -> None:
        """skip_layer1_for_simple=False should not skip."""
        config = VerificationConfig(skip_layer1_for_simple=False)
        assert config.skip_layer1_for_simple is False


# ---------------------------------------------------------------------------
# 5. Pipeline Stage Latency Logging
# ---------------------------------------------------------------------------


class TestPipelineLatencyLogging:
    """Test that pipeline stages log their latency."""

    @pytest.mark.asyncio
    async def test_orchestrator_logs_stage_latencies(self, caplog: Any) -> None:
        """Orchestrator.process should log latency for each pipeline stage."""
        from core_gb.orchestrator import Orchestrator

        with patch.object(Orchestrator, "__init__", lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        # Use a message that classifies as simple to avoid needing AsyncMock decomposer.
        # "Tell me a joke" -> is_simple=True, hits the simple execute path.
        orch._intake = IntakeParser()
        orch._store = MagicMock()
        orch._store.query.return_value = []
        orch._router = MagicMock()
        orch._pattern_store = MagicMock()
        orch._pattern_store.load_all.return_value = []
        orch._pattern_matcher = MagicMock()
        orch._resolver = MagicMock()
        orch._graph_updater = MagicMock()
        orch._executor = AsyncMock()
        orch._executor.execute.return_value = ExecutionResult(
            root_id="test", output="A joke", success=True,
        )
        orch._dag_executor = MagicMock()
        orch._decomposer = AsyncMock()
        orch._intent_classifier = _make_safe_classifier()
        orch._constitutional_checker = _make_safe_constitutional()
        orch._verification_config = VerificationConfig()
        orch._enable_replan = False

        with caplog.at_level(logging.DEBUG, logger="core_gb.orchestrator"):
            await orch.process("Tell me a joke")

        # Check that latency logs are present
        latency_logs = [r for r in caplog.records if "latency" in r.message.lower()]
        assert len(latency_logs) > 0, "Expected pipeline stage latency logs"


# ---------------------------------------------------------------------------
# 6. Trivial Query Performance
# ---------------------------------------------------------------------------


class TestTrivialQueryPerformance:
    """Test that trivial queries resolve in under 100ms."""

    @pytest.mark.asyncio
    async def test_trivial_query_returns_under_100ms(self) -> None:
        """A trivial query like 'hello' should return in <100ms from Orchestrator."""
        from core_gb.orchestrator import Orchestrator

        with patch.object(Orchestrator, "__init__", lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        orch._intake = IntakeParser()
        orch._store = MagicMock()
        orch._store.query.return_value = []
        orch._router = MagicMock()
        orch._pattern_store = MagicMock()
        orch._pattern_store.load_all.return_value = []
        orch._pattern_matcher = MagicMock()
        orch._resolver = MagicMock()
        orch._graph_updater = MagicMock()
        orch._executor = AsyncMock()
        orch._dag_executor = MagicMock()
        orch._decomposer = MagicMock()
        orch._intent_classifier = _make_safe_classifier()
        orch._constitutional_checker = _make_safe_constitutional()
        orch._verification_config = VerificationConfig()
        orch._enable_replan = False

        start = time.perf_counter()
        result = await orch.process("hello")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.success is True
        assert elapsed_ms < 100, f"Trivial query took {elapsed_ms:.1f}ms, expected <100ms"
        # Should NOT have called the LLM executor
        orch._executor.execute.assert_not_called()
