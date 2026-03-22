"""Tests for failure reflection engine -- generates structured reflections on failed tasks."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import CompletionResult, ExecutionResult, Reflection, TaskNode
from graph.reflection import ReflectionEngine
from graph.store import GraphStore
from graph.updater import GraphUpdater
from models.base import ModelProvider


def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _failure_result(root_id: str = "root") -> ExecutionResult:
    return ExecutionResult(
        root_id=root_id,
        output="",
        success=False,
        total_nodes=4,
        total_tokens=800,
        total_latency_ms=200.0,
        errors=("API timeout on weather lookup",),
    )


def _success_result(root_id: str = "root") -> ExecutionResult:
    return ExecutionResult(
        root_id=root_id,
        output="Sunny in all cities",
        success=True,
        total_nodes=4,
        total_tokens=1200,
        total_latency_ms=350.0,
    )


def _mock_provider(reflection_json: str | None = None) -> ModelProvider:
    """Create a mock ModelProvider that returns a reflection JSON."""
    if reflection_json is None:
        reflection_json = json.dumps({
            "what_failed": "Weather API call timed out for Amsterdam",
            "why": "The external weather API was unresponsive, likely due to rate limiting",
            "what_to_try": "Add retry logic with exponential backoff, or use a fallback weather API",
        })
    provider = MagicMock(spec=ModelProvider)
    provider.name = "mock"
    provider.complete = AsyncMock(return_value=CompletionResult(
        content=reflection_json,
        model="test-model",
        tokens_in=100,
        tokens_out=50,
        latency_ms=200.0,
        cost=0.0,
    ))
    return provider


class TestReflectionDataclass:
    """Reflection dataclass stores structured failure analysis."""

    def test_reflection_fields(self) -> None:
        """Reflection has required fields: what_failed, why, what_to_try."""
        r = Reflection(
            what_failed="API timeout",
            why="Rate limited",
            what_to_try="Add retry logic",
        )
        assert r.what_failed == "API timeout"
        assert r.why == "Rate limited"
        assert r.what_to_try == "Add retry logic"

    def test_reflection_is_frozen(self) -> None:
        """Reflection is immutable (frozen dataclass)."""
        r = Reflection(
            what_failed="API timeout",
            why="Rate limited",
            what_to_try="Add retry logic",
        )
        with pytest.raises(AttributeError):
            r.what_failed = "something else"  # type: ignore[misc]


class TestReflectionEngine:
    """ReflectionEngine generates structured reflections from failed tasks."""

    @pytest.mark.asyncio
    async def test_generates_reflection_from_failure(self) -> None:
        """reflect() returns a Reflection with structured fields from LLM output."""
        provider = _mock_provider()
        engine = ReflectionEngine(provider=provider)

        result = _failure_result()
        reflection = await engine.reflect(
            task_description="Weather in Amsterdam, London, and Berlin",
            result=result,
        )

        assert reflection is not None
        assert reflection.what_failed == "Weather API call timed out for Amsterdam"
        assert "rate limiting" in reflection.why.lower()
        assert "retry" in reflection.what_to_try.lower()

    @pytest.mark.asyncio
    async def test_calls_provider_with_correct_prompt(self) -> None:
        """reflect() sends a structured prompt to the LLM provider."""
        provider = _mock_provider()
        engine = ReflectionEngine(provider=provider)

        result = _failure_result()
        await engine.reflect(
            task_description="Weather in Amsterdam, London, and Berlin",
            result=result,
        )

        provider.complete.assert_called_once()
        call_args = provider.complete.call_args
        messages = call_args[0][0]
        model = call_args[0][1]

        # System message should instruct structured output
        assert any("reflection" in m.get("content", "").lower() for m in messages)
        # User message should contain task description and errors
        assert any("Weather in Amsterdam" in m.get("content", "") for m in messages)
        assert any("API timeout" in m.get("content", "") for m in messages)
        assert isinstance(model, str)

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_output(self) -> None:
        """reflect() returns None when LLM output is not valid JSON."""
        provider = _mock_provider(reflection_json="not valid json at all")
        engine = ReflectionEngine(provider=provider)

        result = _failure_result()
        reflection = await engine.reflect(
            task_description="Weather in Amsterdam",
            result=result,
        )

        assert reflection is None

    @pytest.mark.asyncio
    async def test_handles_incomplete_json_output(self) -> None:
        """reflect() returns None when LLM output is JSON but missing required fields."""
        incomplete = json.dumps({"what_failed": "something"})
        provider = _mock_provider(reflection_json=incomplete)
        engine = ReflectionEngine(provider=provider)

        result = _failure_result()
        reflection = await engine.reflect(
            task_description="Weather in Amsterdam",
            result=result,
        )

        assert reflection is None

    @pytest.mark.asyncio
    async def test_handles_provider_error(self) -> None:
        """reflect() returns None when the LLM provider raises an exception."""
        provider = MagicMock(spec=ModelProvider)
        provider.name = "mock"
        provider.complete = AsyncMock(side_effect=Exception("provider down"))
        engine = ReflectionEngine(provider=provider)

        result = _failure_result()
        reflection = await engine.reflect(
            task_description="Weather in Amsterdam",
            result=result,
        )

        assert reflection is None

    @pytest.mark.asyncio
    async def test_skips_reflection_on_success(self) -> None:
        """reflect() returns None when result.success is True."""
        provider = _mock_provider()
        engine = ReflectionEngine(provider=provider)

        result = _success_result()
        reflection = await engine.reflect(
            task_description="Weather in Amsterdam",
            result=result,
        )

        assert reflection is None
        provider.complete.assert_not_called()


class TestReflectionOfEdgeType:
    """REFLECTION_OF edge type exists in the schema."""

    def test_reflection_of_edge_in_schema(self) -> None:
        """REFLECTION_OF edge type is defined in EDGE_TYPES."""
        from graph.schema import EDGE_TYPES

        edge_names = [e.name for e in EDGE_TYPES]
        assert "REFLECTION_OF" in edge_names

    def test_reflection_of_edge_direction(self) -> None:
        """REFLECTION_OF goes from Memory to Task."""
        from graph.schema import EDGE_TYPES

        edge = next(e for e in EDGE_TYPES if e.name == "REFLECTION_OF")
        assert edge.from_type == "Memory"
        assert edge.to_type == "Task"

    def test_reflection_of_table_created(self) -> None:
        """REFLECTION_OF edge table is created during store.initialize()."""
        store = _make_store()
        # If initialization succeeded and we can query, the table exists.
        # Verify by creating nodes and an edge.
        store.create_node("Memory", {
            "id": "mem_test",
            "content": "test",
            "category": "reflection",
            "confidence": 1.0,
            "source_episode": "task_test",
        })
        store.create_node("Task", {
            "id": "task_test",
            "description": "test task",
            "domain": "synthesis",
            "complexity": 1,
            "status": "failed",
            "tokens_used": 0,
            "latency_ms": 0.0,
        })
        result = store.create_edge("REFLECTION_OF", "mem_test", "task_test")
        assert result is True
        store.close()


class TestGraphUpdaterReflection:
    """GraphUpdater invokes reflection on failure and stores results in graph."""

    @pytest.mark.asyncio
    async def test_stores_reflection_memory_node_on_failure(self) -> None:
        """On failure, updater stores a Memory node with category=reflection."""
        store = _make_store()
        provider = _mock_provider()
        engine = ReflectionEngine(provider=provider)
        updater = GraphUpdater(store, reflection_engine=engine)

        result = _failure_result()
        await updater.update_async(
            "Weather in Amsterdam, London, and Berlin",
            [],
            result,
        )

        # Verify the Task node was created
        task_node = store.get_node("Task", "root")
        assert task_node is not None
        assert task_node["status"] == "failed"

        # Find the Memory node via REFLECTION_OF edge
        rows = store.query(
            "MATCH (m:Memory)-[:REFLECTION_OF]->(t:Task) "
            "WHERE t.id = $tid RETURN m.id, m.content, m.category, m.source_episode",
            {"tid": "root"},
        )
        assert len(rows) == 1
        memory_row = rows[0]
        assert memory_row["m.category"] == "reflection"
        assert memory_row["m.source_episode"] == "root"

        # Content should be the structured reflection as JSON
        content = json.loads(str(memory_row["m.content"]))
        assert "what_failed" in content
        assert "why" in content
        assert "what_to_try" in content
        store.close()

    @pytest.mark.asyncio
    async def test_creates_reflection_of_edge(self) -> None:
        """On failure, updater creates a REFLECTION_OF edge from Memory to Task."""
        store = _make_store()
        provider = _mock_provider()
        engine = ReflectionEngine(provider=provider)
        updater = GraphUpdater(store, reflection_engine=engine)

        result = _failure_result()
        await updater.update_async(
            "Weather in Amsterdam",
            [],
            result,
        )

        rows = store.query(
            "MATCH (m:Memory)-[:REFLECTION_OF]->(t:Task) "
            "WHERE t.id = $tid RETURN m.id, t.id",
            {"tid": "root"},
        )
        assert len(rows) == 1
        store.close()

    @pytest.mark.asyncio
    async def test_no_reflection_on_success(self) -> None:
        """On success, updater does not create a reflection Memory node."""
        store = _make_store()
        provider = _mock_provider()
        engine = ReflectionEngine(provider=provider)
        updater = GraphUpdater(store, reflection_engine=engine)

        result = _success_result()
        await updater.update_async(
            "Weather in Amsterdam",
            [],
            result,
        )

        rows = store.query(
            "MATCH (m:Memory)-[:REFLECTION_OF]->(t:Task) RETURN m.id",
        )
        assert len(rows) == 0
        provider.complete.assert_not_called()
        store.close()

    @pytest.mark.asyncio
    async def test_reflection_failure_does_not_break_update(self) -> None:
        """If reflection engine fails, update still succeeds."""
        store = _make_store()
        provider = MagicMock(spec=ModelProvider)
        provider.name = "mock"
        provider.complete = AsyncMock(side_effect=Exception("provider down"))
        engine = ReflectionEngine(provider=provider)
        updater = GraphUpdater(store, reflection_engine=engine)

        result = _failure_result()
        # Should not raise
        await updater.update_async(
            "Weather in Amsterdam",
            [],
            result,
        )

        # Task node should still be recorded
        task_node = store.get_node("Task", "root")
        assert task_node is not None
        assert task_node["status"] == "failed"
        store.close()

    def test_sync_update_still_works_without_engine(self) -> None:
        """Original sync update() still works when no reflection engine is provided."""
        store = _make_store()
        updater = GraphUpdater(store)

        result = _failure_result()
        pattern_id = updater.update(
            "Weather in Amsterdam",
            [],
            result,
        )

        assert pattern_id is None
        task_node = store.get_node("Task", "root")
        assert task_node is not None
        assert task_node["status"] == "failed"
        store.close()

    @pytest.mark.asyncio
    async def test_reflection_memory_content_is_valid_json(self) -> None:
        """Reflection Memory node content is parseable JSON with expected structure."""
        store = _make_store()
        provider = _mock_provider()
        engine = ReflectionEngine(provider=provider)
        updater = GraphUpdater(store, reflection_engine=engine)

        result = _failure_result()
        await updater.update_async(
            "Weather in Amsterdam",
            [],
            result,
        )

        rows = store.query(
            "MATCH (m:Memory)-[:REFLECTION_OF]->(t:Task) "
            "WHERE t.id = $tid RETURN m.content",
            {"tid": "root"},
        )
        assert len(rows) == 1
        content = json.loads(str(rows[0]["m.content"]))
        assert isinstance(content["what_failed"], str)
        assert isinstance(content["why"], str)
        assert isinstance(content["what_to_try"], str)
        store.close()
