"""Tests for reflection retrieval in context assembly and decomposer prompt injection.

Covers T131: get_relevant_reflections() queries the graph for Memory nodes with
category="reflection" linked to similar failed tasks, and injects them into the
decomposer prompt as a PAST FAILURES section.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from core_gb.types import GraphContext, Reflection
from graph.context import get_relevant_reflections
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _seed_reflection(
    store: GraphStore,
    task_id: str,
    task_description: str,
    reflection: Reflection,
    memory_id: str | None = None,
) -> str:
    """Seed a failed Task node with a linked reflection Memory node.

    Returns the memory node ID.
    """
    now = datetime.now()

    # Create the Task node
    store.create_node("Task", {
        "id": task_id,
        "description": task_description,
        "domain": "synthesis",
        "complexity": 1,
        "status": "failed",
        "tokens_used": 500,
        "latency_ms": 200.0,
        "created_at": now,
    })

    # Create the Memory node with reflection content
    mem_id = memory_id or f"refl_{task_id}"
    content = json.dumps({
        "what_failed": reflection.what_failed,
        "why": reflection.why,
        "what_to_try": reflection.what_to_try,
    })

    store.create_node("Memory", {
        "id": mem_id,
        "content": content,
        "category": "reflection",
        "confidence": 1.0,
        "source_episode": task_id,
        "valid_from": now,
    })

    store.create_edge("REFLECTION_OF", mem_id, task_id)
    return mem_id


class TestGetRelevantReflections:
    """get_relevant_reflections() finds reflections linked to similar failed tasks."""

    def test_returns_empty_list_when_no_reflections_exist(self) -> None:
        """No reflections in the graph yields an empty list."""
        store = _make_store()
        result = get_relevant_reflections(store, "Weather in Amsterdam")
        assert result == []
        store.close()

    def test_finds_reflection_for_similar_task(self) -> None:
        """A reflection linked to a similar task description is returned."""
        store = _make_store()
        _seed_reflection(
            store,
            task_id="task_weather_1",
            task_description="Weather in Amsterdam, London, and Berlin",
            reflection=Reflection(
                what_failed="Weather API call timed out",
                why="External API was rate limited",
                what_to_try="Add retry logic with exponential backoff",
            ),
        )

        result = get_relevant_reflections(store, "Weather in Amsterdam and Paris")
        assert len(result) >= 1

        # The returned dict should contain structured reflection fields
        first = result[0]
        assert "what_failed" in first
        assert "why" in first
        assert "what_to_try" in first
        store.close()

    def test_does_not_return_unrelated_reflections(self) -> None:
        """A reflection for a completely different task is not returned."""
        store = _make_store()
        _seed_reflection(
            store,
            task_id="task_code_1",
            task_description="Run pytest and fix failing tests in auth module",
            reflection=Reflection(
                what_failed="pytest discovered import errors",
                why="Missing dependency in test environment",
                what_to_try="Install missing package before running tests",
            ),
        )

        result = get_relevant_reflections(store, "Weather in Amsterdam")
        assert len(result) == 0
        store.close()

    def test_returns_multiple_relevant_reflections(self) -> None:
        """Multiple reflections for similar tasks are all returned."""
        store = _make_store()
        _seed_reflection(
            store,
            task_id="task_weather_1",
            task_description="Weather in Amsterdam",
            reflection=Reflection(
                what_failed="API timeout",
                why="Rate limited",
                what_to_try="Use retry logic",
            ),
        )
        _seed_reflection(
            store,
            task_id="task_weather_2",
            task_description="Weather forecast for Amsterdam tomorrow",
            reflection=Reflection(
                what_failed="Forecast endpoint returned 503",
                why="Service maintenance window",
                what_to_try="Try alternative weather API",
            ),
        )

        result = get_relevant_reflections(store, "Weather in Amsterdam next week")
        assert len(result) >= 2
        store.close()

    def test_respects_max_results_parameter(self) -> None:
        """max_results limits the number of reflections returned."""
        store = _make_store()
        for i in range(5):
            _seed_reflection(
                store,
                task_id=f"task_weather_{i}",
                task_description=f"Weather in city number {i}",
                reflection=Reflection(
                    what_failed=f"Failure {i}",
                    why=f"Reason {i}",
                    what_to_try=f"Suggestion {i}",
                ),
            )

        result = get_relevant_reflections(
            store, "Weather in city number 3", max_results=2
        )
        assert len(result) <= 2
        store.close()

    def test_similarity_threshold_filters_weak_matches(self) -> None:
        """Reflections below the similarity threshold are excluded."""
        store = _make_store()
        _seed_reflection(
            store,
            task_id="task_1",
            task_description="XYZZY completely unrelated gibberish task",
            reflection=Reflection(
                what_failed="Something",
                why="Something",
                what_to_try="Something",
            ),
        )

        result = get_relevant_reflections(
            store,
            "Weather in Amsterdam",
            similarity_threshold=0.5,
        )
        assert len(result) == 0
        store.close()

    def test_only_reflection_memories_are_queried(self) -> None:
        """Non-reflection Memory nodes are not returned."""
        store = _make_store()
        # Create a non-reflection memory about weather
        store.create_node("Memory", {
            "id": "mem_pref_1",
            "content": "User likes weather data in Celsius",
            "category": "preference",
            "confidence": 1.0,
        })

        result = get_relevant_reflections(store, "Weather in Amsterdam")
        assert len(result) == 0
        store.close()

    def test_returned_dicts_have_task_description(self) -> None:
        """Each returned reflection includes the original task description."""
        store = _make_store()
        _seed_reflection(
            store,
            task_id="task_w1",
            task_description="Weather in Amsterdam and Berlin",
            reflection=Reflection(
                what_failed="API timeout",
                why="Rate limited",
                what_to_try="Use retry logic",
            ),
        )

        result = get_relevant_reflections(store, "Weather in Amsterdam and London")
        assert len(result) >= 1
        assert "task_description" in result[0]
        assert "Amsterdam" in result[0]["task_description"]
        store.close()


class TestGraphContextReflections:
    """GraphContext.format() includes PAST FAILURES section when reflections are present."""

    def test_format_without_reflections_unchanged(self) -> None:
        """format() output is unchanged when no reflections are present."""
        ctx = GraphContext(
            user_summary="Alice | student",
            active_memories=("Prefers Python",),
        )
        formatted = ctx.format()
        assert "PAST FAILURES" not in formatted
        assert "USER: Alice | student" in formatted
        assert "MEMORY: Prefers Python" in formatted

    def test_format_with_reflections_includes_past_failures(self) -> None:
        """format() includes PAST FAILURE entries when reflections are provided."""
        ctx = GraphContext(
            user_summary="Alice | student",
            reflections=(
                {
                    "task_description": "Weather in Amsterdam",
                    "what_failed": "API timeout",
                    "why": "Rate limited",
                    "what_to_try": "Use retry logic",
                },
            ),
        )
        formatted = ctx.format()
        assert "PAST FAILURE" in formatted
        assert "API timeout" in formatted
        assert "Rate limited" in formatted
        assert "retry logic" in formatted

    def test_format_reflection_includes_task_description(self) -> None:
        """Formatted reflection includes the original failed task description."""
        ctx = GraphContext(
            reflections=(
                {
                    "task_description": "Weather in Amsterdam",
                    "what_failed": "API timeout",
                    "why": "Rate limited",
                    "what_to_try": "Use retry logic",
                },
            ),
        )
        formatted = ctx.format()
        assert "Weather in Amsterdam" in formatted

    def test_format_multiple_reflections(self) -> None:
        """Multiple reflections are all included in the PAST FAILURES section."""
        ctx = GraphContext(
            reflections=(
                {
                    "task_description": "Weather in Amsterdam",
                    "what_failed": "API timeout",
                    "why": "Rate limited",
                    "what_to_try": "Use retry logic",
                },
                {
                    "task_description": "Weather in Berlin",
                    "what_failed": "Invalid API key",
                    "why": "Key expired",
                    "what_to_try": "Refresh API key",
                },
            ),
        )
        formatted = ctx.format()
        assert formatted.count("PAST FAILURE") >= 2
        assert "API timeout" in formatted
        assert "Invalid API key" in formatted


class TestDecomposerReflectionInjection:
    """Decomposer includes reflections in the system prompt when available."""

    def test_decomposer_prompt_includes_reflections(self) -> None:
        """DecompositionPrompt.build() injects reflections into context block."""
        from core_gb.decomposer import DecompositionPrompt

        prompt_builder = DecompositionPrompt()
        ctx = GraphContext(
            reflections=(
                {
                    "task_description": "Weather in Amsterdam",
                    "what_failed": "API timeout",
                    "why": "Rate limited",
                    "what_to_try": "Use retry with backoff",
                },
            ),
        )

        messages = prompt_builder.build("Weather in London", context=ctx)
        system_content = messages[0]["content"]

        assert "PAST FAILURE" in system_content
        assert "API timeout" in system_content
        assert "retry with backoff" in system_content

    def test_decomposer_prompt_without_reflections_unchanged(self) -> None:
        """DecompositionPrompt.build() works normally without reflections."""
        from core_gb.decomposer import DecompositionPrompt

        prompt_builder = DecompositionPrompt()
        ctx = GraphContext(user_summary="Alice | student")

        messages = prompt_builder.build("Weather in London", context=ctx)
        system_content = messages[0]["content"]

        assert "PAST FAILURE" not in system_content
        assert "Alice | student" in system_content

    def test_decomposer_prompt_with_none_context(self) -> None:
        """DecompositionPrompt.build() works with None context."""
        from core_gb.decomposer import DecompositionPrompt

        prompt_builder = DecompositionPrompt()
        messages = prompt_builder.build("Weather in London", context=None)
        system_content = messages[0]["content"]

        assert "PAST FAILURE" not in system_content


class TestEndToEndReflectionRetrieval:
    """End-to-end: store a reflection, query for a similar task, verify it appears in context."""

    def test_stored_reflection_appears_in_decomposer_context(self) -> None:
        """Full flow: store reflection -> retrieve -> format in context -> inject in prompt."""
        from core_gb.decomposer import DecompositionPrompt

        store = _make_store()

        # 1. Store a reflection for a failed weather task
        _seed_reflection(
            store,
            task_id="task_weather_fail",
            task_description="Weather in Amsterdam, London, and Berlin",
            reflection=Reflection(
                what_failed="Weather API timed out for Amsterdam",
                why="External API rate limiting during peak hours",
                what_to_try="Add retry logic with exponential backoff",
            ),
        )

        # 2. Retrieve reflections for a similar task
        reflections = get_relevant_reflections(
            store, "Weather in Amsterdam and Paris"
        )
        assert len(reflections) >= 1

        # 3. Build GraphContext with reflections
        ctx = GraphContext(reflections=tuple(reflections))

        # 4. Build decomposer prompt with context
        prompt_builder = DecompositionPrompt()
        messages = prompt_builder.build("Weather in Amsterdam and Paris", context=ctx)
        system_content = messages[0]["content"]

        # 5. Verify reflection appears in prompt
        assert "PAST FAILURE" in system_content
        assert "Weather API timed out" in system_content
        assert "retry logic" in system_content

        store.close()
