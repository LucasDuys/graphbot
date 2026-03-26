"""Regression tests for the 30-task capability suite from run_capability_tests.py.

Tests are grouped by category. Simple tasks should succeed, safety tasks should
be blocked, and complex tasks should trigger decomposition. All LLM providers
are mocked -- no real API calls.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core_gb.orchestrator import Orchestrator
from core_gb.types import CompletionResult, ExecutionResult
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter
from scripts.run_capability_tests import TASKS


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class SequentialMockProvider(ModelProvider):
    """Mock provider returning pre-configured responses in order."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = responses
        self._call_count = 0
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


def _completion(content: str, tokens: int = 10, cost: float = 0.0) -> CompletionResult:
    """Build a minimal CompletionResult for mocking."""
    return CompletionResult(
        content=content,
        model="mock-model",
        tokens_in=tokens,
        tokens_out=tokens,
        latency_ms=10.0,
        cost=cost,
    )


def _safety_refusal() -> CompletionResult:
    """A mock response that refuses harmful requests."""
    return _completion("I'm sorry, I cannot help with that request.")


def _make_orchestrator(
    responses: list[CompletionResult],
    force_decompose: bool = False,
) -> tuple[Orchestrator, SequentialMockProvider]:
    """Create an Orchestrator with in-memory store and mock provider."""
    store = GraphStore(db_path=None)
    store.initialize()
    provider = SequentialMockProvider(responses)
    router = ModelRouter(provider=provider)
    orchestrator = Orchestrator(store, router, force_decompose=force_decompose)
    return orchestrator, provider


def _tasks_by_category(category: str) -> list[tuple[str, str]]:
    """Filter the 30 tasks by category prefix."""
    return [(cat, task) for cat, task in TASKS if cat == category]


# ---------------------------------------------------------------------------
# Verify task definitions
# ---------------------------------------------------------------------------


class TestCapabilityTaskDefinitions:
    """The 30 capability tasks are importable and well-formed."""

    def test_correct_count(self) -> None:
        assert len(TASKS) == 30

    def test_all_have_category_and_task(self) -> None:
        for i, (category, task) in enumerate(TASKS):
            assert category, f"Task {i} missing category"
            assert task, f"Task {i} missing task text"

    def test_expected_categories_present(self) -> None:
        categories = {cat for cat, _ in TASKS}
        expected = {
            "Simple Q&A", "Decomposition", "Reasoning",
            "Tool:File", "Tool:Shell", "Tool:Web",
            "Knowledge", "Creative", "Safety:Block",
            "Analysis", "Translation", "Summarization",
            "Cache:Hit", "Code",
        }
        assert expected.issubset(categories), (
            f"Missing categories: {expected - categories}"
        )


# ---------------------------------------------------------------------------
# Simple Q&A tasks (should succeed with direct execution)
# ---------------------------------------------------------------------------


class TestSimpleQA:
    """Simple Q&A tasks complete successfully via direct LLM execution."""

    async def test_simple_qa_tasks_succeed(self) -> None:
        tasks = _tasks_by_category("Simple Q&A")
        assert len(tasks) == 5
        for _, task_text in tasks:
            orchestrator, provider = _make_orchestrator([
                _completion(f"Answer to: {task_text[:40]}"),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            assert result.success is True, (
                f"Simple Q&A failed for: {task_text[:60]}"
            )


# ---------------------------------------------------------------------------
# Knowledge tasks (should succeed with direct execution)
# ---------------------------------------------------------------------------


class TestKnowledge:
    """Knowledge/factual tasks complete successfully."""

    async def test_knowledge_tasks_succeed(self) -> None:
        tasks = _tasks_by_category("Knowledge")
        assert len(tasks) == 3
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion(f"The answer is: relevant knowledge about {task_text[:30]}"),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            assert result.success is True


# ---------------------------------------------------------------------------
# Reasoning tasks (should succeed)
# ---------------------------------------------------------------------------


class TestReasoning:
    """Multi-step reasoning tasks complete successfully."""

    async def test_reasoning_tasks_succeed(self) -> None:
        tasks = _tasks_by_category("Reasoning")
        assert len(tasks) == 2
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion("The calculated answer is 42."),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            assert result.success is True


# ---------------------------------------------------------------------------
# Creative tasks (should succeed)
# ---------------------------------------------------------------------------


class TestCreative:
    """Creative/generation tasks complete successfully."""

    async def test_creative_tasks_succeed(self) -> None:
        tasks = _tasks_by_category("Creative")
        assert len(tasks) == 2
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion("A creative response to the prompt."),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            assert result.success is True


# ---------------------------------------------------------------------------
# Safety:Block tasks (should be blocked by safety layers)
# ---------------------------------------------------------------------------


class TestSafetyBlock:
    """Safety:Block tasks are blocked by pre-decomposition safety layers."""

    async def test_safety_tasks_blocked(self) -> None:
        tasks = _tasks_by_category("Safety:Block")
        assert len(tasks) == 3
        for _, task_text in tasks:
            orchestrator, provider = _make_orchestrator([
                _safety_refusal(),
                _safety_refusal(),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            # Safety tasks must be blocked -- either pre-decomposition
            # (zero LLM calls) or post-decomposition (constitutional/DAG check).
            assert result.success is False, (
                f"Safety task was NOT blocked: {task_text[:60]}"
            )


# ---------------------------------------------------------------------------
# Decomposition tasks (should trigger multi-node execution)
# ---------------------------------------------------------------------------


class TestDecomposition:
    """Complex decomposition tasks trigger multi-node DAG execution."""

    async def test_decomposition_tasks_process(self) -> None:
        tasks = _tasks_by_category("Decomposition")
        assert len(tasks) == 3
        tree_json = json.dumps({
            "nodes": [
                {
                    "id": "root",
                    "description": "Compare items",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 3,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["part_a", "part_b", "agg"],
                },
                {
                    "id": "part_a",
                    "description": "First part",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["part_a_result"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "part_b",
                    "description": "Second part",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["part_b_result"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "agg",
                    "description": "Aggregate",
                    "domain": "synthesis",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": ["part_a", "part_b"],
                    "provides": ["summary"],
                    "consumes": ["part_a_result", "part_b_result"],
                    "is_atomic": True,
                    "children": [],
                },
            ]
        })
        for _, task_text in tasks:
            orchestrator, provider = _make_orchestrator([
                _completion(tree_json, tokens=50),
                _completion("Part A analysis"),
                _completion("Part B analysis"),
                _completion("Aggregated result"),
                _completion("Synthesized comparison."),
            ], force_decompose=True)
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            # Should have called provider multiple times (decompose + execute)
            assert len(provider.call_log) >= 2, (
                f"Expected decomposition for: {task_text[:60]}"
            )


# ---------------------------------------------------------------------------
# Tool tasks (should attempt tool use or decompose)
# ---------------------------------------------------------------------------


class TestToolTasks:
    """Tool-use tasks process through the orchestrator (tools are mocked)."""

    async def test_file_tool_tasks(self) -> None:
        tasks = _tasks_by_category("Tool:File")
        assert len(tasks) == 2
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion("File contents: example.py, test.py"),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)

    async def test_shell_tool_tasks(self) -> None:
        tasks = _tasks_by_category("Tool:Shell")
        assert len(tasks) == 2
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion("Python 3.11.5"),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)

    async def test_web_tool_tasks(self) -> None:
        tasks = _tasks_by_category("Tool:Web")
        assert len(tasks) == 1
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion("Kuzu is a high-performance embedded graph database."),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)


# ---------------------------------------------------------------------------
# Analysis, Translation, Summarization, Code, Cache
# ---------------------------------------------------------------------------


class TestMiscCategories:
    """Remaining categories: analysis, translation, summarization, code, cache."""

    async def test_analysis_tasks_succeed(self) -> None:
        tasks = _tasks_by_category("Analysis")
        assert len(tasks) == 2
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion("Classification: positive and negative."),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            assert result.success is True

    async def test_translation_task_succeeds(self) -> None:
        tasks = _tasks_by_category("Translation")
        assert len(tasks) == 1
        orchestrator, _ = _make_orchestrator([
            _completion("Bonjour, Buenos dias, Guten Morgen"),
        ])
        result = await orchestrator.process(tasks[0][1])
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    async def test_summarization_task_succeeds(self) -> None:
        tasks = _tasks_by_category("Summarization")
        assert len(tasks) == 1
        orchestrator, _ = _make_orchestrator([
            _completion("ML enables systems to learn from data automatically."),
        ])
        result = await orchestrator.process(tasks[0][1])
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    async def test_code_tasks_succeed(self) -> None:
        tasks = _tasks_by_category("Code")
        assert len(tasks) == 2
        for _, task_text in tasks:
            orchestrator, _ = _make_orchestrator([
                _completion("def is_prime(n): return n > 1 and all(n % i for i in range(2, n))"),
            ])
            result = await orchestrator.process(task_text)
            assert isinstance(result, ExecutionResult)
            assert result.success is True

    async def test_cache_hit_task(self) -> None:
        """Cache:Hit is a repeat of 'What is the capital of France?' --
        should succeed regardless of whether cache is populated."""
        tasks = _tasks_by_category("Cache:Hit")
        assert len(tasks) == 1
        orchestrator, _ = _make_orchestrator([
            _completion("Paris"),
        ])
        result = await orchestrator.process(tasks[0][1])
        assert isinstance(result, ExecutionResult)
        assert result.success is True
