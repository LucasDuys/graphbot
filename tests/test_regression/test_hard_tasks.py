"""Regression tests for the 10 hard task definitions from stress_test.py.

Runs each hard task through a mocked Orchestrator and verifies it either
completes successfully or produces a diagnosed failure. No real API calls.
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
from scripts.stress_test import (
    STRESS_TASKS,
    StressTask,
    diagnose_failure,
)


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


def _make_orchestrator(responses: list[CompletionResult]) -> tuple[Orchestrator, SequentialMockProvider]:
    """Create an Orchestrator backed by an in-memory store and mock provider."""
    store = GraphStore(db_path=None)
    store.initialize()
    provider = SequentialMockProvider(responses)
    router = ModelRouter(provider=provider)
    orchestrator = Orchestrator(store, router)
    return orchestrator, provider


# ---------------------------------------------------------------------------
# Verify task definitions are importable and well-formed
# ---------------------------------------------------------------------------


class TestStressTaskDefinitions:
    """The 10 hard task definitions are importable and well-formed."""

    def test_correct_count(self) -> None:
        assert len(STRESS_TASKS) == 10

    def test_all_have_unique_ids(self) -> None:
        ids = [t.id for t in STRESS_TASKS]
        assert len(ids) == len(set(ids))

    def test_all_fields_populated(self) -> None:
        for task in STRESS_TASKS:
            assert task.id, f"Task missing id: {task}"
            assert task.name, f"Task missing name: {task.id}"
            assert task.category, f"Task missing category: {task.id}"
            assert task.description, f"Task missing description: {task.id}"
            assert 1 <= task.difficulty <= 5, f"Invalid difficulty for {task.id}"
            assert task.expected_behavior, f"Task missing expected_behavior: {task.id}"
            assert task.why_hard, f"Task missing why_hard: {task.id}"


# ---------------------------------------------------------------------------
# Individual hard task regression tests (mocked orchestrator)
# ---------------------------------------------------------------------------


class TestHardTaskExecution:
    """Each hard task runs through the mocked orchestrator and either succeeds
    or produces a diagnosed failure (not an unhandled crash)."""

    async def test_stress_01_multi_hop(self) -> None:
        """Multi-hop reasoning: simple enough for direct execution."""
        orchestrator, provider = _make_orchestrator([
            _completion("The answer is 16."),
        ])
        result = await orchestrator.process(STRESS_TASKS[0].description)
        assert isinstance(result, ExecutionResult)
        assert result.success is True or len(result.errors) > 0

    async def test_stress_02_ambiguous(self) -> None:
        """Ambiguous instruction: should handle gracefully."""
        orchestrator, _ = _make_orchestrator([
            _completion("The instruction 'Make it better' is too vague. Please provide more context."),
        ])
        result = await orchestrator.process(STRESS_TASKS[1].description)
        assert isinstance(result, ExecutionResult)
        # Either succeeds with clarification request or fails with diagnosis
        assert result.output or result.errors

    async def test_stress_03_tool_chain(self) -> None:
        """3+ tool chain: decomposes into multiple subtasks."""
        tree_json = json.dumps({
            "nodes": [
                {
                    "id": "root",
                    "description": "Find Python version, search release date, save to file",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 3,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["get_version", "search_date", "write_file"],
                },
                {
                    "id": "get_version",
                    "description": "Get the Python version installed",
                    "domain": "system",
                    "task_type": "RETRIEVE",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["python_version"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "search_date",
                    "description": "Search for the release date of this Python version",
                    "domain": "web",
                    "task_type": "RETRIEVE",
                    "complexity": 1,
                    "depends_on": ["get_version"],
                    "provides": ["release_date"],
                    "consumes": ["python_version"],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "write_file",
                    "description": "Save version and release date to python_version_info.txt",
                    "domain": "file",
                    "task_type": "ACT",
                    "complexity": 1,
                    "depends_on": ["search_date"],
                    "provides": ["saved_file"],
                    "consumes": ["python_version", "release_date"],
                    "is_atomic": True,
                    "children": [],
                },
            ]
        })
        orchestrator, provider = _make_orchestrator([
            _completion(tree_json, tokens=50),
            _completion("Python 3.11.5"),
            _completion("Released October 2023"),
            _completion("Saved to python_version_info.txt"),
            _completion("Python 3.11.5 was released October 2023."),
        ])
        result = await orchestrator.process(STRESS_TASKS[2].description)
        assert isinstance(result, ExecutionResult)
        # Should have attempted decomposition (multiple provider calls)
        assert len(provider.call_log) >= 2

    async def test_stress_04_dynamic_tool(self) -> None:
        """Dynamic tool needed: SHA256 hash calculation."""
        orchestrator, _ = _make_orchestrator([
            _completion("The SHA256 hash of 'graphbot' is e3b7a0f3c29e..."),
        ])
        result = await orchestrator.process(STRESS_TASKS[3].description)
        assert isinstance(result, ExecutionResult)
        assert result.output or result.errors

    async def test_stress_05_deep_context(self) -> None:
        """Deep graph context: requires querying knowledge graph."""
        orchestrator, _ = _make_orchestrator([
            _completion("Based on available data, the most used tools are: file, web, shell."),
        ])
        result = await orchestrator.process(STRESS_TASKS[4].description)
        assert isinstance(result, ExecutionResult)

    async def test_stress_06_contradictory(self) -> None:
        """Contradictory instruction: should detect contradiction."""
        orchestrator, _ = _make_orchestrator([
            _completion("This request contains a contradiction: 5 words vs 500 words."),
        ])
        result = await orchestrator.process(STRESS_TASKS[5].description)
        assert isinstance(result, ExecutionResult)
        assert result.output or result.errors

    async def test_stress_07_multi_language(self) -> None:
        """Multi-language translation: parallel decomposition."""
        orchestrator, _ = _make_orchestrator([
            _completion("hello in 10 languages: Bonjour, Hola, Hallo, Ciao, ..."),
        ])
        result = await orchestrator.process(STRESS_TASKS[6].description)
        assert isinstance(result, ExecutionResult)

    async def test_stress_08_recursive_decomp(self) -> None:
        """Recursive decomposition: deep tree with 25+ leaves."""
        tree_json = json.dumps({
            "nodes": [
                {
                    "id": "root",
                    "description": "Compare 5 languages across 5 dimensions",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 5,
                    "depends_on": [],
                    "provides": [],
                    "consumes": [],
                    "is_atomic": False,
                    "children": ["py", "rs", "agg"],
                },
                {
                    "id": "py",
                    "description": "Analyze Python across 5 dimensions",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["python_analysis"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "rs",
                    "description": "Analyze Rust across 5 dimensions",
                    "domain": "synthesis",
                    "task_type": "THINK",
                    "complexity": 1,
                    "depends_on": [],
                    "provides": ["rust_analysis"],
                    "consumes": [],
                    "is_atomic": True,
                    "children": [],
                },
                {
                    "id": "agg",
                    "description": "Aggregate comparison",
                    "domain": "synthesis",
                    "task_type": "WRITE",
                    "complexity": 1,
                    "depends_on": ["py", "rs"],
                    "provides": ["comparison"],
                    "consumes": ["python_analysis", "rust_analysis"],
                    "is_atomic": True,
                    "children": [],
                },
            ]
        })
        orchestrator, provider = _make_orchestrator([
            _completion(tree_json, tokens=50),
            _completion("Python: dynamic, slow, huge ecosystem"),
            _completion("Rust: static, fast, safe"),
            _completion("Comparison summary"),
            _completion("Synthesized comparison of languages."),
        ])
        result = await orchestrator.process(STRESS_TASKS[7].description)
        assert isinstance(result, ExecutionResult)
        # Intake may classify this as simple (single LLM call) or complex
        # (decomposition). Either path is valid for regression testing.
        assert len(provider.call_log) >= 1

    async def test_stress_09_time_sensitive(self) -> None:
        """Time-sensitive query: current time in Tokyo."""
        orchestrator, _ = _make_orchestrator([
            _completion("The current time in Tokyo is 15:30 JST."),
        ])
        result = await orchestrator.process(STRESS_TASKS[8].description)
        assert isinstance(result, ExecutionResult)

    async def test_stress_10_meta_reasoning(self) -> None:
        """Meta-reasoning: explain approach then solve."""
        orchestrator, _ = _make_orchestrator([
            _completion("Step 1: multiply 47 by 83. 47 * 83 = 3901. The answer is 3901."),
        ])
        result = await orchestrator.process(STRESS_TASKS[9].description)
        assert isinstance(result, ExecutionResult)
        assert result.success is True


# ---------------------------------------------------------------------------
# Failure diagnosis regression
# ---------------------------------------------------------------------------


class TestFailureDiagnosis:
    """diagnose_failure() correctly categorizes failures for each hard task."""

    def test_timeout_diagnosis(self) -> None:
        task = STRESS_TASKS[0]
        diag = diagnose_failure(task, None, None, elapsed_ms=70_000)
        assert diag.category == "timeout"

    def test_runtime_error_diagnosis(self) -> None:
        task = STRESS_TASKS[0]
        diag = diagnose_failure(task, None, RuntimeError("boom"), elapsed_ms=100)
        assert diag.category == "runtime_error"
        assert "boom" in diag.root_cause

    def test_tool_error_diagnosis(self) -> None:
        task = STRESS_TASKS[0]
        diag = diagnose_failure(task, None, RuntimeError("tool registry not found"), elapsed_ms=100)
        assert diag.category == "tool_failure"
