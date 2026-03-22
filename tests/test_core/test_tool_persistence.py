"""Tests for generated tool persistence and reload from knowledge graph.

Covers T157 acceptance criteria:
1. ToolFactory.load_from_graph() works on startup.
2. Orchestrator init loads persisted tools from graph into ToolRegistry.
3. Persisted tools available immediately without re-generation.
4. Create tool -> persist -> restart (new ToolFactory instance) -> tools reloaded.
"""

from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.tool_factory import GeneratedTool, ToolFactory, _sandbox_exec
from core_gb.types import CompletionResult
from graph.store import GraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _mock_router(response_content: str = "") -> MagicMock:
    """Return a mock ModelRouter whose route() returns the given content."""
    mock = MagicMock()
    mock.route = AsyncMock(return_value=CompletionResult(
        content=response_content,
        model="mock-model",
        tokens_in=20,
        tokens_out=50,
        latency_ms=100.0,
        cost=0.0,
    ))
    return mock


CELSIUS_CODE = textwrap.dedent("""\
    def celsius_to_fahrenheit(celsius: float) -> float:
        \"\"\"Convert Celsius to Fahrenheit.\"\"\"
        return celsius * 9.0 / 5.0 + 32.0
""")

CELSIUS_LLM_RESPONSE = textwrap.dedent(f"""\
    ```python
    {CELSIUS_CODE}```
    test_input: 100.0
    expected_output: 212.0
""")

FIBONACCI_CODE = textwrap.dedent("""\
    def fibonacci(n: int) -> int:
        \"\"\"Return the n-th Fibonacci number (0-indexed).\"\"\"
        if n <= 0:
            return 0
        if n == 1:
            return 1
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
""")

FIBONACCI_LLM_RESPONSE = textwrap.dedent(f"""\
    ```python
    {FIBONACCI_CODE}```
    test_input: 10
    expected_output: 55
""")


# ---------------------------------------------------------------------------
# Test: load_from_graph works on startup (basic round-trip)
# ---------------------------------------------------------------------------


class TestLoadFromGraphOnStartup:
    """Verify that load_from_graph() restores previously persisted tools."""

    def test_load_from_graph_restores_single_tool(self) -> None:
        """Persist one tool, create new factory, load_from_graph restores it."""
        store = _make_store()
        router = _mock_router()
        factory1 = ToolFactory(router=router, store=store)

        # Generate and register a tool (persists to graph).
        tool = factory1.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=CELSIUS_LLM_RESPONSE,
        )
        assert tool is not None
        factory1.register(tool)

        # Verify the Skill node exists in the graph.
        rows = store.query(
            "MATCH (s:Skill) WHERE s.name = $name RETURN s.name",
            params={"name": "celsius_to_fahrenheit"},
        )
        assert len(rows) == 1

        # Simulate restart: new factory instance, same store.
        factory2 = ToolFactory(router=router, store=store)
        assert factory2.list_tools() == []  # Empty before load.

        loaded_count = factory2.load_from_graph()
        assert loaded_count == 1

        restored = factory2.get_tool("celsius_to_fahrenheit")
        assert restored is not None
        assert restored.name == "celsius_to_fahrenheit"
        assert callable(restored.func)
        assert restored.func(100.0) == 212.0
        assert restored.func(0.0) == 32.0
        store.close()

    def test_load_from_graph_restores_multiple_tools(self) -> None:
        """Persist two tools, new factory loads both."""
        store = _make_store()
        router = _mock_router()
        factory1 = ToolFactory(router=router, store=store)

        tool_c = factory1.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=CELSIUS_LLM_RESPONSE,
        )
        assert tool_c is not None
        factory1.register(tool_c)

        tool_f = factory1.generate(
            task_description="Compute Fibonacci number",
            llm_response=FIBONACCI_LLM_RESPONSE,
        )
        assert tool_f is not None
        factory1.register(tool_f)

        # New factory, same store.
        factory2 = ToolFactory(router=router, store=store)
        loaded_count = factory2.load_from_graph()
        assert loaded_count == 2

        assert factory2.get_tool("celsius_to_fahrenheit") is not None
        assert factory2.get_tool("fibonacci") is not None
        assert factory2.get_tool("fibonacci").func(10) == 55
        store.close()

    def test_load_from_graph_returns_zero_when_no_store(self) -> None:
        """Without a store, load_from_graph returns 0."""
        router = _mock_router()
        factory = ToolFactory(router=router, store=None)
        assert factory.load_from_graph() == 0

    def test_load_from_graph_skips_invalid_skill(self) -> None:
        """If a Skill node has invalid source_code, it is skipped."""
        store = _make_store()
        router = _mock_router()

        # Manually insert a Skill node with broken code.
        store.create_node("Skill", {
            "id": "broken-skill-001",
            "name": "broken_func",
            "description": "A broken function",
            "path": "def broken_func(x)\n    return x + 1",  # SyntaxError
        })

        factory = ToolFactory(router=router, store=store)
        loaded_count = factory.load_from_graph()
        assert loaded_count == 0
        assert factory.get_tool("broken_func") is None
        store.close()

    def test_load_from_graph_loads_valid_skips_invalid(self) -> None:
        """Mixed scenario: one valid Skill node, one invalid. Only valid loads."""
        store = _make_store()
        router = _mock_router()

        # Insert a valid Skill node manually.
        store.create_node("Skill", {
            "id": "valid-skill-001",
            "name": "double_it",
            "description": "Double a number",
            "path": "def double_it(x: int) -> int:\n    return x * 2\n",
        })

        # Insert an invalid Skill node.
        store.create_node("Skill", {
            "id": "broken-skill-002",
            "name": "broken_func",
            "description": "Broken function",
            "path": "import os\ndef broken_func(): pass",
        })

        factory = ToolFactory(router=router, store=store)
        loaded_count = factory.load_from_graph()
        assert loaded_count == 1
        assert factory.get_tool("double_it") is not None
        assert factory.get_tool("double_it").func(7) == 14
        assert factory.get_tool("broken_func") is None
        store.close()


# ---------------------------------------------------------------------------
# Test: Persisted tools available immediately without re-generation
# ---------------------------------------------------------------------------


class TestPersistedToolsAvailableWithoutRegeneration:
    """Loaded tools should be callable without any LLM call."""

    def test_loaded_tool_callable_without_llm(self) -> None:
        """After load_from_graph, tools work without any router/LLM call."""
        store = _make_store()
        router = _mock_router()
        factory1 = ToolFactory(router=router, store=store)

        tool = factory1.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=CELSIUS_LLM_RESPONSE,
        )
        assert tool is not None
        factory1.register(tool)

        # New factory -- router.route should NOT be called during load.
        router2 = _mock_router()
        factory2 = ToolFactory(router=router2, store=store)
        factory2.load_from_graph()

        restored = factory2.get_tool("celsius_to_fahrenheit")
        assert restored is not None
        assert restored.func(37.0) == pytest.approx(98.6)

        # Confirm no LLM call was made.
        router2.route.assert_not_awaited()
        store.close()

    def test_loaded_tool_found_by_keyword_search(self) -> None:
        """find_tool works on loaded tools (keyword match)."""
        store = _make_store()
        router = _mock_router()
        factory1 = ToolFactory(router=router, store=store)

        tool = factory1.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=CELSIUS_LLM_RESPONSE,
        )
        assert tool is not None
        factory1.register(tool)

        factory2 = ToolFactory(router=router, store=store)
        factory2.load_from_graph()

        found = factory2.find_tool("celsius fahrenheit conversion")
        assert found is not None
        assert found.name == "celsius_to_fahrenheit"
        store.close()


# ---------------------------------------------------------------------------
# Test: Orchestrator init loads persisted tools from graph
# ---------------------------------------------------------------------------


class TestOrchestratorLoadsToolsOnInit:
    """Orchestrator.__init__ should create ToolFactory and load from graph."""

    def test_orchestrator_has_tool_factory(self) -> None:
        """Orchestrator creates a ToolFactory attribute on init."""
        store = _make_store()
        router = _mock_router()

        from core_gb.orchestrator import Orchestrator
        orchestrator = Orchestrator(store=store, router=router)

        assert hasattr(orchestrator, "_tool_factory")
        assert isinstance(orchestrator._tool_factory, ToolFactory)
        store.close()

    def test_orchestrator_loads_persisted_tools(self) -> None:
        """Orchestrator loads tools from graph during init."""
        store = _make_store()
        router = _mock_router()

        # Pre-populate graph with a Skill node.
        store.create_node("Skill", {
            "id": "pre-loaded-skill-001",
            "name": "triple_it",
            "description": "Triple a number",
            "path": "def triple_it(x: int) -> int:\n    return x * 3\n",
        })

        from core_gb.orchestrator import Orchestrator
        orchestrator = Orchestrator(store=store, router=router)

        # The tool factory should have the pre-loaded tool.
        loaded_tool = orchestrator._tool_factory.get_tool("triple_it")
        assert loaded_tool is not None
        assert loaded_tool.func(5) == 15
        store.close()

    def test_orchestrator_tool_factory_connected_to_registry(self) -> None:
        """ToolRegistry can look up generated tools via the factory."""
        store = _make_store()
        router = _mock_router()

        # Pre-populate graph with a Skill node.
        store.create_node("Skill", {
            "id": "pre-loaded-skill-002",
            "name": "square_it",
            "description": "Square a number",
            "path": "def square_it(x: int) -> int:\n    return x * x\n",
        })

        from core_gb.orchestrator import Orchestrator
        orchestrator = Orchestrator(store=store, router=router)

        # Verify the tool is accessible through the registry's generated tools.
        assert orchestrator._tool_registry.has_generated_tool("square_it")
        store.close()


# ---------------------------------------------------------------------------
# Test: Full persistence round-trip (create -> persist -> restart -> reload)
# ---------------------------------------------------------------------------


class TestFullPersistenceRoundTrip:
    """End-to-end: create_tool -> persist -> new instances -> reload -> execute."""

    async def test_create_persist_restart_reload(self) -> None:
        """Full round-trip: async create_tool, persist, new factory loads it."""
        store = _make_store()
        router = _mock_router(CELSIUS_LLM_RESPONSE)
        factory1 = ToolFactory(router=router, store=store)

        # Create tool via full pipeline (LLM call -> sandbox -> register -> persist).
        tool = await factory1.create_tool(
            task_description="Convert temperature from Celsius to Fahrenheit",
        )
        assert tool is not None
        assert tool.name == "celsius_to_fahrenheit"
        assert router.route.await_count == 1

        # Simulate restart: new factory + new router (no pre-loaded response needed).
        router2 = _mock_router()
        factory2 = ToolFactory(router=router2, store=store)
        loaded = factory2.load_from_graph()
        assert loaded == 1

        # Tool is available and functional.
        restored = factory2.get_tool("celsius_to_fahrenheit")
        assert restored is not None
        assert restored.func(100.0) == 212.0
        assert restored.func(0.0) == 32.0
        assert restored.source_code.strip() != ""

        # No LLM call needed.
        router2.route.assert_not_awaited()
        store.close()

    async def test_create_tool_reuses_loaded_from_graph(self) -> None:
        """After load_from_graph, create_tool finds the existing tool."""
        store = _make_store()
        router = _mock_router(CELSIUS_LLM_RESPONSE)
        factory1 = ToolFactory(router=router, store=store)

        # Create and persist.
        tool = await factory1.create_tool(
            task_description="Convert temperature from Celsius to Fahrenheit",
        )
        assert tool is not None

        # New factory, load from graph.
        router2 = _mock_router()
        factory2 = ToolFactory(router=router2, store=store)
        factory2.load_from_graph()

        # create_tool should find the existing tool without LLM call.
        existing = await factory2.create_tool(
            task_description="celsius fahrenheit conversion",
        )
        assert existing is not None
        assert existing.name == "celsius_to_fahrenheit"
        router2.route.assert_not_awaited()
        store.close()

    def test_tool_metadata_preserved_across_restart(self) -> None:
        """Tool description and source_code survive persistence round-trip."""
        store = _make_store()
        router = _mock_router()
        factory1 = ToolFactory(router=router, store=store)

        tool = factory1.generate(
            task_description="Compute Fibonacci number",
            llm_response=FIBONACCI_LLM_RESPONSE,
        )
        assert tool is not None
        factory1.register(tool)

        # New factory, reload.
        factory2 = ToolFactory(router=router, store=store)
        factory2.load_from_graph()

        restored = factory2.get_tool("fibonacci")
        assert restored is not None
        assert "fibonacci" in restored.description.lower() or "fibonacci" in restored.name.lower()
        assert "def fibonacci" in restored.source_code
        assert restored.task_description != ""
        store.close()
