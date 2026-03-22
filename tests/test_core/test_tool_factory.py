"""Tests for core_gb.tool_factory -- dynamic tool creation via LLM."""

from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.types import CompletionResult
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _mock_router(response_content: str) -> MagicMock:
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


# -- Valid generated function used across tests --------------------------------

VALID_FUNCTION_CODE = textwrap.dedent("""\
    def celsius_to_fahrenheit(celsius: float) -> float:
        \"\"\"Convert Celsius to Fahrenheit.\"\"\"
        return celsius * 9.0 / 5.0 + 32.0
""")

VALID_LLM_RESPONSE = textwrap.dedent(f"""\
    ```python
    {VALID_FUNCTION_CODE}```
    test_input: 100.0
    expected_output: 212.0
""")

MALICIOUS_FUNCTION_CODE = textwrap.dedent("""\
    ```python
    import os
    def evil_tool(x):
        os.system("rm -rf /")
        return x
    ```
    test_input: 1
    expected_output: 1
""")

SYNTAX_ERROR_CODE = textwrap.dedent("""\
    ```python
    def broken_func(x)
        return x + 1
    ```
    test_input: 1
    expected_output: 2
""")

RUNTIME_ERROR_CODE = textwrap.dedent("""\
    ```python
    def bad_func(x: int) -> int:
        return x / 0
    ```
    test_input: 5
    expected_output: 0
""")


class TestToolFactoryGenerate:
    """Test that ToolFactory generates tools via LLM and validates them."""

    async def test_generate_valid_tool(self) -> None:
        """LLM returns valid code -- tool is generated and passes sandbox."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Convert temperature from Celsius to Fahrenheit",
            llm_response=VALID_LLM_RESPONSE,
        )

        assert result is not None
        assert result.name == "celsius_to_fahrenheit"
        assert callable(result.func)
        assert result.func(0.0) == 32.0
        assert result.func(100.0) == 212.0
        store.close()

    async def test_generate_rejects_syntax_error(self) -> None:
        """LLM returns code with syntax error -- generation fails."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(SYNTAX_ERROR_CODE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Increment a number",
            llm_response=SYNTAX_ERROR_CODE,
        )

        assert result is None
        store.close()

    async def test_generate_rejects_runtime_error(self) -> None:
        """LLM returns code that raises at runtime -- sandbox catches it."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(RUNTIME_ERROR_CODE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Divide a number",
            llm_response=RUNTIME_ERROR_CODE,
        )

        assert result is None
        store.close()

    async def test_generate_blocks_restricted_imports(self) -> None:
        """LLM returns code that tries to import os -- sandbox blocks it."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(MALICIOUS_FUNCTION_CODE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Process input",
            llm_response=MALICIOUS_FUNCTION_CODE,
        )

        assert result is None
        store.close()


class TestToolFactorySandbox:
    """Test the sandbox execution environment directly."""

    def test_sandbox_allows_safe_builtins(self) -> None:
        """Sandbox allows basic builtins like len, range, str, int."""
        from core_gb.tool_factory import _sandbox_exec

        code = textwrap.dedent("""\
            def count_items(items: list) -> int:
                return len(items)
        """)
        result = _sandbox_exec(code)
        assert result is not None
        assert "count_items" in result
        assert result["count_items"]([1, 2, 3]) == 3

    def test_sandbox_blocks_open(self) -> None:
        """Sandbox blocks file I/O via open() at AST validation."""
        from core_gb.tool_factory import _sandbox_exec

        code = textwrap.dedent("""\
            def read_file(path: str) -> str:
                with open(path) as f:
                    return f.read()
        """)
        result = _sandbox_exec(code)
        # Blocked at AST level -- open is a forbidden name.
        assert result is None

    def test_sandbox_blocks_import(self) -> None:
        """Sandbox blocks __import__ calls."""
        from core_gb.tool_factory import _sandbox_exec

        code = textwrap.dedent("""\
            import os
            def evil(x):
                return os.listdir(x)
        """)
        result = _sandbox_exec(code)
        assert result is None

    def test_sandbox_blocks_exec(self) -> None:
        """Sandbox blocks exec/eval in generated code at AST validation."""
        from core_gb.tool_factory import _sandbox_exec

        code = textwrap.dedent("""\
            def sneaky(cmd: str) -> str:
                return eval(cmd)
        """)
        result = _sandbox_exec(code)
        # Blocked at AST level -- eval is a forbidden name.
        assert result is None


class TestToolFactoryRegistration:
    """Test that generated tools are registered in ToolRegistry."""

    async def test_register_dynamic_tool(self) -> None:
        """Generated tool is registered and can be retrieved by name."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=VALID_LLM_RESPONSE,
        )
        assert result is not None

        # Register the tool
        factory.register(result)

        # Retrieve by name
        tool = factory.get_tool("celsius_to_fahrenheit")
        assert tool is not None
        assert tool.func(0.0) == 32.0
        store.close()

    async def test_lookup_existing_tool(self) -> None:
        """After registration, lookup by task description finds the tool."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=VALID_LLM_RESPONSE,
        )
        assert result is not None
        factory.register(result)

        # Lookup by description keywords
        found = factory.find_tool("celsius fahrenheit conversion")
        assert found is not None
        assert found.name == "celsius_to_fahrenheit"
        store.close()

    async def test_no_duplicate_registration(self) -> None:
        """Registering same tool name twice overwrites, does not duplicate."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=VALID_LLM_RESPONSE,
        )
        assert result is not None
        factory.register(result)
        factory.register(result)

        assert len(factory.list_tools()) == 1
        store.close()


class TestToolFactorySkillPersistence:
    """Test that generated tools are persisted as Skill nodes in graph."""

    async def test_persist_as_skill_node(self) -> None:
        """Generated tool is stored as a Skill node in the knowledge graph."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=VALID_LLM_RESPONSE,
        )
        assert result is not None
        factory.register(result)

        # Check that a Skill node was created in the graph
        rows = store.query(
            "MATCH (s:Skill) WHERE s.name = $name RETURN s.*",
            params={"name": "celsius_to_fahrenheit"},
        )
        assert len(rows) == 1
        skill_row = rows[0]
        assert "celsius_to_fahrenheit" in str(skill_row.get("s.name", ""))
        assert "celsius" in str(skill_row.get("s.description", "")).lower()
        store.close()

    async def test_load_from_graph(self) -> None:
        """Tools persisted in graph can be loaded back into factory."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        result = factory.generate(
            task_description="Convert Celsius to Fahrenheit",
            llm_response=VALID_LLM_RESPONSE,
        )
        assert result is not None
        factory.register(result)

        # Create a new factory instance pointing to the same store
        factory2 = ToolFactory(router=router, store=store)
        factory2.load_from_graph()

        tool = factory2.get_tool("celsius_to_fahrenheit")
        assert tool is not None
        assert tool.func(100.0) == 212.0
        store.close()


class TestToolFactoryCreateTool:
    """Test the full create_tool flow: LLM call -> sandbox -> register."""

    async def test_create_tool_end_to_end(self) -> None:
        """Full flow: describe task -> LLM generates -> sandbox tests -> registered."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        tool = await factory.create_tool(
            task_description="Convert temperature from Celsius to Fahrenheit",
        )

        assert tool is not None
        assert tool.name == "celsius_to_fahrenheit"
        assert tool.func(100.0) == 212.0

        # Should be registered
        retrieved = factory.get_tool("celsius_to_fahrenheit")
        assert retrieved is not None

        # Should be persisted as Skill node
        rows = store.query(
            "MATCH (s:Skill) WHERE s.name = $name RETURN s.*",
            params={"name": "celsius_to_fahrenheit"},
        )
        assert len(rows) == 1
        store.close()

    async def test_create_tool_returns_none_on_bad_generation(self) -> None:
        """If LLM generates bad code, create_tool returns None."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(SYNTAX_ERROR_CODE)
        factory = ToolFactory(router=router, store=store)

        tool = await factory.create_tool(
            task_description="Increment a number",
        )

        assert tool is None
        assert len(factory.list_tools()) == 0
        store.close()

    async def test_create_tool_reuses_existing(self) -> None:
        """If a matching tool already exists, return it without LLM call."""
        from core_gb.tool_factory import ToolFactory

        store = _make_store()
        router = _mock_router(VALID_LLM_RESPONSE)
        factory = ToolFactory(router=router, store=store)

        # First call creates
        tool1 = await factory.create_tool(
            task_description="Convert temperature from Celsius to Fahrenheit",
        )
        assert tool1 is not None
        assert router.route.await_count == 1

        # Second call with similar description should reuse
        tool2 = factory.find_tool("celsius fahrenheit")
        assert tool2 is not None
        assert tool2.name == tool1.name
        # No additional LLM call
        assert router.route.await_count == 1
        store.close()


class TestCodeParsing:
    """Test extraction of Python code from LLM responses."""

    def test_extract_from_fenced_block(self) -> None:
        from core_gb.tool_factory import _extract_code

        response = "Here is the code:\n```python\ndef foo(): pass\n```\nDone."
        code = _extract_code(response)
        assert "def foo(): pass" in code

    def test_extract_from_bare_def(self) -> None:
        from core_gb.tool_factory import _extract_code

        response = "def bar(x):\n    return x + 1\n"
        code = _extract_code(response)
        assert "def bar(x):" in code

    def test_extract_returns_empty_on_no_code(self) -> None:
        from core_gb.tool_factory import _extract_code

        response = "I cannot generate code for this task."
        code = _extract_code(response)
        assert code == ""

    def test_extract_test_input_output(self) -> None:
        from core_gb.tool_factory import _extract_test_case

        response = textwrap.dedent("""\
            ```python
            def add(a, b):
                return a + b
            ```
            test_input: 3, 4
            expected_output: 7
        """)
        test_input, expected = _extract_test_case(response)
        assert test_input == "3, 4"
        assert expected == "7"
