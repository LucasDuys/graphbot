"""Dynamic tool creation via LLM with sandbox validation.

When no existing tool matches a task, ToolFactory generates a Python function
via an LLM call, validates it in a restricted sandbox, registers it in the
in-memory tool registry, and persists it as a Skill node in the knowledge graph
for future reuse.

Informed by LATM (Large Language Models as Tool Makers) and CREATOR patterns.
"""

from __future__ import annotations

import ast
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus

logger = logging.getLogger(__name__)

# Builtins allowed in the sandbox -- no file I/O, no imports, no exec/eval.
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

# AST node types that are forbidden in generated code.
_FORBIDDEN_AST_NODES: set[type] = {
    ast.Import,
    ast.ImportFrom,
}

# Names that must not appear in the generated code (attribute access).
_FORBIDDEN_NAMES: set[str] = {
    "__import__",
    "__builtins__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__class__",
    "exec",
    "eval",
    "compile",
    "open",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "input",
    "memoryview",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "object",
}

# Prompt template for tool generation.
_TOOL_GEN_PROMPT = """\
You are a tool-making assistant. Given a task description, generate a single \
Python function that accomplishes the task.

Requirements:
- The function must be self-contained (no imports, no external dependencies).
- Use only basic Python builtins (len, range, str, int, float, list, dict, etc.).
- Include a clear docstring explaining what the function does.
- Include full type hints on parameters and return type.
- The function name should be descriptive and use snake_case.

After the code block, provide a test case:
test_input: <value(s) to pass to the function>
expected_output: <the expected return value>

Task: {task_description}

Respond with ONLY the function in a ```python code block, followed by the test case.
"""


@dataclass
class GeneratedTool:
    """A dynamically generated tool function with metadata."""

    name: str
    description: str
    func: Callable[..., Any]
    source_code: str
    task_description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _extract_code(response: str) -> str:
    """Extract Python code from an LLM response.

    Looks for fenced code blocks first (```python ... ```), then falls back
    to detecting bare function definitions.

    Returns empty string if no code is found.
    """
    # Try fenced code block first.
    fenced = re.search(
        r"```(?:python)?\s*\n(.*?)```",
        response,
        re.DOTALL,
    )
    if fenced:
        return fenced.group(1).strip()

    # Fall back to bare def.
    lines = response.strip().splitlines()
    code_lines: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith("def "):
            capturing = True
        if capturing:
            # Stop at non-indented, non-empty line after function body.
            if code_lines and line and not line[0].isspace() and not line.startswith("def "):
                break
            code_lines.append(line)

    return "\n".join(code_lines).strip()


def _extract_test_case(response: str) -> tuple[str, str]:
    """Extract test_input and expected_output from an LLM response.

    Returns (test_input, expected_output) as raw strings.
    Returns ("", "") if no test case is found.
    """
    test_input = ""
    expected_output = ""

    input_match = re.search(r"test_input:\s*(.+)", response, re.IGNORECASE)
    if input_match:
        test_input = input_match.group(1).strip()

    output_match = re.search(r"expected_output:\s*(.+)", response, re.IGNORECASE)
    if output_match:
        expected_output = output_match.group(1).strip()

    return test_input, expected_output


def _validate_ast(code: str) -> bool:
    """Validate that code does not contain forbidden AST nodes or names.

    Returns True if the code is safe, False otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        # Block import statements.
        if type(node) in _FORBIDDEN_AST_NODES:
            logger.warning("Sandbox: blocked forbidden AST node %s", type(node).__name__)
            return False

        # Block forbidden name references.
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            logger.warning("Sandbox: blocked forbidden name '%s'", node.id)
            return False

        # Block forbidden attribute access (e.g. obj.__class__).
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            logger.warning("Sandbox: blocked forbidden attribute '%s'", node.attr)
            return False

        # Block forbidden string references in calls (e.g. getattr(x, '__class__')).
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _FORBIDDEN_NAMES:
                logger.warning("Sandbox: blocked forbidden string constant '%s'", node.value)
                return False

    return True


def _sandbox_exec(code: str) -> dict[str, Any] | None:
    """Execute code in a restricted sandbox and return the namespace.

    Returns a dict of defined names (functions, variables) on success.
    Returns None if the code fails AST validation or execution.
    """
    if not _validate_ast(code):
        return None

    restricted_globals: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
    namespace: dict[str, Any] = {}

    try:
        compiled = compile(code, "<generated_tool>", "exec")
        exec(compiled, restricted_globals, namespace)  # noqa: S102
    except Exception as exc:
        logger.warning("Sandbox execution failed: %s", exc)
        return None

    return namespace


def _extract_function_name(code: str) -> str:
    """Extract the first function name from a code string."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name

    return ""


def _extract_docstring(code: str) -> str:
    """Extract the docstring from the first function definition."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            return docstring or ""

    return ""


class ToolFactory:
    """Generates, validates, and registers dynamic tools via LLM.

    Workflow:
    1. Receive a task description for which no existing tool matches.
    2. Call LLM to generate a Python function.
    3. Validate the generated code in a restricted sandbox.
    4. Test the function with provided test cases.
    5. Register the function for future use.
    6. Persist as a Skill node in the knowledge graph.
    """

    def __init__(
        self,
        router: Any,
        store: Any | None = None,
    ) -> None:
        self._router = router
        self._store = store
        self._tools: dict[str, GeneratedTool] = {}

    def generate(
        self,
        task_description: str,
        llm_response: str,
    ) -> GeneratedTool | None:
        """Parse, validate, and sandbox-test a tool from an LLM response.

        Args:
            task_description: The original task that triggered tool generation.
            llm_response: Raw LLM output containing the function code and test case.

        Returns:
            A GeneratedTool on success, or None if validation/sandbox fails.
        """
        code = _extract_code(llm_response)
        if not code:
            logger.warning("No code found in LLM response for: %s", task_description)
            return None

        # Sandbox the code.
        namespace = _sandbox_exec(code)
        if namespace is None:
            logger.warning("Sandbox rejected code for: %s", task_description)
            return None

        func_name = _extract_function_name(code)
        if not func_name or func_name not in namespace:
            logger.warning("No function found in sandbox namespace for: %s", task_description)
            return None

        func = namespace[func_name]
        if not callable(func):
            logger.warning("Extracted name '%s' is not callable", func_name)
            return None

        # Run the test case if provided.
        test_input_str, expected_str = _extract_test_case(llm_response)
        if test_input_str and expected_str:
            if not self._run_test(func, test_input_str, expected_str):
                logger.warning(
                    "Test case failed for generated tool '%s'", func_name,
                )
                return None

        docstring = _extract_docstring(code)
        description = docstring or task_description

        return GeneratedTool(
            name=func_name,
            description=description,
            func=func,
            source_code=code,
            task_description=task_description,
        )

    def register(self, tool: GeneratedTool) -> None:
        """Register a generated tool and persist it as a Skill node.

        If a tool with the same name is already registered, it is overwritten.
        """
        self._tools[tool.name] = tool
        logger.info("Registered dynamic tool: %s", tool.name)

        # Persist as Skill node in the knowledge graph.
        if self._store is not None:
            self._persist_skill(tool)

    def get_tool(self, name: str) -> GeneratedTool | None:
        """Retrieve a registered tool by exact name."""
        return self._tools.get(name)

    def find_tool(self, query: str) -> GeneratedTool | None:
        """Find a registered tool by keyword match against name and description.

        Uses simple word overlap scoring. Returns the best match or None.
        """
        if not self._tools:
            return None

        query_words = set(query.lower().split())
        best_tool: GeneratedTool | None = None
        best_score: float = 0.0

        for tool in self._tools.values():
            target_words = set(tool.name.lower().replace("_", " ").split())
            target_words |= set(tool.description.lower().split())
            target_words |= set(tool.task_description.lower().split())

            overlap = len(query_words & target_words)
            if overlap > best_score:
                best_score = overlap
                best_tool = tool

        if best_score > 0:
            return best_tool
        return None

    def list_tools(self) -> list[GeneratedTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    async def create_tool(
        self,
        task_description: str,
    ) -> GeneratedTool | None:
        """Full tool creation pipeline: LLM call -> sandbox -> register.

        1. Check if a matching tool already exists.
        2. If not, call the LLM to generate one.
        3. Validate and sandbox-test the generated code.
        4. Register on success and persist to knowledge graph.

        Returns the GeneratedTool on success, or None on failure.
        """
        # Check for existing tool first.
        existing = self.find_tool(task_description)
        if existing is not None:
            logger.info("Reusing existing tool '%s' for: %s", existing.name, task_description)
            return existing

        # Call LLM to generate the tool.
        prompt = _TOOL_GEN_PROMPT.format(task_description=task_description)
        task_node = TaskNode(
            id=str(uuid.uuid4()),
            description=task_description,
            is_atomic=True,
            domain=Domain.CODE,
            complexity=2,
            status=TaskStatus.READY,
        )
        messages = [
            {"role": "system", "content": "You are a tool-making assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            result: CompletionResult = await self._router.route(task_node, messages)
        except Exception as exc:
            logger.error("LLM call failed for tool generation: %s", exc)
            return None

        llm_response = result.content

        # Parse, validate, and sandbox-test.
        tool = self.generate(
            task_description=task_description,
            llm_response=llm_response,
        )
        if tool is None:
            return None

        # Register and persist.
        self.register(tool)
        return tool

    def load_from_graph(self) -> int:
        """Load previously persisted Skill nodes from the knowledge graph.

        Re-executes the stored source code in a sandbox and registers
        valid tools. Returns the number of tools successfully loaded.
        """
        if self._store is None:
            return 0

        rows = self._store.query("MATCH (s:Skill) RETURN s.*")
        loaded = 0

        for row in rows:
            name = str(row.get("s.name", ""))
            source_code = str(row.get("s.path", ""))
            description = str(row.get("s.description", ""))
            skill_id = str(row.get("s.id", ""))

            if not name or not source_code:
                continue

            # Re-validate in sandbox.
            namespace = _sandbox_exec(source_code)
            if namespace is None:
                logger.warning("Failed to reload Skill '%s' from graph", name)
                continue

            if name not in namespace or not callable(namespace[name]):
                logger.warning("Skill '%s' not found in reloaded namespace", name)
                continue

            tool = GeneratedTool(
                id=skill_id,
                name=name,
                description=description,
                func=namespace[name],
                source_code=source_code,
                task_description=description,
            )
            self._tools[name] = tool
            loaded += 1
            logger.info("Loaded Skill '%s' from knowledge graph", name)

        return loaded

    def _persist_skill(self, tool: GeneratedTool) -> None:
        """Create or update a Skill node in the knowledge graph.

        Uses the tool's source code stored in the ``path`` field (the Skill
        schema uses ``path`` as a STRING, which we repurpose for source code
        storage).
        """
        # Check if a Skill with this name already exists.
        existing = self._store.query(
            "MATCH (s:Skill) WHERE s.name = $name RETURN s.id",
            params={"name": tool.name},
        )

        if existing:
            # Update existing node.
            skill_id = str(existing[0].get("s.id", ""))
            self._store.update_node("Skill", skill_id, {
                "description": tool.description,
                "path": tool.source_code,
            })
            logger.debug("Updated Skill node '%s' in graph", tool.name)
        else:
            # Create new Skill node.
            self._store.create_node("Skill", {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "path": tool.source_code,
            })
            logger.debug("Created Skill node '%s' in graph", tool.name)

    @staticmethod
    def _run_test(func: Callable[..., Any], test_input_str: str, expected_str: str) -> bool:
        """Run a test case against a generated function.

        Parses the test_input and expected_output strings, calls the function,
        and compares the result. Returns True if the test passes.
        """
        try:
            # Parse test input -- try as a Python literal first.
            test_args = _parse_test_value(test_input_str)
            expected = _parse_test_value(expected_str)

            # Call the function with the parsed arguments.
            if isinstance(test_args, tuple):
                actual = func(*test_args)
            else:
                actual = func(test_args)

            # Compare with type coercion for numeric types.
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                return abs(float(actual) - float(expected)) < 1e-9
            return actual == expected

        except Exception as exc:
            logger.warning("Test execution failed: %s", exc)
            return False


def _parse_test_value(value_str: str) -> Any:
    """Parse a test value string into a Python object.

    Handles: numbers, strings, lists, tuples, dicts, booleans, None.
    For comma-separated values without brackets, returns a tuple.
    """
    stripped = value_str.strip()

    # Try ast.literal_eval first for structured values.
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        pass

    # Check for comma-separated values (multiple function arguments).
    if "," in stripped:
        parts = [p.strip() for p in stripped.split(",")]
        parsed_parts: list[Any] = []
        for part in parts:
            try:
                parsed_parts.append(ast.literal_eval(part))
            except (ValueError, SyntaxError):
                parsed_parts.append(part)
        return tuple(parsed_parts)

    # Fall back to string.
    return stripped
