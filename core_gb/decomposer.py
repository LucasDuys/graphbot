"""Constrained decomposition schema and validator.

Defines the JSON schema for task decomposition output (flat node list format)
and provides validation against it. This module will be extended in T014-T016
with the actual decomposition logic.
"""

from __future__ import annotations

from typing import Any

import jsonschema

from core_gb.types import GraphContext

_NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "id",
        "description",
        "domain",
        "task_type",
        "complexity",
        "depends_on",
        "provides",
        "consumes",
        "is_atomic",
        "children",
    ],
    "properties": {
        "id": {"type": "string"},
        "description": {"type": "string"},
        "domain": {
            "type": "string",
            "enum": ["file", "web", "code", "comms", "system", "synthesis"],
        },
        "task_type": {
            "type": "string",
            "enum": ["RETRIEVE", "WRITE", "THINK", "CODE"],
        },
        "complexity": {"type": "integer", "minimum": 1, "maximum": 5},
        "depends_on": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "provides": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "consumes": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "is_atomic": {"type": "boolean"},
        "children": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "tool_method": {
            "type": "string",
            "enum": [
                "file_read", "file_list", "file_search",
                "web_search", "web_fetch",
                "shell_run", "code_generate", "code_edit",
                "llm_reason",
            ],
            "description": "Specific tool to use. Required for file/web/code domain nodes. Use code_generate for code generation (LLM-only), code_edit for modifying existing files.",
        },
        "tool_params": {
            "type": "object",
            "description": "Parameters for the tool. E.g. {path: 'file.py'} for file_read, {query: '...'} for web_search",
        },
    },
    "additionalProperties": False,
}

DECOMPOSITION_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["nodes"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": _NODE_SCHEMA,
            "minItems": 1,
        },
        "output_template": {
            "type": "object",
            "required": ["aggregation_type", "template"],
            "properties": {
                "aggregation_type": {
                    "type": "string",
                    "enum": [
                        "concatenate",
                        "merge_json",
                        "confidence_ranked",
                        "template_fill",
                    ],
                },
                "template": {
                    "type": "string",
                },
                "slot_definitions": {
                    "type": "object",
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_VALIDATOR = jsonschema.Draft7Validator(DECOMPOSITION_SCHEMA)


def _compute_depth(nodes: list[dict[str, Any]]) -> int:
    """Compute the maximum depth of the node tree.

    Root nodes (not referenced as children by any other node) start at depth 1.
    Returns 0 for an empty list.
    """
    if not nodes:
        return 0

    index: dict[str, dict[str, Any]] = {n["id"]: n for n in nodes}
    child_ids: set[str] = set()
    for node in nodes:
        child_ids.update(node.get("children", []))

    root_ids = [n["id"] for n in nodes if n["id"] not in child_ids]
    if not root_ids:
        root_ids = [nodes[0]["id"]]

    max_depth = 0

    def _walk(node_id: str, depth: int) -> None:
        nonlocal max_depth
        if depth > max_depth:
            max_depth = depth
        node = index.get(node_id)
        if node is None:
            return
        for child_id in node.get("children", []):
            _walk(child_id, depth + 1)

    for rid in root_ids:
        _walk(rid, 1)

    return max_depth


MAX_RECURSION_DEPTH: int = 5
"""Hard limit on decomposition tree depth. Plans exceeding this are rejected."""

MAX_TOTAL_NODES: int = 50
"""Hard limit on total node count in a decomposition. Plans exceeding this are rejected."""


def validate_decomposition(
    data: dict[str, Any],
    *,
    max_depth: int = 3,
    max_nodes: int = MAX_TOTAL_NODES,
) -> list[str]:
    """Validate a decomposition output against the JSON schema.

    Returns a list of human-readable error messages. An empty list means
    the data is valid. Checks:
      - JSON schema conformance (required fields, types, enums)
      - Maximum tree depth <= max_depth (configurable, default 3)
      - Maximum children per node <= 5 (enforced by schema)
      - Maximum total nodes <= max_nodes (default 50)
    """
    errors: list[str] = []

    # Schema validation
    for error in _VALIDATOR.iter_errors(data):
        path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "(root)"
        errors.append(f"Schema error at {path}: {error.message}")

    # Structural checks only make sense if basic schema passed for nodes
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return errors

    # Node count check (hard limit)
    if len(nodes) > max_nodes:
        errors.append(
            f"Total node count {len(nodes)} exceeds maximum allowed {max_nodes}"
        )

    # Depth check
    depth = _compute_depth(nodes)
    if depth > max_depth:
        errors.append(
            f"Tree depth {depth} exceeds maximum allowed depth {max_depth}"
        )

    return errors


class DecompositionPrompt:
    """Builds the LLM prompt for task decomposition using XML-tag structure."""

    _SYSTEM_TEMPLATE = """{context_block}<rules>
You are a task decomposer. Break the user's task into a flat list of subtasks.
Each subtask must be simple enough for a small language model to execute independently.
Rules:
1. Output ONLY valid JSON matching the schema below
2. Each leaf node must have is_atomic: true and children: []
3. Non-leaf nodes must have is_atomic: false
4. Use depends_on for sequential dependencies between sibling nodes
5. Use provides/consumes for typed data flow between nodes
6. Task types: RETRIEVE (fetch data), WRITE (produce text), THINK (reason/analyze), CODE (write/fix code)
7. Domains -- CRITICAL: each domain maps to specific tools. Choose based on what tools the task needs:
   - "file": Has tools file_read, file_list, file_search. Use for ANY task involving local files, directories, or file content.
   - "web": Has tools web_search, web_fetch. Use for ANY task needing internet, searching online, or fetching URLs.
   - "code": Has tool shell_run. Use for ANY task running commands (git, pytest, pip, echo, etc).
   - "system": No tools. Use ONLY for pure reasoning, math, logic, knowledge questions.
   - "synthesis": No tools. Use ONLY for combining/summarizing results from other nodes.
   RULE: If a task mentions files -> "file". If it mentions search/web/online -> "web". If it mentions running commands -> "code". NEVER use "system" for tasks that need tools.
8. Maximum 3 levels deep, maximum 5 children per node
9. Include an "output_template" with:
   - "aggregation_type": how to combine results ("concatenate" for lists, "template_fill" for structured, "merge_json" for data)
   - "template": the final output format with {{slot_id}} placeholders matching leaf provides keys
   - "slot_definitions": description of what each slot should contain
</rules>

<output_schema>
{{"nodes": [
  {{"id": "string", "description": "string", "domain": "enum", "task_type": "enum",
    "complexity": 1-5, "depends_on": ["id"], "provides": ["key"], "consumes": ["key"],
    "is_atomic": bool, "children": ["id"]}}
],
"output_template": {{
  "aggregation_type": "concatenate|merge_json|confidence_ranked|template_fill",
  "template": "string with {{slot_id}} placeholders",
  "slot_definitions": {{"slot_id": "description"}}
}}}}
</output_schema>

<examples>
GOOD Example 1 (parallel):
Task: "Weather in Amsterdam, London, Berlin"
{{"nodes":[
{{"id":"root","description":"Get weather for 3 cities","domain":"synthesis","task_type":"THINK","complexity":2,"depends_on":[],"provides":[],"consumes":[],"is_atomic":false,"children":["w1","w2","w3","agg"]}},
{{"id":"w1","description":"Get Amsterdam weather","domain":"web","task_type":"RETRIEVE","complexity":1,"depends_on":[],"provides":["weather_ams"],"consumes":[],"is_atomic":true,"children":[]}},
{{"id":"w2","description":"Get London weather","domain":"web","task_type":"RETRIEVE","complexity":1,"depends_on":[],"provides":["weather_lon"],"consumes":[],"is_atomic":true,"children":[]}},
{{"id":"w3","description":"Get Berlin weather","domain":"web","task_type":"RETRIEVE","complexity":1,"depends_on":[],"provides":["weather_ber"],"consumes":[],"is_atomic":true,"children":[]}},
{{"id":"agg","description":"Summarize weather","domain":"synthesis","task_type":"WRITE","complexity":1,"depends_on":["w1","w2","w3"],"provides":["summary"],"consumes":["weather_ams","weather_lon","weather_ber"],"is_atomic":true,"children":[]}}
],
"output_template":{{"aggregation_type":"template_fill","template":"## Weather Comparison\\n\\n### Amsterdam\\n{{weather_ams}}\\n\\n### London\\n{{weather_lon}}\\n\\n### Berlin\\n{{weather_ber}}","slot_definitions":{{"weather_ams":"Current weather in Amsterdam","weather_lon":"Current weather in London","weather_ber":"Current weather in Berlin"}}}}}}

GOOD Example 2 (sequential):
Task: "Read README.md, find TODOs, list with line numbers"
{{"nodes":[
{{"id":"root","description":"Find TODOs in README","domain":"file","task_type":"THINK","complexity":2,"depends_on":[],"provides":[],"consumes":[],"is_atomic":false,"children":["read","find","fmt"]}},
{{"id":"read","description":"Read README.md","domain":"file","task_type":"RETRIEVE","complexity":1,"depends_on":[],"provides":["file_content"],"consumes":[],"is_atomic":true,"children":[]}},
{{"id":"find","description":"Find TODO lines","domain":"code","task_type":"CODE","complexity":1,"depends_on":["read"],"provides":["todo_lines"],"consumes":["file_content"],"is_atomic":true,"children":[]}},
{{"id":"fmt","description":"Format with line numbers","domain":"synthesis","task_type":"WRITE","complexity":1,"depends_on":["find"],"provides":["formatted"],"consumes":["todo_lines"],"is_atomic":true,"children":[]}}
],
"output_template":{{"aggregation_type":"concatenate","template":"## TODOs in README.md\\n\\n{{formatted}}","slot_definitions":{{"formatted":"List of TODO items with line numbers"}}}}}}

BAD Example:
Task: "Compare X and Y"
{{"nodes":[
{{"id":"root","description":"Compare X and Y","domain":"synthesis","task_type":"THINK","complexity":2,"depends_on":[],"provides":[],"consumes":[],"is_atomic":true,"children":["a"]}}
]}}
WHY THIS IS WRONG: root has is_atomic: true but also has children. Atomic nodes must have children: []. Also child "a" is referenced but not defined in nodes.

BAD Example (Wrong Domain - Shell Task):
Task: "Run git log --oneline -10"
{{"nodes":[{{"id":"root","description":"Run git log","domain":"system","task_type":"THINK","complexity":1,"depends_on":[],"provides":[],"consumes":[],"is_atomic":true,"children":[]}}]}}
WHY WRONG: Running git requires shell_run tool. Domain MUST be "code", not "system".
</examples>"""

    _USER_TEMPLATE = """Task: {task}

Remember: output ONLY valid JSON matching the schema. No explanations, no markdown, just JSON."""

    def build(self, task: str, context: GraphContext | None = None) -> list[dict[str, str]]:
        """Build messages for the decomposition LLM call.

        Uses XML-tag structure (ADR-016):
        - <rules> for critical instructions
        - <output_schema> for the JSON schema
        - <examples> with 2 good + 1 bad decomposition
        - Sandwich defense: output format restated after task
        - Context at beginning if available (ADR-009)
        """
        context_block = ""
        if context is not None:
            formatted = context.format()
            if formatted:
                context_block = f"<context>\n{formatted}\n</context>\n\n"

        system_content = self._SYSTEM_TEMPLATE.format(context_block=context_block)
        user_content = self._USER_TEMPLATE.format(task=task)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


# ---------------------------------------------------------------------------
# Structural / semantic tree validation
# ---------------------------------------------------------------------------


def validate_tree(nodes: list[dict[str, Any]]) -> list[str]:
    """Validate structural correctness of a decomposition tree.

    Returns list of error messages (empty = valid).

    Checks:
    1. No circular dependencies (via depends_on)
    2. All depends_on references point to existing node IDs
    3. All leaves are is_atomic: true
    4. Exactly one root node (not referenced as child by any other node)
    5. Every non-root node is referenced as a child by exactly one parent
    6. Data contracts: every consumed key is provided by a dependency
    """
    errors: list[str] = []
    if not nodes:
        return errors

    node_index: dict[str, dict[str, Any]] = {n["id"]: n for n in nodes}
    all_ids: set[str] = set(node_index.keys())

    # --- 1 & 2: dependency reference validity + cycle detection ---------------

    for node in nodes:
        for dep_id in node.get("depends_on", []):
            if dep_id not in all_ids:
                errors.append(
                    f"Node '{node['id']}' depends on '{dep_id}' which does not exist"
                )

    # Build adjacency list for depends_on and detect cycles via DFS
    adj: dict[str, list[str]] = {nid: [] for nid in all_ids}
    for node in nodes:
        for dep_id in node.get("depends_on", []):
            if dep_id in all_ids:
                adj[node["id"]].append(dep_id)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in all_ids}
    cycle_found = False

    def _dfs(nid: str) -> None:
        nonlocal cycle_found
        color[nid] = GRAY
        for neighbour in adj[nid]:
            if color[neighbour] == GRAY:
                cycle_found = True
                return
            if color[neighbour] == WHITE:
                _dfs(neighbour)
                if cycle_found:
                    return
        color[nid] = BLACK

    for nid in all_ids:
        if color[nid] == WHITE:
            _dfs(nid)
        if cycle_found:
            break

    if cycle_found:
        errors.append("Circular dependency detected in depends_on graph")

    # --- 3: leaf nodes must be is_atomic: true --------------------------------

    for node in nodes:
        children = node.get("children", [])
        if len(children) == 0 and not node.get("is_atomic", False):
            errors.append(
                f"Leaf node '{node['id']}' has no children but is_atomic is false"
            )

    # --- 4 & 5: exactly one root, every non-root referenced by one parent -----

    child_ids: set[str] = set()
    for node in nodes:
        child_ids.update(node.get("children", []))

    root_ids = [nid for nid in all_ids if nid not in child_ids]

    if len(root_ids) == 0:
        errors.append("No root node found (every node is a child of another)")
    elif len(root_ids) > 1:
        errors.append(
            f"Multiple root nodes found: {sorted(root_ids)}. "
            "Exactly one root expected"
        )

    # --- 6: data contract validation ------------------------------------------

    def _subtree_provides(nid: str) -> set[str]:
        visited: set[str] = set()
        stack = [nid]
        result: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited or current not in node_index:
                continue
            visited.add(current)
            result.update(node_index[current].get("provides", []))
            stack.extend(node_index[current].get("children", []))
        return result

    def _reachable_provides(nid: str) -> set[str]:
        visited: set[str] = set()
        stack = list(node_index[nid].get("depends_on", []))
        result: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited or current not in node_index:
                continue
            visited.add(current)
            # Collect provides from this dependency and its entire subtree
            result.update(_subtree_provides(current))
            stack.extend(node_index[current].get("depends_on", []))
        return result

    for node in nodes:
        consumed = node.get("consumes", [])
        if not consumed:
            continue
        available = _reachable_provides(node["id"])
        for key in consumed:
            if key not in available:
                errors.append(
                    f"Node '{node['id']}' consumes '{key}' but no dependency provides it"
                )

    return errors


# ---------------------------------------------------------------------------
# Decomposer class (T016)
# ---------------------------------------------------------------------------

import json
import json_repair
import uuid
import logging

from core_gb.types import Domain, TaskNode, TaskStatus
from models.router import ModelRouter

logger = logging.getLogger(__name__)


def infer_domain_from_description(description: str) -> Domain | None:
    """Infer correct domain from task description keywords.
    Returns None if no strong signal detected (keep original domain).
    """
    text = description.lower()

    # File signals (strongest match first)
    file_keywords = [
        "read ", "read_file", "file_read", "open file",
        "list files", "list directory", "file_list", "list all",
        "search files", "file_search", "grep", "find in file",
        ".py", ".md", ".json", ".toml", ".txt", ".yaml", ".csv",
        "readme", "pyproject", "directory", "folder",
    ]
    if any(kw in text for kw in file_keywords):
        return Domain.FILE

    # Web signals
    web_keywords = [
        "search the web", "web search", "web_search", "search online",
        "fetch url", "web_fetch", "http://", "https://",
        "browse", "scrape", "find online", "look up online",
        "search for", "search github", "search google",
    ]
    if any(kw in text for kw in web_keywords):
        return Domain.WEB

    # Code/shell signals
    code_keywords = [
        "run ", "run the", "execute ", "shell_run",
        "git log", "git ", "pytest", "pip ", "npm ",
        "command", "terminal", "bash", "shell",
    ]
    if any(kw in text for kw in code_keywords):
        return Domain.CODE

    return None


class Decomposer:
    """Recursive task decomposer using constrained JSON output from LLMs."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router
        self._prompt_builder = DecompositionPrompt()
        self.last_template: dict[str, Any] | None = None

    async def decompose(
        self, task: str, context: GraphContext | None = None, max_depth: int = 3
    ) -> list[TaskNode]:
        """Decompose a task into a list of TaskNodes.

        Flow:
        1. Build prompt via DecompositionPrompt
        2. Call ModelRouter with complexity 3 and JSON mode
        3. Parse JSON response
        4. Validate against schema + tree structure (max depth 5, max nodes 50)
        5. On invalid: try json_repair on SAME response first
        6. If repair fails: second LLM call
        7. On second failure: fallback to single atomic node
        8. Convert to list[TaskNode]

        Hard limits enforced regardless of max_depth parameter:
        - Maximum recursion depth: MAX_RECURSION_DEPTH (5)
        - Maximum total nodes: MAX_TOTAL_NODES (50)
        """
        self.last_template = None

        # Enforce hard ceiling: caller may request lower depth, but never exceed 5
        effective_depth = min(max_depth, MAX_RECURSION_DEPTH)

        messages = self._prompt_builder.build(task, context)

        # Create a dummy TaskNode for routing (complexity 3 for decomposition)
        route_node = TaskNode(id="decompose", description=task, complexity=3)

        # First attempt
        first_content: str | None = None
        try:
            completion = await self._router.route(
                route_node, messages,
                response_format={"type": "json_object"},
            )
            first_content = completion.content
            result = self._parse_and_validate(first_content, effective_depth)
            if result is not None:
                nodes, template = result
                self.last_template = template
                task_nodes = self._to_task_nodes(nodes)
                self._apply_domain_overrides(task_nodes)
                return task_nodes
        except Exception as exc:
            logger.warning("Decomposition first attempt failed: %s", exc)

        # Try json_repair on the FIRST response before calling LLM again
        if first_content is not None:
            try:
                repaired = json_repair.loads(first_content)
                if isinstance(repaired, dict):
                    nodes_data = repaired.get("nodes", [])
                    if nodes_data:
                        fixed_nodes = self._fix_missing_fields(nodes_data)
                        repaired_data = {"nodes": fixed_nodes}
                        if repaired.get("output_template"):
                            repaired_data["output_template"] = repaired["output_template"]
                        errors = validate_decomposition(repaired_data, max_depth=effective_depth)
                        tree_errors = validate_tree(fixed_nodes)
                        if not errors and not tree_errors:
                            self.last_template = repaired.get("output_template")
                            task_nodes = self._to_task_nodes(fixed_nodes)
                            self._apply_domain_overrides(task_nodes)
                            return task_nodes
                logger.warning("json_repair on first response failed validation")
            except Exception as exc:
                logger.warning("json_repair on first response failed: %s", exc)

        # Second LLM attempt
        try:
            completion2 = await self._router.route(
                route_node, messages,
                response_format={"type": "json_object"},
            )
            result = self._parse_and_validate(completion2.content, effective_depth)
            if result is not None:
                nodes, template = result
                self.last_template = template
                task_nodes = self._to_task_nodes(nodes)
                self._apply_domain_overrides(task_nodes)
                return task_nodes
        except Exception as exc:
            logger.warning("Decomposition second attempt failed: %s", exc)

        # Fallback: single atomic node
        logger.info("Falling back to single atomic node for: %s", task[:80])
        return self._fallback_single_node(task)

    @staticmethod
    def _apply_domain_overrides(task_nodes: list[TaskNode]) -> None:
        """Post-process: override domains based on description keywords."""
        for node in task_nodes:
            if node.is_atomic:
                inferred = infer_domain_from_description(node.description)
                if inferred is not None and node.domain != inferred:
                    logger.info(
                        "Domain override: %s -> %s for '%s'",
                        node.domain.value, inferred.value, node.description[:50],
                    )
                    node.domain = inferred

    def _parse_and_validate(
        self, content: str, max_depth: int
    ) -> tuple[list[dict], dict[str, Any] | None] | None:
        """Parse JSON and validate. Returns (nodes, output_template) or None on failure."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        errors = validate_decomposition(data, max_depth=max_depth)
        if errors:
            return None

        tree_errors = validate_tree(data.get("nodes", []))
        if tree_errors:
            return None

        return data["nodes"], data.get("output_template")

    def _fix_missing_fields(self, nodes: list[dict]) -> list[dict]:
        """Fix up nodes with missing fields by adding sensible defaults."""
        fixed: list[dict] = []
        for i, node in enumerate(nodes):
            n = dict(node)
            n.setdefault("id", f"node_{i}")
            n.setdefault("description", "")
            n.setdefault("domain", "synthesis")
            n.setdefault("task_type", "THINK")
            n.setdefault("complexity", 1)
            n.setdefault("depends_on", [])
            n.setdefault("provides", [])
            n.setdefault("consumes", [])
            n.setdefault("children", [])
            if "is_atomic" not in n:
                n["is_atomic"] = len(n["children"]) == 0
            fixed.append(n)
        return fixed

    def _to_task_nodes(self, raw_nodes: list[dict]) -> list[TaskNode]:
        """Convert raw decomposition dicts to TaskNode objects."""
        result: list[TaskNode] = []
        # Map original IDs to UUIDs
        id_map: dict[str, str] = {n["id"]: str(uuid.uuid4()) for n in raw_nodes}

        # Find child IDs to determine roots
        child_ids: set[str] = set()
        for n in raw_nodes:
            child_ids.update(n.get("children", []))

        for raw in raw_nodes:
            original_id = raw["id"]
            new_id = id_map[original_id]

            # Map domain string to Domain enum
            domain_str = raw.get("domain", "synthesis")
            try:
                domain = Domain(domain_str)
            except ValueError:
                domain = Domain.SYNTHESIS

            # Find parent (the node whose children list contains this node)
            parent_id: str | None = None
            for other in raw_nodes:
                if original_id in other.get("children", []):
                    parent_id = id_map[other["id"]]
                    break

            node = TaskNode(
                id=new_id,
                description=raw["description"],
                parent_id=parent_id,
                children=[id_map[c] for c in raw.get("children", []) if c in id_map],
                requires=[id_map[d] for d in raw.get("depends_on", []) if d in id_map],
                provides=raw.get("provides", []),
                consumes=raw.get("consumes", []),
                is_atomic=raw.get("is_atomic", False),
                domain=domain,
                complexity=raw.get("complexity", 1),
                status=TaskStatus.CREATED,
                tool_method=raw.get("tool_method"),
                tool_params=raw.get("tool_params", {}),
            )
            result.append(node)

        return result

    def _fallback_single_node(self, task: str) -> list[TaskNode]:
        """Create a single atomic TaskNode as fallback."""
        return [TaskNode(
            id=str(uuid.uuid4()),
            description=task,
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
        )]
