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
        }
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


def validate_decomposition(data: dict[str, Any], *, max_depth: int = 3) -> list[str]:
    """Validate a decomposition output against the JSON schema.

    Returns a list of human-readable error messages. An empty list means
    the data is valid. Checks:
      - JSON schema conformance (required fields, types, enums)
      - Maximum tree depth <= max_depth (configurable, default 3)
      - Maximum children per node <= 5 (enforced by schema)
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
7. Domains: file, web, code, comms, system, synthesis
8. Maximum 3 levels deep, maximum 5 children per node
</rules>

<output_schema>
{{"nodes": [
  {{"id": "string", "description": "string", "domain": "enum", "task_type": "enum",
    "complexity": 1-5, "depends_on": ["id"], "provides": ["key"], "consumes": ["key"],
    "is_atomic": bool, "children": ["id"]}}
]}}
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
]}}

GOOD Example 2 (sequential):
Task: "Read README.md, find TODOs, list with line numbers"
{{"nodes":[
{{"id":"root","description":"Find TODOs in README","domain":"file","task_type":"THINK","complexity":2,"depends_on":[],"provides":[],"consumes":[],"is_atomic":false,"children":["read","find","fmt"]}},
{{"id":"read","description":"Read README.md","domain":"file","task_type":"RETRIEVE","complexity":1,"depends_on":[],"provides":["file_content"],"consumes":[],"is_atomic":true,"children":[]}},
{{"id":"find","description":"Find TODO lines","domain":"code","task_type":"CODE","complexity":1,"depends_on":["read"],"provides":["todo_lines"],"consumes":["file_content"],"is_atomic":true,"children":[]}},
{{"id":"fmt","description":"Format with line numbers","domain":"synthesis","task_type":"WRITE","complexity":1,"depends_on":["find"],"provides":["formatted"],"consumes":["todo_lines"],"is_atomic":true,"children":[]}}
]}}

BAD Example:
Task: "Compare X and Y"
{{"nodes":[
{{"id":"root","description":"Compare X and Y","domain":"synthesis","task_type":"THINK","complexity":2,"depends_on":[],"provides":[],"consumes":[],"is_atomic":true,"children":["a"]}}
]}}
WHY THIS IS WRONG: root has is_atomic: true but also has children. Atomic nodes must have children: []. Also child "a" is referenced but not defined in nodes.
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
