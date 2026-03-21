"""Constrained decomposition schema and validator.

Defines the JSON schema for task decomposition output (flat node list format)
and provides validation against it. This module will be extended in T014-T016
with the actual decomposition logic.
"""

from __future__ import annotations

from typing import Any

import jsonschema

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
