"""Knowledge graph schema definitions for Kuzu.

Node types: User, Project, File, Service, Contact, Pattern, Memory, Task, Skill, ExecutionTree
Edge types: OWNS, USES, STUDIES_AT, ABOUT, PRODUCED, CREATED_PATTERN, DEPENDS_ON, CONTEXT_FROM,
            INVOLVES, DERIVED_FROM
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeType:
    """Definition of a node type in the knowledge graph."""

    name: str
    properties: dict[str, str]


@dataclass(frozen=True)
class EdgeType:
    """Definition of an edge type in the knowledge graph."""

    name: str
    from_type: str
    to_type: str
    properties: dict[str, str]


# Node type definitions
NODE_TYPES: list[NodeType] = [
    NodeType("User", {
        "id": "STRING", "name": "STRING", "role": "STRING",
        "institution": "STRING", "interests": "STRING",
    }),
    NodeType("Project", {
        "id": "STRING", "name": "STRING", "path": "STRING",
        "language": "STRING", "framework": "STRING", "status": "STRING",
    }),
    NodeType("File", {
        "id": "STRING", "path": "STRING", "type": "STRING",
        "description": "STRING",
    }),
    NodeType("Service", {
        "id": "STRING", "name": "STRING", "type": "STRING",
        "url": "STRING", "status": "STRING",
    }),
    NodeType("Contact", {
        "id": "STRING", "name": "STRING", "relationship": "STRING",
        "platform": "STRING",
    }),
    NodeType("PatternNode", {
        "id": "STRING", "trigger_template": "STRING", "description": "STRING",
        "variable_slots": "STRING", "tree_template": "STRING", "success_count": "INT64",
        "avg_tokens": "DOUBLE", "avg_latency_ms": "DOUBLE",
        "created_at": "TIMESTAMP", "last_used": "TIMESTAMP",
    }),
    NodeType("Memory", {
        "id": "STRING", "content": "STRING", "category": "STRING",
        "confidence": "DOUBLE", "source_episode": "STRING",
        "valid_from": "TIMESTAMP", "valid_until": "TIMESTAMP",
    }),
    NodeType("Task", {
        "id": "STRING", "description": "STRING", "domain": "STRING",
        "complexity": "INT64", "status": "STRING",
        "tokens_used": "INT64", "latency_ms": "DOUBLE",
        "created_at": "TIMESTAMP", "completed_at": "TIMESTAMP",
    }),
    NodeType("Skill", {
        "id": "STRING", "name": "STRING", "description": "STRING",
        "path": "STRING",
    }),
    NodeType("ExecutionTree", {
        "id": "STRING", "root_task_id": "STRING",
        "total_nodes": "INT64", "total_tokens": "INT64",
        "total_latency_ms": "DOUBLE", "created_at": "TIMESTAMP",
    }),
]

# Edge type definitions
EDGE_TYPES: list[EdgeType] = [
    EdgeType("OWNS", "User", "Project", {}),
    EdgeType("USES", "User", "Service", {}),
    EdgeType("STUDIES_AT", "User", "Service", {"program": "STRING", "since": "TIMESTAMP"}),
    EdgeType("ABOUT", "Memory", "User", {}),
    EdgeType("ABOUT_PROJECT", "Memory", "Project", {}),
    EdgeType("PRODUCED", "Task", "File", {}),
    EdgeType("CREATED_PATTERN", "ExecutionTree", "PatternNode", {}),
    EdgeType("DEPENDS_ON", "Task", "Task", {"data_key": "STRING"}),
    EdgeType("CONTEXT_FROM", "Task", "Memory", {}),
    EdgeType("INVOLVES", "Task", "Service", {}),
    EdgeType("DERIVED_FROM", "ExecutionTree", "Task", {}),
    EdgeType("HAS_SKILL", "User", "Skill", {}),
]


def get_create_node_cypher(node_type: NodeType) -> str:
    """Generate Cypher CREATE NODE TABLE statement for a node type."""
    props = ", ".join(f"{k} {v}" for k, v in node_type.properties.items())
    return f"CREATE NODE TABLE IF NOT EXISTS {node_type.name}({props}, PRIMARY KEY(id))"


def get_create_edge_cypher(edge_type: EdgeType) -> str:
    """Generate Cypher CREATE REL TABLE statement for an edge type."""
    props = ""
    if edge_type.properties:
        props = ", " + ", ".join(f"{k} {v}" for k, v in edge_type.properties.items())
    return (
        f"CREATE REL TABLE IF NOT EXISTS {edge_type.name}"
        f"(FROM {edge_type.from_type} TO {edge_type.to_type}{props})"
    )
