"""Core data structures for the GraphBot execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    CREATED = "created"
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FlowType(str, Enum):
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    FALLBACK = "fallback"


class Domain(str, Enum):
    FILE = "file"
    WEB = "web"
    CODE = "code"
    COMMS = "comms"
    SYSTEM = "system"
    SYNTHESIS = "synthesis"


@dataclass
class TaskNode:
    """The universal unit of work in the GraphBot execution engine.

    Every task -- whether a single leaf operation or a complex multi-level tree --
    is represented as a TaskNode. Nodes declare typed data contracts via `provides`
    and `consumes` fields, enabling zero-copy data forwarding between dependent nodes.
    """

    id: str
    description: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)

    # Dependency contracts (typed data flow)
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    consumes: list[str] = field(default_factory=list)

    # Execution parameters
    is_atomic: bool = False
    domain: Domain = Domain.SYNTHESIS
    complexity: int = 1
    flow_type: FlowType = FlowType.PARALLEL
    model: str | None = None
    status: TaskStatus = TaskStatus.CREATED

    # Data flow
    context: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class ExecutionResult:
    """Result from executing a task tree."""

    root_id: str
    output: str
    success: bool
    total_nodes: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class Pattern:
    """A reusable execution tree template extracted from completed tasks."""

    id: str
    trigger: str
    description: str
    variable_slots: list[str] = field(default_factory=list)
    tree_template: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime | None = None


@dataclass
class GraphContext:
    """Compact context assembled from the knowledge graph for a specific task."""

    user_summary: str = ""
    relevant_entities: list[dict[str, str]] = field(default_factory=list)
    active_memories: list[str] = field(default_factory=list)
    matching_patterns: list[Pattern] = field(default_factory=list)
    token_count: int = 0

    def format(self) -> str:
        """Format context as a compact string for prompt injection."""
        parts: list[str] = []
        if self.user_summary:
            parts.append(f"USER: {self.user_summary}")
        for entity in self.relevant_entities:
            label = entity.get("type", "ENTITY")
            name = entity.get("name", "unknown")
            details = entity.get("details", "")
            parts.append(f"{label}: {name} | {details}")
        for memory in self.active_memories:
            parts.append(f"MEMORY: {memory}")
        for pattern in self.matching_patterns:
            parts.append(
                f"PATTERN: \"{pattern.trigger}\" available "
                f"({pattern.success_count} successes, avg {pattern.avg_tokens:.0f} tokens)"
            )
        return "\n".join(parts)
