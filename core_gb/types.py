"""Core data structures for the GraphBot execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_gb.verification import VerificationResult


class TaskStatus(str, Enum):
    CREATED = "created"
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


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

    TaskNode is intentionally mutable: it tracks execution state (status, output,
    timing) that changes during pipeline processing. For safe concurrent reads,
    use snapshot() to get a frozen copy.
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
    expandable: bool = False
    domain: Domain = Domain.SYNTHESIS
    complexity: int = 1
    flow_type: FlowType = FlowType.PARALLEL
    model: str | None = None
    status: TaskStatus = TaskStatus.CREATED

    # Data flow
    context: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)

    # Tool routing (Phase 11)
    tool_method: str | None = None
    tool_params: dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class ConditionalNode(TaskNode):
    """A branch node that routes execution to then_branch or else_branch.

    The condition is evaluated against predecessor output using simple string
    matching (e.g. "contains 'valid'"). When the condition evaluates to True,
    nodes listed in then_branch are executed and else_branch nodes are marked
    SKIPPED. When False, else_branch executes and then_branch is SKIPPED.

    ConditionalNode is not atomic -- it acts as a routing gate, not a unit of
    work. The DAGExecutor handles condition evaluation and branch selection
    during the dispatch loop.
    """

    condition: str = ""
    then_branch: list[str] = field(default_factory=list)
    else_branch: list[str] = field(default_factory=list)


@dataclass
class LoopNode(TaskNode):
    """A loop node that iteratively executes its body nodes until an exit
    condition is met or max_iterations is reached.

    Each iteration runs the body nodes (via the DAG executor), checks the
    exit condition against the combined output, and -- if not satisfied --
    feeds the previous iteration's output as context into the next iteration,
    enabling retry-with-context behaviour.

    LoopNode is not atomic -- it acts as a control flow wrapper. The
    DAGExecutor.execute_loop method handles iteration, condition checking,
    and context injection.

    Attributes:
        max_iterations: Hard cap on the number of iterations. Defaults to 3.
        exit_condition: A string expression evaluated against the iteration
            output. Format is ``"<check>:<value>"`` where ``<check>`` is a
            predicate such as ``contains``. An empty string means no early
            exit (runs until max_iterations).
        body_nodes: Ordered list of node IDs that form the loop body. These
            must be present in the nodes list passed to ``execute_loop``.
    """

    max_iterations: int = 3
    exit_condition: str = ""
    body_nodes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompletionResult:
    """Result from a single LLM completion call.

    Attributes:
        logprobs: Optional list of token log-probabilities returned by the
            provider (e.g. OpenRouter). None when the provider does not
            support or did not return logprobs.
    """

    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost: float
    logprobs: list[float] | None = None


@dataclass(frozen=True)
class ExecutionResult:
    """Result from executing a task tree. Frozen for safe concurrent reads."""

    root_id: str
    output: str
    success: bool
    total_nodes: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    context_tokens: int = 0
    model_used: str = ""
    tools_used: int = 0
    llm_calls: int = 0
    nodes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    verification_results: tuple[VerificationResult, ...] = ()


@dataclass(frozen=True)
class Reflection:
    """Structured failure analysis from a failed task execution.

    Generated by the reflection engine after a task fails. Stored as a Memory
    node in the knowledge graph for future retrieval during decomposition.
    """

    what_failed: str
    why: str
    what_to_try: str


@dataclass(frozen=True)
class Pattern:
    """A reusable execution tree template extracted from completed tasks."""

    id: str
    trigger: str
    description: str
    variable_slots: tuple[str, ...] = ()
    tree_template: str = ""
    success_count: int = 0
    failure_count: int = 0
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime | None = None


@dataclass(frozen=True)
class GraphContext:
    """Compact context assembled from the knowledge graph for a specific task.
    Frozen for safe concurrent reads across parallel task execution.
    """

    user_summary: str = ""
    relevant_entities: tuple[dict[str, str], ...] = ()
    active_memories: tuple[str, ...] = ()
    matching_patterns: tuple[Pattern, ...] = ()
    reflections: tuple[dict[str, str], ...] = ()
    total_tokens: int = 0
    token_count: int = 0

    def format(self) -> str:
        """Format context as a compact string for prompt injection.

        Includes a PAST FAILURES section when reflections are available,
        listing each reflection with its original task description, what
        failed, the root cause, and what to try differently.
        """
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
        if self.reflections:
            for refl in self.reflections:
                task_desc = refl.get("task_description", "unknown task")
                what_failed = refl.get("what_failed", "")
                why = refl.get("why", "")
                what_to_try = refl.get("what_to_try", "")
                parts.append(
                    f"PAST FAILURE [{task_desc}]: "
                    f"{what_failed} -- cause: {why} -- suggestion: {what_to_try}"
                )
        return "\n".join(parts)
