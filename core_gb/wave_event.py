"""Wave-complete event dataclass for DAG execution progress tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WaveCompleteEvent:
    """Emitted after each topological wave finishes in the DAG executor.

    A wave is a batch of nodes at the same topological depth that can
    execute in parallel. After all nodes in a wave complete, this event
    is emitted before the next wave begins.

    Attributes:
        wave_index: Zero-based index of the completed wave.
        completed_nodes: Node IDs that completed in this wave.
        accumulated_results: Map of node ID to output string for all
            nodes completed so far (across all waves up to and including
            this one).
        remaining_nodes: Node IDs that have not yet been executed.
    """

    wave_index: int
    completed_nodes: list[str]
    accumulated_results: dict[str, str]
    remaining_nodes: list[str]
