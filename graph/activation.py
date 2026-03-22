"""ACT-R activation model for graph node retrieval scoring.

Implements a simplified ACT-R (Adaptive Control of Thought -- Rational) model
for computing activation scores on knowledge graph nodes. Activation determines
which nodes are most relevant for retrieval, based on recency of access and
frequency of use.

Formula: activation_score = base_level + recency_boost + frequency_boost

- base_level: log of access count (how established a memory trace is)
- recency_boost: exponential decay based on time since last access
- frequency_boost: logarithmic scaling of total accesses (prevents runaway scores)

Scores are computed on retrieval as a property, never stored persistently.
This avoids stale activation data and keeps the graph schema clean.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


class ActivationModel:
    """Computes ACT-R activation scores for knowledge graph nodes.

    Activation scores determine retrieval priority: higher activation means
    the node is more likely to be relevant. Scores are computed on the fly
    from access metadata (access_count, last_accessed), never persisted.

    Args:
        decay_rate: Controls how fast recency decays. Higher values mean
            faster decay. Default 0.05 (half-life of ~14 days).
        recency_weight: Multiplier for the recency boost component.
            Default 1.0.
        frequency_weight: Multiplier for the frequency boost component.
            Default 1.0.
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        recency_weight: float = 1.0,
        frequency_weight: float = 1.0,
    ) -> None:
        self._decay_rate = decay_rate
        self._recency_weight = recency_weight
        self._frequency_weight = frequency_weight

    def activation_score(
        self,
        access_count: int,
        last_accessed: datetime | None,
    ) -> float:
        """Compute activation score for a node.

        activation = base_level + recency_boost + frequency_boost

        Args:
            access_count: Total number of times this node has been accessed.
            last_accessed: Timestamp of the most recent access. None means
                the node has never been accessed (minimum recency).

        Returns:
            Non-negative activation score as a float.
        """
        bl = self.base_level(access_count)
        rb = self.recency_boost(last_accessed)
        fb = self.frequency_boost(access_count)
        return bl + rb + fb

    def base_level(self, access_count: int) -> float:
        """Compute base-level activation from access count.

        Uses log(1 + access_count) so that:
        - 0 accesses -> 0.0 (log(1) = 0)
        - More accesses -> higher base level, but sublinearly

        Args:
            access_count: Total access count for the node.

        Returns:
            Non-negative base level value.
        """
        if access_count <= 0:
            return 0.0
        return math.log(1.0 + access_count)

    def recency_boost(self, last_accessed: datetime | None) -> float:
        """Compute recency boost using exponential decay.

        boost = recency_weight * exp(-decay_rate * days_since_access)

        Recent accesses get a boost close to recency_weight. As time passes,
        the boost decays exponentially toward zero.

        Args:
            last_accessed: Timestamp of last access. None returns 0.0.

        Returns:
            Non-negative recency boost value.
        """
        if last_accessed is None:
            return 0.0

        now = datetime.now(timezone.utc)

        # Ensure last_accessed is timezone-aware for comparison
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        delta = now - last_accessed
        days_elapsed = max(0.0, delta.total_seconds() / 86400.0)

        return self._recency_weight * math.exp(-self._decay_rate * days_elapsed)

    def frequency_boost(self, access_count: int) -> float:
        """Compute frequency boost proportional to access count.

        Uses logarithmic scaling: frequency_weight * log(1 + access_count)
        to prevent runaway scores for heavily accessed nodes while still
        rewarding frequent access.

        Args:
            access_count: Total access count for the node.

        Returns:
            Non-negative frequency boost value.
        """
        if access_count <= 0:
            return 0.0
        return self._frequency_weight * math.log(1.0 + access_count)

    def score_batch(
        self,
        nodes: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Compute activation scores for a batch of nodes.

        Each node dict must contain 'access_count' (int) and 'last_accessed'
        (datetime or None) keys. Additional keys are preserved.

        Args:
            nodes: List of node dicts with access metadata.

        Returns:
            List of (node_dict, score) tuples sorted by score descending.
        """
        scored: list[tuple[dict[str, Any], float]] = []
        for node in nodes:
            count = int(node.get("access_count", 0))
            last = node.get("last_accessed")
            score = self.activation_score(access_count=count, last_accessed=last)
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
