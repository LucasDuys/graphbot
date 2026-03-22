"""PPR-based retrieval via power iteration for context assembly.

Implements Personalized PageRank (PPR) to spread activation from seed entities
through the knowledge graph, enabling retrieval of relevant nodes far beyond
the standard 2-hop traversal. Uses power iteration with no external graph
library dependencies -- only the existing GraphStore Cypher interface.

Typical usage:
    retriever = PPRRetriever(store=graph_store)
    results = retriever.retrieve(seed_ids=["entity_1", "entity_2"], top_k=10)
    # results: [(node_id, ppr_score), ...] sorted by score descending
"""

from __future__ import annotations

import logging
from collections import defaultdict

from graph.schema import EDGE_TYPES, NODE_TYPES
from graph.store import GraphStore

logger = logging.getLogger(__name__)

# Default PPR parameters
_DEFAULT_ALPHA: float = 0.85
_DEFAULT_MAX_ITERATIONS: int = 20
_DEFAULT_CONVERGENCE_THRESHOLD: float = 1e-6


class PPRRetriever:
    """Approximate Personalized PageRank retrieval using power iteration.

    Spreads activation from seed nodes through the knowledge graph to discover
    relevant nodes that a fixed-hop traversal would miss. Returns top-K nodes
    ranked by PPR score, respecting an optional token budget.

    The algorithm:
    1. Build an adjacency list from the graph (undirected -- edges flow both ways)
    2. Initialize the probability vector with uniform weight on seed nodes
    3. Iterate: p(t+1) = alpha * M * p(t) + (1 - alpha) * seed_vector
       where M is the column-stochastic transition matrix
    4. Stop when L1 norm of change < convergence_threshold or max_iterations hit
    5. Return top-K nodes by PPR score, filtered by token budget

    Args:
        store: The GraphStore to query for graph structure.
        alpha: Damping factor / probability of following an edge (vs teleporting
            back to seed). Higher values explore further. Default 0.85.
        max_iterations: Maximum number of power iteration steps. Default 20.
        convergence_threshold: L1 norm convergence criterion. Default 1e-6.
    """

    def __init__(
        self,
        store: GraphStore,
        alpha: float = _DEFAULT_ALPHA,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        convergence_threshold: float = _DEFAULT_CONVERGENCE_THRESHOLD,
    ) -> None:
        self._store = store
        self.alpha: float = alpha
        self.max_iterations: int = max_iterations
        self.convergence_threshold: float = convergence_threshold

    def _build_adjacency(self) -> tuple[dict[str, set[str]], set[str]]:
        """Build an undirected adjacency list from all edges in the graph.

        Returns:
            (adjacency, all_node_ids) where adjacency maps each node ID to its
            set of neighbor IDs (both directions), and all_node_ids is the
            complete set of node IDs in the graph.
        """
        adjacency: dict[str, set[str]] = defaultdict(set)
        all_node_ids: set[str] = set()

        # Collect all node IDs first
        for node_type in NODE_TYPES:
            rows = self._store.query(f"MATCH (n:{node_type.name}) RETURN n.id")
            for row in rows:
                node_id = str(row["n.id"])
                all_node_ids.add(node_id)

        # Build adjacency from edges (treat as undirected for PPR spread)
        for edge_type in EDGE_TYPES:
            cypher = (
                f"MATCH (a:{edge_type.from_type})-[:{edge_type.name}]->"
                f"(b:{edge_type.to_type}) RETURN a.id, b.id"
            )
            try:
                rows = self._store.query(cypher)
                for row in rows:
                    from_id = str(row["a.id"])
                    to_id = str(row["b.id"])
                    adjacency[from_id].add(to_id)
                    adjacency[to_id].add(from_id)
            except Exception:
                # Edge table may be empty or query may fail; skip gracefully
                logger.debug(
                    "Skipping edge type %s during adjacency build", edge_type.name
                )

        return adjacency, all_node_ids

    def _power_iteration(
        self,
        seed_ids: list[str],
        adjacency: dict[str, set[str]],
        all_node_ids: set[str],
    ) -> dict[str, float]:
        """Run power iteration to compute approximate PPR scores.

        Args:
            seed_ids: Node IDs to use as teleport targets (personalization).
            adjacency: Undirected adjacency list.
            all_node_ids: All node IDs in the graph.

        Returns:
            Dict mapping node_id -> PPR score. Scores sum to approximately 1.0.
        """
        if not seed_ids or not all_node_ids:
            return {}

        # Filter seeds to only those that exist in the graph
        valid_seeds = [sid for sid in seed_ids if sid in all_node_ids]
        if not valid_seeds:
            return {}

        num_seeds = len(valid_seeds)

        # Seed vector: uniform distribution over seed nodes
        seed_vector: dict[str, float] = {}
        for sid in valid_seeds:
            seed_vector[sid] = 1.0 / num_seeds

        # Initialize PPR vector to the seed vector
        ppr: dict[str, float] = dict(seed_vector)

        alpha = self.alpha

        for iteration in range(self.max_iterations):
            new_ppr: dict[str, float] = {}

            # Compute dangling node mass: probability held by nodes with no neighbors.
            # In standard PageRank, dangling nodes redistribute their mass to the
            # personalization vector (seed nodes) rather than losing it.
            dangling_mass = 0.0
            for node_id in all_node_ids:
                if not adjacency.get(node_id):
                    dangling_mass += ppr.get(node_id, 0.0)

            # Spread: for each node, distribute its score to neighbors
            # p_new[v] = alpha * (sum(p[u]/deg(u) for u in neighbors(v)) + dangling * seed[v])
            #          + (1 - alpha) * seed[v]
            for node_id in all_node_ids:
                spread_score = 0.0
                neighbors = adjacency.get(node_id, set())
                for neighbor_id in neighbors:
                    neighbor_degree = len(adjacency.get(neighbor_id, set()))
                    if neighbor_degree > 0:
                        spread_score += ppr.get(neighbor_id, 0.0) / neighbor_degree

                # Dangling mass redistributed proportionally to seed vector
                dangling_contribution = dangling_mass * seed_vector.get(node_id, 0.0)

                teleport_score = seed_vector.get(node_id, 0.0)
                new_ppr[node_id] = (
                    alpha * (spread_score + dangling_contribution)
                    + (1.0 - alpha) * teleport_score
                )

            # Check convergence: L1 norm of change
            l1_diff = 0.0
            for node_id in all_node_ids:
                l1_diff += abs(new_ppr.get(node_id, 0.0) - ppr.get(node_id, 0.0))

            ppr = new_ppr

            if l1_diff < self.convergence_threshold:
                logger.debug(
                    "PPR converged after %d iterations (L1 diff: %.2e)",
                    iteration + 1,
                    l1_diff,
                )
                break

        return ppr

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for a text string.

        Uses the same heuristic as GraphStore: tokens ~= word_count * 1.3.
        """
        return max(1, int(len(text.split()) * 1.3))

    def _get_node_text(self, node_id: str) -> str:
        """Get a text representation of a node for token estimation.

        Searches all node tables for the node and concatenates its properties.
        """
        for node_type in NODE_TYPES:
            node = self._store.get_node(node_type.name, node_id)
            if node is not None:
                parts: list[str] = []
                for key, value in node.items():
                    if value is not None and key != "id":
                        parts.append(str(value))
                return " ".join(parts) if parts else node_id
        return node_id

    def retrieve(
        self,
        seed_ids: list[str],
        top_k: int = 10,
        max_tokens: int | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top-K nodes ranked by Personalized PageRank score.

        Seeds the PPR computation from the given entity IDs, spreads activation
        through the graph via power iteration, and returns the highest-scoring
        nodes. Optionally respects a token budget.

        Args:
            seed_ids: Entity IDs to seed the PPR from (e.g., from entity
                resolution). These receive the teleport probability mass.
            top_k: Maximum number of results to return.
            max_tokens: Optional token budget. When set, stops adding nodes
                once the cumulative estimated token cost exceeds this limit.
                None means no token budget constraint (only top_k applies).

        Returns:
            List of (node_id, ppr_score) tuples sorted by score descending.
            Scores are in (0, 1] and sum to approximately 1.0 across all graph
            nodes (only the top-K slice is returned).
        """
        if not seed_ids:
            return []

        if max_tokens is not None and max_tokens <= 0:
            return []

        # Build adjacency from graph
        adjacency, all_node_ids = self._build_adjacency()
        if not all_node_ids:
            return []

        # Run power iteration
        ppr_scores = self._power_iteration(seed_ids, adjacency, all_node_ids)
        if not ppr_scores:
            return []

        # Sort by score descending, filter out zero-score nodes
        ranked = sorted(
            ((nid, score) for nid, score in ppr_scores.items() if score > 0.0),
            key=lambda x: x[1],
            reverse=True,
        )

        # Apply top_k limit
        ranked = ranked[:top_k]

        # Apply token budget if specified
        if max_tokens is not None:
            budget_filtered: list[tuple[str, float]] = []
            used_tokens = 0
            for node_id, score in ranked:
                node_text = self._get_node_text(node_id)
                cost = self._estimate_tokens(node_text)
                if used_tokens + cost > max_tokens:
                    break
                used_tokens += cost
                budget_filtered.append((node_id, score))
            return budget_filtered

        return ranked
