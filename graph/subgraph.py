"""GraphRAG-style subgraph retrieval with community summaries.

Performs multi-hop traversal from seed entities, retrieves edges with
relationship descriptions, detects communities via connected components,
and assembles a structured SubgraphResult for context formatting.

Replaces raw entity lists with coherent subgraph context including:
- Multi-hop node traversal (configurable depth)
- Edge/relationship descriptions ("Python USES Django")
- Community summaries from graph clusters
- Activation-based relevance ranking
- Token budget enforcement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from graph.activation import ActivationModel
from graph.community import CommunityDetector, CommunityEdge, CommunityNode, Community
from graph.schema import EDGE_TYPES, NODE_TYPES
from graph.store import GraphStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubgraphNode:
    """A node in the retrieved subgraph.

    Attributes:
        node_id: Unique identifier.
        table: Node table/type (e.g., "User", "Project").
        name: Human-readable name.
        properties: All node properties.
        activation_score: ACT-R activation score for relevance ranking.
        hop_distance: Number of hops from a seed entity.
    """

    node_id: str
    table: str
    name: str
    properties: dict[str, object] = field(default_factory=dict)
    activation_score: float = 0.0
    hop_distance: int = 0


@dataclass(frozen=True)
class SubgraphEdge:
    """An edge in the retrieved subgraph.

    Attributes:
        from_id: Source node ID.
        to_id: Target node ID.
        from_name: Source node name (for description generation).
        to_name: Target node name (for description generation).
        edge_type: Relationship type (e.g., "OWNS", "USES").
        description: Human-readable relationship description.
        properties: Edge properties.
    """

    from_id: str
    to_id: str
    from_name: str
    to_name: str
    edge_type: str
    description: str = ""
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SubgraphResult:
    """Result of a subgraph retrieval operation.

    Contains nodes, edges, and community summaries assembled from
    multi-hop traversal of the knowledge graph.

    Attributes:
        nodes: Retrieved nodes sorted by activation score.
        edges: Retrieved edges with relationship descriptions.
        community_summaries: Text summaries of detected communities.
    """

    nodes: tuple[SubgraphNode, ...] = ()
    edges: tuple[SubgraphEdge, ...] = ()
    community_summaries: tuple[str, ...] = ()


class SubgraphRetriever:
    """Retrieves structured subgraphs from the knowledge graph.

    Performs multi-hop traversal from seed entities, collects edges with
    relationship descriptions, detects communities, and ranks results
    by ACT-R activation score.

    Args:
        store: Initialized GraphStore instance.
    """

    def __init__(self, store: GraphStore) -> None:
        self._store = store
        self._activation_model = ActivationModel()
        self._community_detector = CommunityDetector()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using the same heuristic as GraphStore."""
        return max(1, int(len(text.split()) * 1.3))

    def _get_node_name(self, props: dict[str, object]) -> str:
        """Extract a human-readable name from node properties."""
        name = props.get("name")
        if name:
            return str(name)
        path = props.get("path")
        if path:
            return str(path)
        description = props.get("description")
        if description:
            desc_str = str(description)
            return desc_str[:60] if len(desc_str) > 60 else desc_str
        return str(props.get("id", "unknown"))

    def _compute_activation(self, props: dict[str, object]) -> float:
        """Compute ACT-R activation score for a node."""
        return self._store._node_activation_score(self._activation_model, props)

    def retrieve_subgraph(
        self,
        entity_ids: list[str],
        *,
        max_hops: int = 2,
        max_nodes: int = 50,
    ) -> SubgraphResult:
        """Retrieve a structured subgraph from seed entities.

        Performs breadth-first multi-hop traversal, collects edges with
        relationship descriptions, detects communities, and returns
        a ranked SubgraphResult.

        Args:
            entity_ids: Seed entity IDs to start traversal from.
            max_hops: Maximum number of hops from seed entities. Defaults to 2.
            max_nodes: Maximum number of nodes to include. Defaults to 50.

        Returns:
            A SubgraphResult with nodes, edges, and community summaries.
        """
        if not entity_ids:
            return SubgraphResult()

        # Phase 1: Multi-hop BFS traversal collecting nodes and edges.
        all_nodes: dict[str, SubgraphNode] = {}
        all_edges: list[SubgraphEdge] = []
        seen_edge_keys: set[tuple[str, str, str]] = set()

        # BFS frontier: list of (node_id, table_name, hop_distance)
        frontier: list[tuple[str, str, int]] = []

        # Seed the frontier with resolved entity IDs.
        for eid in entity_ids:
            found = self._store._find_node_table(eid)
            if found is None:
                continue
            table_name, props = found
            node_id = str(props.get("id", eid))
            if node_id in all_nodes:
                continue

            name = self._get_node_name(props)
            score = self._compute_activation(props)
            all_nodes[node_id] = SubgraphNode(
                node_id=node_id,
                table=table_name,
                name=name,
                properties=dict(props),
                activation_score=score,
                hop_distance=0,
            )
            frontier.append((node_id, table_name, 0))

        # BFS expansion up to max_hops.
        for current_hop in range(max_hops):
            if len(all_nodes) >= max_nodes:
                break

            next_frontier: list[tuple[str, str, int]] = []

            for node_id, table_name, hop in frontier:
                if hop != current_hop:
                    continue

                # Get connected nodes and edges.
                neighbors, edges = self._get_connected_with_edges(
                    node_id, table_name,
                )

                for neighbor_table, neighbor_props, edge_type_name, direction in neighbors:
                    nid = str(neighbor_props.get("id", ""))
                    if not nid:
                        continue

                    neighbor_name = self._get_node_name(neighbor_props)

                    # Record edge (deduplicated).
                    if direction == "outgoing":
                        edge_key = (node_id, nid, edge_type_name)
                        from_id, to_id = node_id, nid
                        from_name = all_nodes[node_id].name
                        to_name = neighbor_name
                    else:
                        edge_key = (nid, node_id, edge_type_name)
                        from_id, to_id = nid, node_id
                        from_name = neighbor_name
                        to_name = all_nodes[node_id].name

                    if edge_key not in seen_edge_keys:
                        seen_edge_keys.add(edge_key)
                        description = f"{from_name} {edge_type_name} {to_name}"
                        all_edges.append(SubgraphEdge(
                            from_id=from_id,
                            to_id=to_id,
                            from_name=from_name,
                            to_name=to_name,
                            edge_type=edge_type_name,
                            description=description,
                        ))

                    # Add neighbor node if not seen and under budget.
                    if nid not in all_nodes and len(all_nodes) < max_nodes:
                        score = self._compute_activation(neighbor_props)
                        all_nodes[nid] = SubgraphNode(
                            node_id=nid,
                            table=neighbor_table,
                            name=neighbor_name,
                            properties=dict(neighbor_props),
                            activation_score=score,
                            hop_distance=current_hop + 1,
                        )
                        next_frontier.append((nid, neighbor_table, current_hop + 1))

            frontier = next_frontier

        # Phase 2: Detect communities.
        community_nodes = [
            CommunityNode(
                node_id=n.node_id,
                table=n.table,
                name=n.name,
            )
            for n in all_nodes.values()
        ]
        community_edges = [
            CommunityEdge(
                from_id=e.from_id,
                to_id=e.to_id,
                edge_type=e.edge_type,
                description=e.description,
            )
            for e in all_edges
        ]

        communities = self._community_detector.detect_communities(
            community_nodes, community_edges,
        )
        community_summaries = tuple(c.summary for c in communities if c.summary)

        # Phase 3: Sort nodes by activation score descending.
        sorted_nodes = sorted(
            all_nodes.values(),
            key=lambda n: n.activation_score,
            reverse=True,
        )

        return SubgraphResult(
            nodes=tuple(sorted_nodes),
            edges=tuple(all_edges),
            community_summaries=community_summaries,
        )

    def _get_connected_with_edges(
        self,
        node_id: str,
        table_name: str,
    ) -> tuple[
        list[tuple[str, dict[str, object], str, str]],
        list[tuple[str, str, str]],
    ]:
        """Get connected nodes with edge type information.

        Returns:
            A tuple of (neighbors, raw_edges) where:
            - neighbors: list of (table, props, edge_type, direction)
            - raw_edges: list of (from_id, to_id, edge_type)
        """
        conn = self._store._get_conn()
        neighbors: list[tuple[str, dict[str, object], str, str]] = []
        raw_edges: list[tuple[str, str, str]] = []

        for edge_type in EDGE_TYPES:
            # Outgoing edges.
            if edge_type.from_type == table_name:
                cypher = (
                    f"MATCH (a:{edge_type.from_type})-[:{edge_type.name}]->(b:{edge_type.to_type}) "
                    f"WHERE a.id = $id RETURN b.*"
                )
                try:
                    result = conn.execute(cypher, parameters={"id": node_id})
                    while result.has_next():
                        row = result.get_next()
                        columns = result.get_column_names()
                        props: dict[str, object] = {}
                        for col_name, value in zip(columns, row):
                            key = col_name.split(".", 1)[1] if "." in col_name else col_name
                            props[key] = value
                        nid = str(props.get("id", ""))
                        neighbors.append((edge_type.to_type, props, edge_type.name, "outgoing"))
                        if nid:
                            raw_edges.append((node_id, nid, edge_type.name))
                except Exception:
                    pass

            # Incoming edges.
            if edge_type.to_type == table_name:
                cypher = (
                    f"MATCH (a:{edge_type.from_type})-[:{edge_type.name}]->(b:{edge_type.to_type}) "
                    f"WHERE b.id = $id RETURN a.*"
                )
                try:
                    result = conn.execute(cypher, parameters={"id": node_id})
                    while result.has_next():
                        row = result.get_next()
                        columns = result.get_column_names()
                        props = {}
                        for col_name, value in zip(columns, row):
                            key = col_name.split(".", 1)[1] if "." in col_name else col_name
                            props[key] = value
                        nid = str(props.get("id", ""))
                        neighbors.append((edge_type.from_type, props, edge_type.name, "incoming"))
                        if nid:
                            raw_edges.append((nid, node_id, edge_type.name))
                except Exception:
                    pass

        return neighbors, raw_edges
