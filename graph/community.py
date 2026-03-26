"""Community detection for subgraph clustering.

Uses a simple union-find (disjoint set) algorithm to detect connected components
in a retrieved subgraph. Each community is then summarized as a brief text
description listing its members and relationships.

No external dependencies -- pure Python implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommunityNode:
    """A node within a community.

    Attributes:
        node_id: Unique identifier of the node.
        table: The node table/type (e.g., "User", "Project").
        name: Human-readable name of the node.
        properties: Additional properties for display.
    """

    node_id: str
    table: str
    name: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CommunityEdge:
    """An edge within or between communities.

    Attributes:
        from_id: Source node ID.
        to_id: Target node ID.
        edge_type: The relationship type (e.g., "OWNS", "USES").
        description: Human-readable description (e.g., "Alice OWNS GraphBot").
    """

    from_id: str
    to_id: str
    edge_type: str
    description: str = ""


@dataclass(frozen=True)
class Community:
    """A detected community (connected component) in the subgraph.

    Attributes:
        community_id: Unique identifier for this community.
        nodes: Nodes belonging to this community.
        edges: Edges within this community.
        summary: Generated text summary of the community.
    """

    community_id: int
    nodes: tuple[CommunityNode, ...]
    edges: tuple[CommunityEdge, ...]
    summary: str = ""


class _UnionFind:
    """Disjoint set (union-find) with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root representative of the set containing x."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the sets containing x and y."""
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


class CommunityDetector:
    """Detects communities (connected components) in a subgraph.

    Uses union-find to group nodes connected by edges into communities,
    then generates a text summary for each community.
    """

    def detect_communities(
        self,
        nodes: list[CommunityNode],
        edges: list[CommunityEdge],
    ) -> list[Community]:
        """Detect connected components in the given nodes and edges.

        Args:
            nodes: All nodes in the subgraph.
            edges: All edges in the subgraph.

        Returns:
            List of Community objects, one per connected component.
            Communities are sorted by size descending (largest first).
        """
        if not nodes:
            return []

        uf = _UnionFind()
        node_map: dict[str, CommunityNode] = {}

        for node in nodes:
            uf.find(node.node_id)
            node_map[node.node_id] = node

        for edge in edges:
            if edge.from_id in node_map and edge.to_id in node_map:
                uf.union(edge.from_id, edge.to_id)

        # Group nodes by their root representative.
        groups: dict[str, list[CommunityNode]] = {}
        for node in nodes:
            root = uf.find(node.node_id)
            if root not in groups:
                groups[root] = []
            groups[root].append(node)

        # Group edges by community.
        edge_groups: dict[str, list[CommunityEdge]] = {}
        for edge in edges:
            if edge.from_id in node_map and edge.to_id in node_map:
                root = uf.find(edge.from_id)
                if root not in edge_groups:
                    edge_groups[root] = []
                edge_groups[root].append(edge)

        # Build Community objects sorted by size descending.
        communities: list[Community] = []
        sorted_roots = sorted(groups.keys(), key=lambda r: len(groups[r]), reverse=True)

        for idx, root in enumerate(sorted_roots):
            community_nodes = tuple(groups[root])
            community_edges = tuple(edge_groups.get(root, []))
            summary = self.summarize_community(community_nodes, community_edges)
            communities.append(Community(
                community_id=idx,
                nodes=community_nodes,
                edges=community_edges,
                summary=summary,
            ))

        return communities

    @staticmethod
    def summarize_community(
        nodes: tuple[CommunityNode, ...],
        edges: tuple[CommunityEdge, ...],
    ) -> str:
        """Generate a text summary for a community.

        The summary lists the types and names of members, followed by
        their relationships.

        Args:
            nodes: Nodes in the community.
            edges: Edges in the community.

        Returns:
            A concise text summary of the community.
        """
        if not nodes:
            return ""

        # Group nodes by type for a concise summary.
        type_groups: dict[str, list[str]] = {}
        for node in nodes:
            if node.table not in type_groups:
                type_groups[node.table] = []
            type_groups[node.table].append(node.name)

        parts: list[str] = []
        for node_type, names in type_groups.items():
            names_str = ", ".join(names)
            parts.append(f"{node_type}: {names_str}")

        member_section = "; ".join(parts)

        if edges:
            rel_descriptions = [e.description for e in edges if e.description]
            if rel_descriptions:
                rel_section = "; ".join(rel_descriptions)
                return f"Community of {len(nodes)} entities ({member_section}). Relationships: {rel_section}."

        return f"Community of {len(nodes)} entities ({member_section})."
