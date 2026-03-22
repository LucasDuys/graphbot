"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { useAtom, useAtomValue } from "jotai";
import * as d3 from "d3";
import { kgNodesAtom, kgEdgesAtom, kgPanelOpenAtom } from "@/lib/store";
import { getGraphEntities } from "@/lib/api";
import type { KGNode, KGEdge } from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Color mapping for node types                                       */
/* ------------------------------------------------------------------ */

const NODE_TYPE_COLORS: Record<string, string> = {
  User: "#2563EB",
  Project: "#16A34A",
  File: "#8B5CF6",
  Service: "#E5A100",
  Contact: "#EC4899",
  PatternNode: "#0891B2",
  Memory: "#F97316",
  Task: "#6366F1",
  Skill: "#14B8A6",
  ExecutionTree: "#DC2626",
};

const DEFAULT_NODE_COLOR = "#6B7280";

/* ------------------------------------------------------------------ */
/* D3 simulation types                                                */
/* ------------------------------------------------------------------ */

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  edgeType: string;
}

/* ------------------------------------------------------------------ */
/* Component                                                          */
/* ------------------------------------------------------------------ */

export function KnowledgeGraphPanel(): React.ReactElement | null {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  const [kgNodes, setKgNodes] = useAtom(kgNodesAtom);
  const [kgEdges, setKgEdges] = useAtom(kgEdgesAtom);
  const [isOpen, setIsOpen] = useAtom(kgPanelOpenAtom);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<SimNode | null>(null);

  /* ---------------------------------------------------------------- */
  /* Fetch graph data                                                 */
  /* ---------------------------------------------------------------- */

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getGraphEntities();
      setKgNodes(data.nodes);
      setKgEdges(data.edges);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load graph";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [setKgNodes, setKgEdges]);

  useEffect(() => {
    if (isOpen && kgNodes.length === 0) {
      fetchData();
    }
  }, [isOpen, kgNodes.length, fetchData]);

  /* ---------------------------------------------------------------- */
  /* D3 force layout                                                  */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    if (!isOpen || !svgRef.current || !containerRef.current) return;
    if (kgNodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight - 36; // subtract header

    svg.attr("width", width).attr("height", height);

    // Build simulation data
    const nodeMap = new Map<string, SimNode>();
    const simNodes: SimNode[] = kgNodes.map((n: KGNode) => {
      const node: SimNode = {
        id: n.id,
        label: (n.name as string) || n.id.slice(0, 8),
        type: n._type,
      };
      nodeMap.set(n.id, node);
      return node;
    });

    const simLinks: SimLink[] = kgEdges
      .filter((e: KGEdge) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e: KGEdge) => ({
        source: e.source,
        target: e.target,
        edgeType: e.type,
      }));

    // Zoom container
    const g = svg.append("g");

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        g.attr("transform", event.transform.toString());
      });

    svg.call(zoom);

    // Arrow markers for directed edges
    svg.append("defs").selectAll("marker")
      .data(["arrow"])
      .join("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -3 6 6")
      .attr("refX", 16)
      .attr("refY", 0)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-3L6,0L0,3")
      .attr("fill", "var(--gray-7)");

    // Links
    const link = g.append("g")
      .attr("stroke", "var(--gray-6)")
      .attr("stroke-opacity", 0.6)
      .selectAll<SVGLineElement, SimLink>("line")
      .data(simLinks)
      .join("line")
      .attr("stroke-width", 1)
      .attr("marker-end", "url(#arrow)");

    // Edge labels
    const edgeLabel = g.append("g")
      .selectAll<SVGTextElement, SimLink>("text")
      .data(simLinks)
      .join("text")
      .text((d: SimLink) => d.edgeType)
      .attr("font-size", 8)
      .attr("font-family", "var(--font-mono)")
      .attr("fill", "var(--gray-8)")
      .attr("text-anchor", "middle")
      .attr("dy", -3);

    // Nodes
    const node = g.append("g")
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(simNodes)
      .join("circle")
      .attr("r", 8)
      .attr("fill", (d: SimNode) => NODE_TYPE_COLORS[d.type] || DEFAULT_NODE_COLOR)
      .attr("stroke", "var(--gray-1)")
      .attr("stroke-width", 1.5)
      .attr("cursor", "grab")
      .on("mouseenter", (_event: MouseEvent, d: SimNode) => {
        setHoveredNode(d);
      })
      .on("mouseleave", () => {
        setHoveredNode(null);
      });

    // Node labels
    const nodeLabel = g.append("g")
      .selectAll<SVGTextElement, SimNode>("text")
      .data(simNodes)
      .join("text")
      .text((d: SimNode) => d.label)
      .attr("font-size", 9)
      .attr("font-family", "var(--font-sans)")
      .attr("fill", "var(--gray-10)")
      .attr("text-anchor", "middle")
      .attr("dy", -12);

    // Drag behavior
    const drag = d3.drag<SVGCircleElement, SimNode>()
      .on("start", (event: d3.D3DragEvent<SVGCircleElement, SimNode, SimNode>, d: SimNode) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event: d3.D3DragEvent<SVGCircleElement, SimNode, SimNode>, d: SimNode) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event: d3.D3DragEvent<SVGCircleElement, SimNode, SimNode>, d: SimNode) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    node.call(drag);

    // Force simulation
    const simulation = d3.forceSimulation<SimNode>(simNodes)
      .force("link", d3.forceLink<SimNode, SimLink>(simLinks).id((d: SimNode) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(20))
      .on("tick", () => {
        link
          .attr("x1", (d: SimLink) => (d.source as SimNode).x ?? 0)
          .attr("y1", (d: SimLink) => (d.source as SimNode).y ?? 0)
          .attr("x2", (d: SimLink) => (d.target as SimNode).x ?? 0)
          .attr("y2", (d: SimLink) => (d.target as SimNode).y ?? 0);

        edgeLabel
          .attr("x", (d: SimLink) => (((d.source as SimNode).x ?? 0) + ((d.target as SimNode).x ?? 0)) / 2)
          .attr("y", (d: SimLink) => (((d.source as SimNode).y ?? 0) + ((d.target as SimNode).y ?? 0)) / 2);

        node
          .attr("cx", (d: SimNode) => d.x ?? 0)
          .attr("cy", (d: SimNode) => d.y ?? 0);

        nodeLabel
          .attr("x", (d: SimNode) => d.x ?? 0)
          .attr("y", (d: SimNode) => d.y ?? 0);
      });

    simulationRef.current = simulation;

    return () => {
      simulation.stop();
      simulationRef.current = null;
    };
  }, [isOpen, kgNodes, kgEdges]);

  /* ---------------------------------------------------------------- */
  /* Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div
      ref={containerRef}
      style={{
        position: "fixed",
        bottom: isOpen ? 0 : -320,
        right: 0,
        width: 480,
        height: 360,
        background: "var(--gray-2)",
        border: "1px solid var(--border-color)",
        borderTopLeftRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-lg)",
        transition: "bottom var(--transition-slow)",
        zIndex: 50,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "var(--space-2) var(--space-3)",
          borderBottom: "1px solid var(--border-color)",
          background: "var(--gray-3)",
          cursor: "pointer",
          userSelect: "none",
          height: 36,
          flexShrink: 0,
        }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <span style={{
            fontSize: "var(--text-sm)",
            fontWeight: 600,
            color: "var(--gray-11)",
          }}>
            Knowledge Graph
          </span>
          <span style={{
            fontSize: "var(--text-xs)",
            fontFamily: "var(--font-mono)",
            color: "var(--gray-8)",
          }}>
            {kgNodes.length > 0 ? `${kgNodes.length} nodes` : ""}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          {isOpen && (
            <button
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                fetchData();
              }}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontSize: "var(--text-xs)",
                color: "var(--accent)",
                fontFamily: "var(--font-sans)",
                padding: "2px 6px",
                borderRadius: "var(--radius-sm)",
              }}
              title="Refresh graph data"
            >
              Refresh
            </button>
          )}
          <span style={{
            fontSize: "var(--text-sm)",
            color: "var(--gray-8)",
            transform: isOpen ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform var(--transition-default)",
            display: "inline-block",
          }}>
            ^
          </span>
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
        {loading && (
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "var(--text-sm)",
            color: "var(--gray-8)",
            background: "var(--gray-2)",
            zIndex: 2,
          }}>
            Loading graph...
          </div>
        )}

        {error && (
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "var(--text-sm)",
            color: "var(--status-failed)",
            background: "var(--gray-2)",
            zIndex: 2,
            gap: "var(--space-2)",
          }}>
            <span>{error}</span>
            <button
              onClick={fetchData}
              style={{
                background: "none",
                border: "1px solid var(--border-color)",
                borderRadius: "var(--radius-sm)",
                cursor: "pointer",
                fontSize: "var(--text-xs)",
                color: "var(--accent)",
                padding: "var(--space-1) var(--space-2)",
                fontFamily: "var(--font-sans)",
              }}
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && kgNodes.length === 0 && (
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "var(--text-sm)",
            color: "var(--gray-8)",
          }}>
            No entities in the knowledge graph yet
          </div>
        )}

        <svg
          ref={svgRef}
          style={{ width: "100%", height: "100%", display: "block" }}
        />

        {/* Hover tooltip */}
        {hoveredNode && (
          <div style={{
            position: "absolute",
            bottom: "var(--space-2)",
            left: "var(--space-2)",
            background: "var(--gray-3)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-sm)",
            padding: "var(--space-1) var(--space-2)",
            fontSize: "var(--text-xs)",
            fontFamily: "var(--font-mono)",
            color: "var(--gray-10)",
            pointerEvents: "none",
            zIndex: 3,
          }}>
            <span style={{ color: NODE_TYPE_COLORS[hoveredNode.type] || DEFAULT_NODE_COLOR, fontWeight: 600 }}>
              {hoveredNode.type}
            </span>
            {" / "}
            {hoveredNode.label}
          </div>
        )}

        {/* Legend */}
        {kgNodes.length > 0 && (
          <div style={{
            position: "absolute",
            top: "var(--space-2)",
            right: "var(--space-2)",
            display: "flex",
            flexDirection: "column",
            gap: 2,
            fontSize: "var(--text-xs)",
            color: "var(--gray-9)",
            pointerEvents: "none",
          }}>
            {Object.entries(NODE_TYPE_COLORS)
              .filter(([type]) => kgNodes.some((n: KGNode) => n._type === type))
              .map(([type, color]) => (
                <div key={type} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: color,
                    display: "inline-block",
                    flexShrink: 0,
                  }} />
                  <span>{type}</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
