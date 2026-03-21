"use client";

import { useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useAtomValue } from "jotai";
import { dagNodesAtom, dagEdgesAtom } from "@/lib/store";
import type { DagNodeData } from "@/lib/store";
import { TaskNode } from "./TaskNode";
import ELK from "elkjs/lib/elk.bundled.js";

const elk = new ELK();
const nodeTypes: NodeTypes = { task: TaskNode };

async function layoutNodes(
  dagNodes: DagNodeData[],
  dagEdges: { source: string; target: string; label?: string }[],
): Promise<{ nodes: Node[]; edges: Edge[] }> {
  if (dagNodes.length === 0) return { nodes: [], edges: [] };

  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.spacing.nodeNode": "60",
      "elk.layered.spacing.nodeNodeBetweenLayers": "80",
    },
    children: dagNodes.map((n) => ({
      id: n.id,
      width: 220,
      height: 70,
    })),
    edges: dagEdges.map((e, i) => ({
      id: `e${i}`,
      sources: [e.source],
      targets: [e.target],
    })),
  };

  const laid = await elk.layout(graph);

  const nodes: Node[] = (laid.children || []).map((c) => {
    const data = dagNodes.find((n) => n.id === c.id);
    return {
      id: c.id,
      type: "task",
      position: { x: c.x || 0, y: c.y || 0 },
      data: data || {},
    };
  });

  const edges: Edge[] = dagEdges.map((e, i) => ({
    id: `e${i}`,
    source: e.source,
    target: e.target,
    label: e.label,
    animated: true,
    style: { stroke: "var(--accent-data-flow)" },
  }));

  return { nodes, edges };
}

export function DagCanvas() {
  const dagNodes = useAtomValue(dagNodesAtom);
  const dagEdges = useAtomValue(dagEdgesAtom);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    layoutNodes(dagNodes, dagEdges).then(({ nodes: n, edges: e }) => {
      setNodes(n);
      setEdges(e);
    });
  }, [dagNodes, dagEdges, setNodes, setEdges]);

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background color="var(--border-subtle)" gap={20} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(n) => {
            const status = (n.data?.status as string) || "pending";
            const colors: Record<string, string> = {
              pending: "#94a3b8",
              running: "#f59e0b",
              completed: "#22c55e",
              failed: "#ef4444",
            };
            return colors[status] || colors.pending;
          }}
        />
      </ReactFlow>
    </div>
  );
}
