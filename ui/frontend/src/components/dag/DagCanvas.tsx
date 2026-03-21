"use client";

import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { TaskNode } from "./TaskNode";

const nodeTypes: NodeTypes = {
  task: TaskNode,
};

export function DagCanvas() {
  const [nodes, , onNodesChange] = useNodesState([]);
  const [edges, , onEdgesChange] = useEdgesState([]);

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
              pending: "var(--status-pending)",
              running: "var(--status-running)",
              completed: "var(--status-success)",
              failed: "var(--status-failed)",
            };
            return colors[status] || colors.pending;
          }}
        />
      </ReactFlow>
    </div>
  );
}
