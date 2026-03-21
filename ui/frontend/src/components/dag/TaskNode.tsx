"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { motion } from "framer-motion";

type TaskNodeData = {
  label: string;
  status: "pending" | "running" | "completed" | "failed";
  domain: string;
  complexity: number;
  tokens?: number;
  latency_ms?: number;
  is_atomic: boolean;
};

type TaskNodeType = Node<TaskNodeData, "task">;

export const TaskNode = memo(function TaskNode({ data }: NodeProps<TaskNodeType>) {
  const statusColors: Record<string, string> = {
    pending: "var(--status-pending-bg)",
    running: "var(--status-running-bg)",
    completed: "var(--status-success-bg)",
    failed: "var(--status-failed-bg)",
  };

  const borderColors: Record<string, string> = {
    pending: "var(--status-pending)",
    running: "var(--status-running)",
    completed: "var(--status-success)",
    failed: "var(--status-failed)",
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      style={{
        background: statusColors[data.status] || statusColors.pending,
        border: `1.5px solid ${borderColors[data.status] || borderColors.pending}`,
        borderRadius: "var(--radius-md)",
        padding: "10px 14px",
        minWidth: 180,
        fontFamily: "var(--font-sans)",
        fontSize: 13,
      }}
    >
      <Handle type="target" position={Position.Top} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{data.label}</span>
        <span style={{
          fontSize: 10,
          fontWeight: 700,
          textTransform: "uppercase",
          color: borderColors[data.status],
          letterSpacing: "0.5px",
        }}>
          {data.status}
        </span>
      </div>
      <div style={{ display: "flex", gap: 8, fontSize: 11, color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
        <span>{data.domain}</span>
        {data.is_atomic && <span>atomic</span>}
        {data.tokens !== undefined && <span>{data.tokens} tok</span>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </motion.div>
  );
});
