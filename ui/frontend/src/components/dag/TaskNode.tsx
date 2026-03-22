"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { motion } from "framer-motion";

type TaskNodeData = {
  label: string;
  status: "pending" | "running" | "completed" | "failed";
  domain: string;
  complexity: number;
  is_atomic: boolean;
  tokens?: number;
  latency_ms?: number;
};

const STATUS_COLORS: Record<string, { border: string; bg: string; text: string }> = {
  pending: { border: "var(--status-idle)", bg: "var(--gray-2)", text: "var(--status-idle)" },
  running: { border: "var(--status-running)", bg: "var(--status-running-bg)", text: "var(--status-running)" },
  completed: { border: "var(--status-success)", bg: "var(--gray-2)", text: "var(--status-success)" },
  failed: { border: "var(--status-failed)", bg: "var(--status-failed-bg)", text: "var(--status-failed)" },
};

const DOMAIN_LABELS: Record<string, string> = {
  file: "FILE",
  web: "WEB",
  code: "CODE",
  system: "LLM",
  synthesis: "AGG",
  comms: "COMMS",
};

export const TaskNode = memo(function TaskNode({ data }: NodeProps) {
  const d = data as TaskNodeData;
  const colors = STATUS_COLORS[d.status] || STATUS_COLORS.pending;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      style={{
        background: colors.bg,
        borderLeft: `3px solid ${colors.border}`,
        borderTop: "1px solid var(--border-color)",
        borderRight: "1px solid var(--border-color)",
        borderBottom: "1px solid var(--border-color)",
        borderRadius: "var(--radius-md)",
        padding: "var(--space-2) var(--space-3)",
        minWidth: 200,
        maxWidth: 260,
        boxShadow: d.status === "running" ? "var(--shadow-md)" : "var(--shadow-sm)",
        fontFamily: "var(--font-sans)",
        transition: "box-shadow var(--transition-default), border-color var(--transition-default)",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: "var(--gray-6)", width: 6, height: 6 }} />

      {/* Header: label */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "var(--space-1)" }}>
        <span style={{
          fontSize: "var(--text-sm)",
          fontWeight: 500,
          color: "var(--gray-11)",
          lineHeight: "var(--leading-tight)",
          flex: 1,
          overflow: "hidden",
          textOverflow: "ellipsis",
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
        }}>
          {d.label}
        </span>
      </div>

      {/* Footer: domain badge + metrics */}
      <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginTop: "var(--space-1)" }}>
        <span style={{
          fontSize: "var(--text-xs)",
          fontFamily: "var(--font-mono)",
          fontWeight: 500,
          color: colors.text,
          padding: "1px 5px",
          background: d.status === "pending" ? "var(--gray-3)" : undefined,
          borderRadius: "var(--radius-sm)",
        }}>
          {DOMAIN_LABELS[d.domain] || d.domain}
        </span>
        {d.tokens !== undefined && d.tokens > 0 && (
          <span style={{ fontSize: "var(--text-xs)", color: "var(--gray-8)", fontFamily: "var(--font-mono)" }}>
            {d.tokens} tok
          </span>
        )}
        {d.latency_ms !== undefined && d.latency_ms > 0 && (
          <span style={{ fontSize: "var(--text-xs)", color: "var(--gray-8)", fontFamily: "var(--font-mono)" }}>
            {(d.latency_ms / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} style={{ background: "var(--gray-6)", width: 6, height: 6 }} />
    </motion.div>
  );
});
