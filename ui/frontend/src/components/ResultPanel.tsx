"use client";

import { useAtomValue } from "jotai";
import { taskStateAtom } from "@/lib/store";
import { motion, AnimatePresence } from "framer-motion";

export function ResultPanel() {
  const taskState = useAtomValue(taskStateAtom);

  if (!taskState.result && !taskState.error) {
    return (
      <div style={{
        padding: "var(--space-6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        color: "var(--gray-8)",
        fontSize: "var(--text-sm)",
      }}>
        Results will appear here
      </div>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        style={{ padding: "var(--space-4)", height: "100%", overflow: "auto" }}
      >
        {taskState.error ? (
          <div style={{
            padding: "var(--space-3)",
            background: "var(--status-failed-bg)",
            borderLeft: "3px solid var(--status-failed)",
            borderRadius: "var(--radius-sm)",
            fontSize: "var(--text-sm)",
            fontFamily: "var(--font-mono)",
            color: "var(--status-failed)",
          }}>
            {taskState.error}
          </div>
        ) : taskState.result ? (
          <>
            {/* Metrics bar */}
            <div style={{
              display: "flex",
              gap: "var(--space-3)",
              marginBottom: "var(--space-3)",
              flexWrap: "wrap",
            }}>
              {[
                { label: "Nodes", value: taskState.result.total_nodes },
                { label: "Tokens", value: taskState.result.total_tokens.toLocaleString() },
                { label: "Latency", value: `${(taskState.result.total_latency_ms / 1000).toFixed(1)}s` },
                { label: "Cost", value: `$${taskState.result.total_cost.toFixed(6)}` },
              ].map((m) => (
                <div key={m.label} style={{
                  padding: "var(--space-1) var(--space-2)",
                  background: "var(--gray-3)",
                  borderRadius: "var(--radius-sm)",
                  fontSize: "var(--text-xs)",
                  fontFamily: "var(--font-mono)",
                }}>
                  <span style={{ color: "var(--gray-8)" }}>{m.label} </span>
                  <span style={{ fontWeight: 600 }}>{m.value}</span>
                </div>
              ))}
            </div>

            {/* Output */}
            <div style={{
              padding: "var(--space-3)",
              background: "var(--gray-1)",
              border: "1px solid var(--border-color)",
              borderRadius: "var(--radius-md)",
              fontSize: "var(--text-sm)",
              lineHeight: "var(--leading-relaxed)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}>
              {taskState.result.output}
            </div>
          </>
        ) : null}
      </motion.div>
    </AnimatePresence>
  );
}
