"use client";

import { useAtomValue } from "jotai";
import { taskStateAtom, dagNodesAtom } from "@/lib/store";
import { motion, AnimatePresence } from "framer-motion";

const phaseLabels: Record<string, string> = {
  idle: "Ready",
  intake: "Classifying intent",
  decomposing: "Decomposing task",
  executing: "Executing nodes",
  aggregating: "Aggregating results",
  complete: "Complete",
  error: "Error",
};

export function StatusBar() {
  const taskState = useAtomValue(taskStateAtom);
  const nodes = useAtomValue(dagNodesAtom);

  return (
    <div style={{
      padding: "8px 20px",
      borderBottom: "1px solid var(--border-primary)",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      fontSize: 12,
      fontFamily: "var(--font-mono)",
      color: "var(--text-secondary)",
    }}>
      <AnimatePresence mode="wait">
        <motion.span
          key={taskState.phase}
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 4 }}
          transition={{ duration: 0.2 }}
        >
          {phaseLabels[taskState.phase] || taskState.phase}
          {taskState.intake && ` [${taskState.intake.domain}, complexity ${taskState.intake.complexity}]`}
        </motion.span>
      </AnimatePresence>
      <span>
        {nodes.length > 0 && `${nodes.length} nodes`}
        {taskState.result && (
          <> | {taskState.result.total_tokens} tok | {taskState.result.total_latency_ms.toFixed(0)}ms | ${taskState.result.total_cost.toFixed(6)}</>
        )}
      </span>
    </div>
  );
}
