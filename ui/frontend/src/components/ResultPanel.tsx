"use client";

import { useAtomValue } from "jotai";
import { taskStateAtom } from "@/lib/store";
import { motion, AnimatePresence } from "framer-motion";

export function ResultPanel() {
  const taskState = useAtomValue(taskStateAtom);

  if (!taskState.result && !taskState.error) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: "auto" }}
        exit={{ opacity: 0, height: 0 }}
        style={{
          borderTop: "1px solid var(--border-primary)",
          maxHeight: 300,
          overflow: "auto",
        }}
      >
        <div style={{ padding: "12px 20px" }}>
          {taskState.error ? (
            <div style={{ color: "var(--status-failed)", fontFamily: "var(--font-mono)", fontSize: 13 }}>
              Error: {taskState.error}
            </div>
          ) : taskState.result ? (
            <div style={{ fontFamily: "var(--font-sans)", fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {taskState.result.output}
            </div>
          ) : null}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
