"use client";

import { motion } from "framer-motion";

type EmptyStateProps = {
  phase: string;
};

export function EmptyState({ phase }: EmptyStateProps) {
  const isProcessing = !["idle", "complete", "error"].includes(phase);

  return (
    <div style={{
      width: "100%",
      height: "100%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexDirection: "column",
      gap: "var(--space-4)",
    }}>
      {/* Animated graph icon */}
      <motion.div
        animate={{ opacity: [0.4, 0.8, 0.4] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        style={{
          width: 64,
          height: 64,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="24" cy="8" r="4" fill="var(--gray-6)" />
          <circle cx="12" cy="28" r="4" fill="var(--gray-6)" />
          <circle cx="36" cy="28" r="4" fill="var(--gray-6)" />
          <circle cx="24" cy="42" r="4" fill="var(--gray-6)" />
          <line x1="22" y1="12" x2="14" y2="24" stroke="var(--gray-5)" strokeWidth="1.5" />
          <line x1="26" y1="12" x2="34" y2="24" stroke="var(--gray-5)" strokeWidth="1.5" />
          <line x1="14" y1="32" x2="22" y2="38" stroke="var(--gray-5)" strokeWidth="1.5" />
          <line x1="34" y1="32" x2="26" y2="38" stroke="var(--gray-5)" strokeWidth="1.5" />
        </svg>
      </motion.div>

      <div style={{
        textAlign: "center",
      }}>
        <div style={{
          fontSize: "var(--text-lg)",
          fontWeight: 500,
          color: "var(--gray-9)",
          marginBottom: "var(--space-2)",
        }}>
          {isProcessing ? "Building task graph..." : "No task running"}
        </div>
        <div style={{
          fontSize: "var(--text-sm)",
          color: "var(--gray-7)",
          maxWidth: 280,
          lineHeight: "var(--leading-normal)",
        }}>
          {isProcessing
            ? "Nodes will appear here as the DAG is constructed"
            : "Enter a task above to see the execution DAG visualized in real-time"
          }
        </div>
      </div>
    </div>
  );
}
