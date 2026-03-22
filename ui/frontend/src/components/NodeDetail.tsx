"use client";

import { useAtomValue } from "jotai";
import { dagNodesAtom } from "@/lib/store";

export function NodeDetail() {
  const nodes = useAtomValue(dagNodesAtom);

  if (nodes.length === 0) {
    return (
      <div style={{
        padding: "var(--space-6)",
        color: "var(--gray-8)",
        fontSize: "var(--text-sm)",
        fontFamily: "var(--font-sans)",
        textAlign: "center",
      }}>
        No nodes selected
      </div>
    );
  }

  return (
    <div style={{
      padding: "var(--space-4)",
      fontFamily: "var(--font-sans)",
      fontSize: "var(--text-base)",
    }}>
      <h3 style={{
        fontSize: "var(--text-sm)",
        fontWeight: 600,
        color: "var(--gray-9)",
        textTransform: "uppercase",
        letterSpacing: "0.5px",
        marginBottom: "var(--space-3)",
      }}>
        Node Detail
      </h3>
      <div style={{ color: "var(--gray-8)", fontSize: "var(--text-sm)" }}>
        Select a node in the DAG to view details.
      </div>
    </div>
  );
}
