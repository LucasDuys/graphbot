"use client";

import { useState } from "react";
import { useAtomValue } from "jotai";
import { dagNodesAtom } from "@/lib/store";

export function NodeDetail() {
  const nodes = useAtomValue(dagNodesAtom);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selected = nodes.find((n) => n.id === selectedId);

  if (nodes.length === 0) {
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
        Node details will appear here
      </div>
    );
  }

  if (!selected) {
    return (
      <div style={{ padding: "var(--space-4)" }}>
        <div style={{ fontSize: "var(--text-sm)", fontWeight: 600, marginBottom: "var(--space-3)", color: "var(--gray-11)" }}>
          Nodes ({nodes.length})
        </div>
        {nodes.map((node) => (
          <div
            key={node.id}
            onClick={() => setSelectedId(node.id)}
            style={{
              padding: "var(--space-2) var(--space-3)",
              marginBottom: "var(--space-1)",
              borderRadius: "var(--radius-sm)",
              cursor: "pointer",
              fontSize: "var(--text-sm)",
              color: "var(--gray-10)",
              background: "transparent",
              transition: "background var(--transition-fast)",
            }}
            onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "var(--gray-3)"; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontWeight: 500 }}>{node.label}</span>
              <span style={{
                fontSize: "var(--text-xs)",
                fontFamily: "var(--font-mono)",
                color: node.status === "completed" ? "var(--status-success)" : node.status === "running" ? "var(--status-running)" : "var(--gray-8)",
              }}>
                {node.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div style={{ padding: "var(--space-4)" }}>
      <button
        onClick={() => setSelectedId(null)}
        style={{
          fontSize: "var(--text-xs)",
          color: "var(--accent)",
          background: "none",
          border: "none",
          cursor: "pointer",
          marginBottom: "var(--space-3)",
          fontFamily: "var(--font-sans)",
        }}
      >
        Back to list
      </button>

      <h3 style={{ fontSize: "var(--text-lg)", fontWeight: 600, marginBottom: "var(--space-3)" }}>{selected.label}</h3>

      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "var(--space-1) var(--space-3)", fontSize: "var(--text-sm)" }}>
        <span style={{ color: "var(--gray-8)" }}>Domain</span>
        <span style={{ fontFamily: "var(--font-mono)" }}>{selected.domain}</span>
        <span style={{ color: "var(--gray-8)" }}>Status</span>
        <span>{selected.status}</span>
        <span style={{ color: "var(--gray-8)" }}>Atomic</span>
        <span>{selected.is_atomic ? "Yes" : "No"}</span>
        {selected.tokens !== undefined && <>
          <span style={{ color: "var(--gray-8)" }}>Tokens</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>{selected.tokens}</span>
        </>}
      </div>

      {selected.output && (
        <div style={{ marginTop: "var(--space-4)" }}>
          <div style={{ fontSize: "var(--text-xs)", fontWeight: 600, color: "var(--gray-9)", marginBottom: "var(--space-2)" }}>OUTPUT</div>
          <pre style={{
            padding: "var(--space-3)",
            background: "var(--gray-1)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-sm)",
            fontSize: "var(--text-xs)",
            fontFamily: "var(--font-mono)",
            overflow: "auto",
            maxHeight: 200,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}>
            {selected.output}
          </pre>
        </div>
      )}
    </div>
  );
}
