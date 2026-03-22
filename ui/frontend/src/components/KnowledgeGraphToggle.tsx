"use client";

import { useAtom } from "jotai";
import { kgPanelOpenAtom } from "@/lib/store";

export function KnowledgeGraphToggle(): React.ReactElement {
  const [isOpen, setIsOpen] = useAtom(kgPanelOpenAtom);

  return (
    <button
      onClick={() => setIsOpen(!isOpen)}
      title={isOpen ? "Hide knowledge graph" : "Show knowledge graph"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--space-1)",
        padding: "var(--space-1) var(--space-2)",
        background: isOpen ? "var(--accent-soft)" : "transparent",
        border: `1px solid ${isOpen ? "var(--accent)" : "var(--border-color)"}`,
        borderRadius: "var(--radius-sm)",
        cursor: "pointer",
        fontSize: "var(--text-xs)",
        fontFamily: "var(--font-sans)",
        fontWeight: 500,
        color: isOpen ? "var(--accent)" : "var(--gray-9)",
        transition: "all var(--transition-default)",
      }}
    >
      <svg
        width={12}
        height={12}
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
      >
        {/* Simple graph icon: 3 nodes connected */}
        <circle cx="4" cy="4" r="2" />
        <circle cx="12" cy="4" r="2" />
        <circle cx="8" cy="13" r="2" />
        <line x1="5.5" y1="5.2" x2="7" y2="11.2" />
        <line x1="10.5" y1="5.2" x2="9" y2="11.2" />
        <line x1="6" y1="4" x2="10" y2="4" />
      </svg>
      KG
    </button>
  );
}
