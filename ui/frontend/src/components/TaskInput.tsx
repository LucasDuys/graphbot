"use client";

import { useState, useCallback, type KeyboardEvent } from "react";
import { useAtomValue } from "jotai";
import { taskStateAtom } from "@/lib/store";
import { useTaskExecution } from "@/lib/useTaskExecution";

export function TaskInput() {
  const [input, setInput] = useState("");
  const taskState = useAtomValue(taskStateAtom);
  const { execute } = useTaskExecution();
  const isProcessing = taskState.phase !== "idle" && taskState.phase !== "complete" && taskState.phase !== "error";

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isProcessing) return;
    execute(trimmed);
    setInput("");
  }, [input, isProcessing, execute]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  return (
    <div style={{
      padding: "12px 20px",
      borderTop: "1px solid var(--border-primary)",
      display: "flex",
      gap: 8,
    }}>
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={isProcessing ? "Processing..." : "Enter a task..."}
        disabled={isProcessing}
        style={{
          flex: 1,
          padding: "8px 12px",
          border: "1px solid var(--border-primary)",
          borderRadius: "var(--radius-md)",
          background: "var(--bg-secondary)",
          color: "var(--text-primary)",
          fontFamily: "var(--font-sans)",
          fontSize: 14,
          outline: "none",
        }}
      />
      <button
        onClick={handleSubmit}
        disabled={isProcessing || !input.trim()}
        style={{
          padding: "8px 16px",
          borderRadius: "var(--radius-md)",
          border: "1px solid var(--border-primary)",
          background: isProcessing ? "var(--bg-tertiary)" : "var(--text-primary)",
          color: isProcessing ? "var(--text-tertiary)" : "var(--bg-primary)",
          fontFamily: "var(--font-sans)",
          fontSize: 13,
          fontWeight: 600,
          cursor: isProcessing ? "not-allowed" : "pointer",
        }}
      >
        {isProcessing ? "Running" : "Execute"}
      </button>
    </div>
  );
}
