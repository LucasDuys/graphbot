"use client";

import { useState, useCallback, useEffect, useRef, KeyboardEvent } from "react";
import { useAtomValue } from "jotai";
import { taskStateAtom, executionStartAtom } from "@/lib/store";
import { useTaskExecution } from "@/lib/useTaskExecution";

function formatElapsed(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (minutes > 0) {
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  }
  return `${secs}s`;
}

export function TaskInput() {
  const [input, setInput] = useState("");
  const taskState = useAtomValue(taskStateAtom);
  const executionStart = useAtomValue(executionStartAtom);
  const { execute } = useTaskExecution();
  const isProcessing = !["idle", "complete", "error"].includes(taskState.phase);

  // Elapsed timer
  const [elapsed, setElapsed] = useState<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (executionStart !== null) {
      setElapsed(Date.now() - executionStart);
      timerRef.current = setInterval(() => {
        setElapsed(Date.now() - executionStart);
      }, 100);
    } else {
      setElapsed(0);
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [executionStart]);

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isProcessing) return;
    execute(trimmed);
    setInput("");
  }, [input, isProcessing, execute]);

  return (
    <div style={{
      padding: "var(--space-3) var(--space-5)",
      borderBottom: "1px solid var(--border-color)",
      background: "var(--gray-2)",
    }}>
      <div style={{
        display: "flex",
        gap: "var(--space-2)",
        alignItems: "center",
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e: KeyboardEvent) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
          }}
          placeholder={isProcessing ? "Processing..." : "What would you like GraphBot to do?"}
          disabled={isProcessing}
          style={{
            flex: 1,
            padding: "var(--space-2) var(--space-3)",
            fontSize: "var(--text-lg)",
            fontFamily: "var(--font-sans)",
            background: "var(--gray-1)",
            color: "var(--gray-11)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-md)",
            outline: "none",
            transition: "border-color var(--transition-default), box-shadow var(--transition-default)",
          }}
          onFocus={(e) => {
            e.target.style.borderColor = "var(--accent)";
            e.target.style.boxShadow = "0 0 0 2px var(--accent-soft)";
          }}
          onBlur={(e) => {
            e.target.style.borderColor = "var(--border-color)";
            e.target.style.boxShadow = "none";
          }}
        />
        {isProcessing && executionStart !== null && (
          <span style={{
            fontSize: "var(--text-sm)",
            fontFamily: "var(--font-mono)",
            color: "var(--status-running)",
            fontWeight: 600,
            minWidth: 40,
            textAlign: "right",
          }}>
            {formatElapsed(elapsed)}
          </span>
        )}
        <button
          onClick={handleSubmit}
          disabled={isProcessing || !input.trim()}
          style={{
            padding: "var(--space-2) var(--space-4)",
            fontSize: "var(--text-sm)",
            fontWeight: 600,
            fontFamily: "var(--font-sans)",
            background: isProcessing ? "var(--gray-3)" : "var(--gray-12)",
            color: isProcessing ? "var(--gray-8)" : "var(--gray-1)",
            border: "none",
            borderRadius: "var(--radius-md)",
            cursor: isProcessing ? "not-allowed" : "pointer",
            transition: "background var(--transition-default)",
          }}
        >
          {isProcessing ? "Running..." : "Execute"}
        </button>
      </div>
    </div>
  );
}
