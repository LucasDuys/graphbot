"use client";

import { DagCanvas } from "@/components/dag/DagCanvas";
import { TaskInput } from "@/components/TaskInput";
import { StatusStepper } from "@/components/StatusStepper";
import { NodeDetail } from "@/components/NodeDetail";
import { ResultPanel } from "@/components/ResultPanel";
import { KnowledgeGraphPanel } from "@/components/KnowledgeGraphPanel";
import { KnowledgeGraphToggle } from "@/components/KnowledgeGraphToggle";
import { DarkModeToggle } from "@/components/DarkModeToggle";

export default function Home() {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      background: "var(--gray-1)",
      color: "var(--gray-11)",
      fontFamily: "var(--font-sans)",
    }}>
      {/* Top bar */}
      <header style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "var(--space-3) var(--space-5)",
        borderBottom: "1px solid var(--border-color)",
        background: "var(--gray-2)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
          <span style={{ fontSize: "var(--text-lg)", fontWeight: 600 }}>GraphBot</span>
          <span style={{ fontSize: "var(--text-xs)", color: "var(--gray-8)", fontFamily: "var(--font-mono)" }}>v0.1.0</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
          <StatusStepper />
          <KnowledgeGraphToggle />
          <DarkModeToggle />
        </div>
      </header>

      {/* Task input */}
      <TaskInput />

      {/* Main content */}
      <div style={{
        flex: 1,
        display: "flex",
        overflow: "hidden",
      }}>
        {/* Left: DAG Canvas */}
        <div style={{ flex: "1 1 60%", borderRight: "1px solid var(--border-color)" }}>
          <DagCanvas />
        </div>

        {/* Right: Details + Result */}
        <div style={{
          flex: "0 0 40%",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          minWidth: 320,
          maxWidth: 500,
        }}>
          <div style={{ flex: 1, overflow: "auto", borderBottom: "1px solid var(--border-color)" }}>
            <NodeDetail />
          </div>
          <div style={{ flex: 1, overflow: "auto" }}>
            <ResultPanel />
          </div>
        </div>
      </div>

      {/* Knowledge Graph Panel -- bottom-right overlay */}
      <KnowledgeGraphPanel />
    </div>
  );
}
