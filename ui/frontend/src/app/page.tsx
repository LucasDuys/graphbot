import { DagCanvas } from "@/components/dag/DagCanvas";

export default function Home() {
  return (
    <main style={{
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      background: "var(--bg-primary)",
      color: "var(--text-primary)",
      fontFamily: "var(--font-sans)",
    }}>
      <header style={{
        padding: "12px 20px",
        borderBottom: "1px solid var(--border-primary)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <h1 style={{ fontSize: 16, fontWeight: 600, margin: 0 }}>GraphBot</h1>
        <span style={{ fontSize: 12, color: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}>
          DAG Execution Monitor
        </span>
      </header>
      <div style={{ flex: 1, display: "flex" }}>
        <div style={{ flex: 1 }}>
          <DagCanvas />
        </div>
      </div>
    </main>
  );
}
