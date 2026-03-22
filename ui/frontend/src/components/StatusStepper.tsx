"use client";

import { useAtomValue } from "jotai";
import { taskStateAtom } from "@/lib/store";
import { motion } from "framer-motion";

const STEPS = [
  { key: "intake", label: "Classify" },
  { key: "decomposing", label: "Decompose" },
  { key: "executing", label: "Execute" },
  { key: "complete", label: "Done" },
];

function getStepState(stepKey: string, currentPhase: string): "upcoming" | "active" | "done" {
  const phaseOrder = ["idle", "intake", "decomposing", "executing", "aggregating", "complete", "error"];
  const stepIdx = phaseOrder.indexOf(stepKey);
  const currentIdx = phaseOrder.indexOf(currentPhase);

  if (currentPhase === "idle") return "upcoming";
  if (currentPhase === "error") return stepKey === "complete" ? "upcoming" : "done";
  if (stepIdx < currentIdx) return "done";
  if (stepIdx === currentIdx) return "active";
  return "upcoming";
}

export function StatusStepper() {
  const taskState = useAtomValue(taskStateAtom);

  if (taskState.phase === "idle") return null;

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: "var(--space-1)",
    }}>
      {STEPS.map((step, i) => {
        const state = getStepState(step.key, taskState.phase);
        return (
          <div key={step.key} style={{ display: "flex", alignItems: "center", gap: "var(--space-1)" }}>
            {i > 0 && (
              <div style={{
                width: 16,
                height: 1,
                background: state === "upcoming" ? "var(--gray-5)" : "var(--status-success)",
                transition: "background var(--transition-default)",
              }} />
            )}
            <motion.div
              animate={{
                background: state === "active" ? "var(--status-running)" : state === "done" ? "var(--status-success)" : "var(--gray-5)",
              }}
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                transition: "background var(--transition-default)",
              }}
            />
            <span style={{
              fontSize: "var(--text-xs)",
              fontWeight: state === "active" ? 600 : 400,
              color: state === "active" ? "var(--gray-11)" : state === "done" ? "var(--status-success)" : "var(--gray-8)",
              transition: "color var(--transition-default)",
            }}>
              {step.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
