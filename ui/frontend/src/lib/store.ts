import { atom } from "jotai";

export type NodeStatus = "pending" | "running" | "completed" | "failed";

export type DagNodeData = {
  id: string;
  label: string;
  status: NodeStatus;
  domain: string;
  complexity: number;
  is_atomic: boolean;
  tokens?: number;
  latency_ms?: number;
  output?: string;
  provides?: string[];
  consumes?: string[];
};

export type TaskState = {
  task_id: string | null;
  message: string;
  phase: "idle" | "intake" | "decomposing" | "executing" | "aggregating" | "complete" | "error";
  intake: {
    domain: string;
    complexity: number;
    is_simple: boolean;
  } | null;
  result: {
    output: string;
    total_nodes: number;
    total_tokens: number;
    total_latency_ms: number;
    total_cost: number;
    model_used: string;
    success: boolean;
  } | null;
  error: string | null;
};

export const dagNodesAtom = atom<DagNodeData[]>([]);
export const dagEdgesAtom = atom<{ source: string; target: string; label?: string }[]>([]);
export const taskStateAtom = atom<TaskState>({
  task_id: null,
  message: "",
  phase: "idle",
  intake: null,
  result: null,
  error: null,
});
