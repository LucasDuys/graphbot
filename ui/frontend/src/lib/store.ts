import { atom } from "jotai";
import type { KGNode, KGEdge } from "@/lib/api";

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

export type DagEdgeData = {
  source: string;
  target: string;
  label?: string;
  flowing?: boolean;
};

export const dagNodesAtom = atom<DagNodeData[]>([]);
export const dagEdgesAtom = atom<DagEdgeData[]>([]);
export const taskStateAtom = atom<TaskState>({
  task_id: null,
  message: "",
  phase: "idle",
  intake: null,
  result: null,
  error: null,
});
export const selectedNodeIdAtom = atom<string | null>(null);
export const executionStartAtom = atom<number | null>(null);

// Knowledge graph panel state
export const kgNodesAtom = atom<KGNode[]>([]);
export const kgEdgesAtom = atom<KGEdge[]>([]);
export const kgPanelOpenAtom = atom<boolean>(false);
