"use client";

import { useCallback } from "react";
import { useSetAtom } from "jotai";
import { dagNodesAtom, dagEdgesAtom, taskStateAtom, executionStartAtom } from "./store";
import type { DagNodeData, DagEdgeData } from "./store";
import { submitTask, streamEvents } from "./api";

export function useTaskExecution() {
  const setNodes = useSetAtom(dagNodesAtom);
  const setEdges = useSetAtom(dagEdgesAtom);
  const setTaskState = useSetAtom(taskStateAtom);
  const setExecutionStart = useSetAtom(executionStartAtom);

  const execute = useCallback(async (message: string) => {
    // Reset state
    setNodes([]);
    setEdges([]);
    setExecutionStart(Date.now());
    setTaskState({
      task_id: null,
      message,
      phase: "intake",
      intake: null,
      result: null,
      error: null,
    });

    try {
      const taskId = await submitTask(message);
      setTaskState((prev) => ({ ...prev, task_id: taskId }));

      /* eslint-disable @typescript-eslint/no-explicit-any */
      streamEvents(
        taskId,
        (type: string, payload: any) => {
          switch (type) {
            case "intake.complete":
              setTaskState((prev) => ({
                ...prev,
                phase: "decomposing",
                intake: payload,
              }));
              break;

            case "node.created": {
              const node: DagNodeData = {
                id: payload.id,
                label: payload.description || payload.id,
                status: "pending",
                domain: payload.domain || "synthesis",
                complexity: payload.complexity || 1,
                is_atomic: payload.is_atomic || false,
                provides: payload.provides,
                consumes: payload.consumes,
              };
              setNodes((prev) => [...prev, node]);
              break;
            }

            case "node.status":
              setNodes((prev) =>
                prev.map((n) =>
                  n.id === payload.node_id
                    ? {
                        ...n,
                        status: payload.status,
                        tokens: payload.tokens,
                        latency_ms: payload.latency_ms,
                        output: payload.output,
                      }
                    : n,
                ),
              );
              if (payload.status === "running") {
                setTaskState((prev) => ({ ...prev, phase: "executing" }));
              }
              break;

            case "edge.created":
              setEdges((prev: DagEdgeData[]) => [
                ...prev,
                { source: payload.source, target: payload.target, label: payload.label, flowing: false },
              ]);
              break;

            case "data.flow":
              setEdges((prev: DagEdgeData[]) =>
                prev.map((e: DagEdgeData) =>
                  e.source === payload.source && e.target === payload.target
                    ? { ...e, flowing: true }
                    : e,
                ),
              );
              break;

            case "task.complete":
              setExecutionStart(null);
              setTaskState((prev) => ({
                ...prev,
                phase: "complete",
                result: payload,
              }));
              break;

            case "task.error":
              setExecutionStart(null);
              setTaskState((prev) => ({
                ...prev,
                phase: "error",
                error: payload.error,
              }));
              break;
          }
        },
        () => {},
        (err: Error) => {
          setTaskState((prev) => ({
            ...prev,
            phase: "error",
            error: err.message,
          }));
        },
      );
      /* eslint-enable @typescript-eslint/no-explicit-any */
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to submit task";
      setTaskState((prev) => ({
        ...prev,
        phase: "error",
        error: message,
      }));
    }
  }, [setNodes, setEdges, setTaskState, setExecutionStart]);

  return { execute };
}
