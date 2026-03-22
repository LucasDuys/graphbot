const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function submitTask(message: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/task`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    throw new Error(`Failed to submit task: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return data.task_id;
}

export function streamEvents(
  taskId: string,
  onEvent: (type: string, payload: unknown) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): () => void {
  const eventSource = new EventSource(`${API_BASE}/api/stream/${taskId}`);

  const eventTypes = [
    "task.started",
    "intake.complete",
    "decomposition.started",
    "node.created",
    "node.status",
    "edge.created",
    "data.flow",
    "task.complete",
    "task.error",
    "timeout",
  ];

  for (const type of eventTypes) {
    eventSource.addEventListener(type, (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data);
        onEvent(type, payload);
      } catch {
        onEvent(type, event.data);
      }
    });
  }

  eventSource.onerror = () => {
    eventSource.close();
    onDone();
  };

  return () => eventSource.close();
}

export async function getGraphStats(): Promise<Record<string, number>> {
  const res = await fetch(`${API_BASE}/api/graph/stats`);
  if (!res.ok) {
    throw new Error(`Failed to fetch graph stats: ${res.status}`);
  }
  return res.json();
}

export type KGNode = {
  _type: string;
  id: string;
  name?: string;
  [key: string]: unknown;
};

export type KGEdge = {
  source: string;
  target: string;
  type: string;
  from_type: string;
  to_type: string;
};

export type KGData = {
  nodes: KGNode[];
  edges: KGEdge[];
};

export async function getGraphEntities(): Promise<KGData> {
  const res = await fetch(`${API_BASE}/api/graph/entities`);
  if (!res.ok) {
    throw new Error(`Failed to fetch graph entities: ${res.status}`);
  }
  return res.json();
}
