"""FastAPI backend for GraphBot UI -- streams execution events via SSE."""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Load env
def _load_env() -> None:
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

from core_gb.orchestrator import Orchestrator
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter

app = FastAPI(title="GraphBot UI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_store: GraphStore | None = None
_orchestrator: Orchestrator | None = None
_active_streams: dict[str, asyncio.Queue] = {}


def get_orchestrator() -> Orchestrator:
    global _store, _orchestrator
    if _orchestrator is None:
        graph_dir = Path(__file__).parent.parent / "data" / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        _store = GraphStore(str(graph_dir))
        _store.initialize()
        provider = OpenRouterProvider()
        router = ModelRouter(provider)
        _orchestrator = Orchestrator(_store, router)
    return _orchestrator


class TaskRequest(BaseModel):
    message: str


class TaskResponse(BaseModel):
    task_id: str


@app.post("/api/task", response_model=TaskResponse)
async def submit_task(req: TaskRequest) -> TaskResponse:
    """Submit a task for processing. Returns task_id for SSE stream."""
    task_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _active_streams[task_id] = queue

    # Start processing in background
    asyncio.create_task(_process_task(task_id, req.message, queue))

    return TaskResponse(task_id=task_id)


@app.get("/api/stream/{task_id}")
async def stream_events(task_id: str) -> EventSourceResponse:
    """SSE endpoint that streams execution events for a task."""
    if task_id not in _active_streams:
        async def not_found() -> AsyncGenerator:
            yield {"event": "error", "data": json.dumps({"error": "Task not found"})}
        return EventSourceResponse(not_found())

    queue = _active_streams[task_id]

    async def event_generator() -> AsyncGenerator:
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=120)
                if event is None:  # End signal
                    break
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["payload"]),
                }
        except asyncio.TimeoutError:
            yield {"event": "timeout", "data": json.dumps({"error": "Stream timeout"})}
        finally:
            _active_streams.pop(task_id, None)

    return EventSourceResponse(event_generator())


@app.get("/api/graph/stats")
async def graph_stats() -> dict:
    """Return knowledge graph statistics."""
    orchestrator = get_orchestrator()
    store = orchestrator._store
    stats = {}
    for table in [
        "User", "Project", "Service", "Memory", "Task",
        "PatternNode", "ExecutionTree",
    ]:
        try:
            rows = store.query(f"MATCH (n:{table}) RETURN count(n) AS cnt")
            stats[table] = rows[0]["cnt"] if rows else 0
        except Exception:
            stats[table] = 0
    return stats


async def _process_task(task_id: str, message: str, queue: asyncio.Queue) -> None:
    """Process a task and emit events to the SSE queue."""
    orchestrator = get_orchestrator()

    # Emit start event
    await queue.put({
        "type": "task.started",
        "payload": {"task_id": task_id, "message": message, "timestamp": time.time()},
    })

    # Emit intake event
    intake = orchestrator._intake.parse(message)
    await queue.put({
        "type": "intake.complete",
        "payload": {
            "domain": intake.domain.value,
            "complexity": intake.complexity,
            "is_simple": intake.is_simple,
            "entities": list(intake.entities),
        },
    })

    # Process through orchestrator
    try:
        result = await orchestrator.process(message)

        # Emit completion
        await queue.put({
            "type": "task.complete",
            "payload": {
                "task_id": task_id,
                "success": result.success,
                "output": result.output,
                "total_nodes": result.total_nodes,
                "total_tokens": result.total_tokens,
                "total_latency_ms": result.total_latency_ms,
                "total_cost": result.total_cost,
                "model_used": result.model_used,
            },
        })
    except Exception as exc:
        await queue.put({
            "type": "task.error",
            "payload": {"task_id": task_id, "error": str(exc)},
        })

    # Signal end of stream
    await queue.put(None)


@app.on_event("shutdown")
async def shutdown() -> None:
    global _store
    if _store is not None:
        _store.close()
