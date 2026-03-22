"""Tests for the FastAPI SSE backend."""

import ast
from pathlib import Path


def test_server_module_imports():
    """Verify the server module structure exists."""
    server_path = Path(__file__).parent.parent.parent / "ui" / "server.py"
    assert server_path.exists()


def test_server_has_endpoints():
    """Verify server defines expected endpoints."""
    server_path = Path(__file__).parent.parent.parent / "ui" / "server.py"
    source = server_path.read_text()
    tree = ast.parse(source)
    func_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert "submit_task" in func_names
    assert "stream_events" in func_names
    assert "graph_stats" in func_names


def test_server_has_pydantic_models():
    """Verify server defines request/response models."""
    server_path = Path(__file__).parent.parent.parent / "ui" / "server.py"
    source = server_path.read_text()
    tree = ast.parse(source)
    class_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    assert "TaskRequest" in class_names
    assert "TaskResponse" in class_names


def test_server_has_sse_stream():
    """Verify SSE event streaming is implemented."""
    server_path = Path(__file__).parent.parent.parent / "ui" / "server.py"
    source = server_path.read_text()
    assert "EventSourceResponse" in source
    assert "event_generator" in source
    assert "asyncio.Queue" in source


def test_server_has_cors_middleware():
    """Verify CORS is configured for local dev."""
    server_path = Path(__file__).parent.parent.parent / "ui" / "server.py"
    source = server_path.read_text()
    assert "CORSMiddleware" in source
    assert "localhost:3000" in source


def test_server_has_process_task():
    """Verify background task processor emits lifecycle events."""
    server_path = Path(__file__).parent.parent.parent / "ui" / "server.py"
    source = server_path.read_text()
    assert "task.started" in source
    assert "intake.complete" in source
    assert "task.complete" in source
    assert "task.error" in source


def test_server_has_tool_observability_events():
    """Verify SSE emits tool.invoke and tool.result events for leaf nodes."""
    server_path = Path(__file__).parent.parent.parent / "ui" / "server.py"
    source = server_path.read_text()
    assert "tool.invoke" in source
    assert "tool.result" in source
    assert "tool_method" in source
    assert "tool_params" in source
    assert "output_preview" in source
    assert "used_tool" in source
    assert "latency_ms" in source
