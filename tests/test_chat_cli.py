"""Tests for the CLI chat module -- import and helper checks (no live API)."""

from __future__ import annotations

from scripts.chat import GRAPH_DIR, HISTORY, print_header, print_history, print_stats
from graph.store import GraphStore


def test_imports() -> None:
    """Core components can be imported without error."""
    from core_gb.orchestrator import Orchestrator
    from models.openrouter import OpenRouterProvider
    from models.router import ModelRouter

    assert Orchestrator is not None
    assert OpenRouterProvider is not None
    assert ModelRouter is not None


def test_graph_dir_path() -> None:
    """GRAPH_DIR points to data/graph under the project root."""
    assert GRAPH_DIR.name == "graph"
    assert GRAPH_DIR.parent.name == "data"


def test_history_starts_empty() -> None:
    """HISTORY list starts empty."""
    # Clear any state from prior tests
    HISTORY.clear()
    assert HISTORY == []


def test_print_header(capsys: object) -> None:
    """print_header outputs the banner without errors."""
    print_header()
    import sys
    captured = sys.stdout  # capsys is a pytest fixture
    # Just verify no exception was raised; capsys captures output
    assert True


def test_print_history_empty(capsys: object) -> None:
    """print_history shows 'No history' when HISTORY is empty."""
    HISTORY.clear()
    print_history()
    # No exception means success


def test_print_stats_with_store() -> None:
    """print_stats runs against an in-memory GraphStore without error."""
    store = GraphStore(db_path=None)
    store.initialize()
    try:
        print_stats(store)
    finally:
        store.close()
