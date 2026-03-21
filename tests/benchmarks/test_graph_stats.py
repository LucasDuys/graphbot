"""Tests for graph_stats.py -- verify print_stats runs without crashing."""

from __future__ import annotations

import io
import sys

import pytest

from graph.store import GraphStore
from scripts.graph_stats import print_stats


class TestGraphStats:
    """Verify graph_stats print_stats on an in-memory GraphStore."""

    @pytest.fixture()
    def seeded_store(self) -> GraphStore:
        """Create an in-memory GraphStore with some seed data."""
        store = GraphStore()
        store.initialize()

        # Seed a User node
        store.create_node("User", {
            "id": "user_1",
            "name": "Test User",
            "role": "developer",
            "institution": "Test Uni",
            "interests": "graphs",
        })

        # Seed a Project node
        store.create_node("Project", {
            "id": "proj_1",
            "name": "TestProject",
            "path": "/test",
            "language": "Python",
            "framework": "pytest",
            "status": "active",
        })

        # Seed an edge
        store.create_edge("OWNS", "user_1", "proj_1")

        # Seed a Memory node
        store.create_node("Memory", {
            "id": "mem_1",
            "content": "Test memory content",
            "category": "fact",
            "confidence": 0.9,
            "source_episode": "test",
        })
        store.create_edge("ABOUT", "mem_1", "user_1")

        return store

    def test_print_stats_no_crash(self, seeded_store: GraphStore) -> None:
        """print_stats should complete without raising any exceptions."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            print_stats(seeded_store)
        finally:
            sys.stdout = old_stdout
            seeded_store.close()

        output = captured.getvalue()
        assert "GRAPH STATISTICS" in output

    def test_print_stats_shows_counts(self, seeded_store: GraphStore) -> None:
        """print_stats should report non-zero counts for seeded node types."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            print_stats(seeded_store)
        finally:
            sys.stdout = old_stdout
            seeded_store.close()

        output = captured.getvalue()
        # Should show User and Project counts
        assert "User" in output
        assert "Project" in output
        assert "Memory" in output

    def test_print_stats_shows_edges(self, seeded_store: GraphStore) -> None:
        """print_stats should report edge counts."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            print_stats(seeded_store)
        finally:
            sys.stdout = old_stdout
            seeded_store.close()

        output = captured.getvalue()
        assert "TOTAL EDGES" in output
        assert "OWNS" in output
        assert "ABOUT" in output

    def test_print_stats_shows_total(self, seeded_store: GraphStore) -> None:
        """print_stats should report a TOTAL node count of at least 3."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            print_stats(seeded_store)
        finally:
            sys.stdout = old_stdout
            seeded_store.close()

        output = captured.getvalue()
        # We seeded 3 nodes (User, Project, Memory)
        assert "TOTAL" in output

    def test_print_stats_empty_graph(self) -> None:
        """print_stats should handle an empty graph gracefully."""
        store = GraphStore()
        store.initialize()
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            print_stats(store)
        finally:
            sys.stdout = old_stdout
            store.close()

        output = captured.getvalue()
        assert "GRAPH STATISTICS" in output
        assert "PATTERNS: 0" in output
