"""Tests for forgetting engine -- archives stale memories to cold storage."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from graph.forgetting import ForgettingEngine
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create and initialize an in-memory GraphStore."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _create_memory(
    store: GraphStore,
    memory_id: str,
    content: str = "test memory",
    category: str = "observation",
    confidence: float = 0.8,
    valid_from: datetime | None = None,
) -> str:
    """Helper: create a Memory node with given properties."""
    props: dict[str, object] = {
        "id": memory_id,
        "content": content,
        "category": category,
        "confidence": confidence,
        "source_episode": "episode_test",
    }
    if valid_from is not None:
        props["valid_from"] = valid_from
    store.create_node("Memory", props)
    return memory_id


def _create_task(
    store: GraphStore,
    task_id: str,
    description: str = "test task",
    created_at: datetime | None = None,
) -> str:
    """Helper: create a Task node with given properties."""
    props: dict[str, object] = {
        "id": task_id,
        "description": description,
        "domain": "system",
        "complexity": 1,
        "status": "completed",
        "tokens_used": 100,
        "latency_ms": 50.0,
    }
    if created_at is not None:
        props["created_at"] = created_at
    store.create_node("Task", props)
    return task_id


class TestForgettingEngineInit:
    """ForgettingEngine initialization and configuration."""

    def test_default_threshold_and_grace_period(self) -> None:
        """Default activation threshold is 0.1 and grace period is 30 days."""
        store = _make_store()
        engine = ForgettingEngine(store)
        assert engine.activation_threshold == 0.1
        assert engine.grace_period_days == 30
        store.close()

    def test_custom_threshold_and_grace_period(self) -> None:
        """Custom activation threshold and grace period are accepted."""
        store = _make_store()
        engine = ForgettingEngine(
            store,
            activation_threshold=0.5,
            grace_period_days=7,
        )
        assert engine.activation_threshold == 0.5
        assert engine.grace_period_days == 7
        store.close()

    def test_cold_storage_path_default(self, tmp_path: Path) -> None:
        """Default cold storage path is data/cold_storage.json."""
        store = _make_store()
        engine = ForgettingEngine(store)
        assert engine.cold_storage_path.name == "cold_storage.json"
        store.close()

    def test_cold_storage_path_custom(self, tmp_path: Path) -> None:
        """Custom cold storage path is respected."""
        store = _make_store()
        custom_path = tmp_path / "archive.json"
        engine = ForgettingEngine(store, cold_storage_path=custom_path)
        assert engine.cold_storage_path == custom_path
        store.close()


class TestProtectedNodeTypes:
    """Protected node types (User, Project) are never forgotten."""

    def test_user_node_never_archived(self, tmp_path: Path) -> None:
        """User nodes are never archived regardless of age or inactivity."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        store.create_node("User", {
            "id": "user_1",
            "name": "Lucas",
            "role": "developer",
            "institution": "TU/e",
            "interests": "AI",
        })

        archived = engine.sweep()
        assert archived == 0

        # User node still exists
        node = store.get_node("User", "user_1")
        assert node is not None
        assert node["name"] == "Lucas"
        store.close()

    def test_project_node_never_archived(self, tmp_path: Path) -> None:
        """Project nodes are never archived regardless of age or inactivity."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        store.create_node("Project", {
            "id": "proj_1",
            "name": "GraphBot",
            "path": "/dev/graphbot",
            "language": "Python",
            "framework": "Kuzu",
            "status": "active",
        })

        archived = engine.sweep()
        assert archived == 0

        node = store.get_node("Project", "proj_1")
        assert node is not None
        assert node["name"] == "GraphBot"
        store.close()


class TestStaleMemoryArchival:
    """Stale memories below activation threshold are archived to cold storage."""

    def test_stale_memory_archived(self, tmp_path: Path) -> None:
        """Memory created long ago with low confidence is archived."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=30,
            activation_threshold=0.5,
        )

        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        _create_memory(
            store,
            "mem_stale",
            content="old forgotten fact",
            confidence=0.1,
            valid_from=old_date,
        )

        archived = engine.sweep()
        assert archived == 1

        # Node should be removed from the graph
        node = store.get_node("Memory", "mem_stale")
        assert node is None

        # Node should exist in cold storage
        with open(cold_path, "r") as f:
            data = json.load(f)
        assert len(data["archived_nodes"]) == 1
        entry = data["archived_nodes"][0]
        assert entry["node_id"] == "mem_stale"
        assert entry["table"] == "Memory"
        assert entry["properties"]["content"] == "old forgotten fact"
        store.close()

    def test_recent_memory_not_archived(self, tmp_path: Path) -> None:
        """Memory created within the grace period is not archived even with low confidence."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=30,
            activation_threshold=0.5,
        )

        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        _create_memory(
            store,
            "mem_recent",
            content="fresh memory",
            confidence=0.1,
            valid_from=recent_date,
        )

        archived = engine.sweep()
        assert archived == 0

        node = store.get_node("Memory", "mem_recent")
        assert node is not None
        store.close()

    def test_high_confidence_memory_not_archived(self, tmp_path: Path) -> None:
        """Memory with high confidence (above threshold) is not archived even if old."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=30,
            activation_threshold=0.5,
        )

        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        _create_memory(
            store,
            "mem_confident",
            content="important memory",
            confidence=0.9,
            valid_from=old_date,
        )

        archived = engine.sweep()
        assert archived == 0

        node = store.get_node("Memory", "mem_confident")
        assert node is not None
        store.close()

    def test_multiple_stale_memories_archived(self, tmp_path: Path) -> None:
        """Multiple stale memories are all archived in a single sweep."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=30,
            activation_threshold=0.5,
        )

        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(3):
            _create_memory(
                store,
                f"mem_stale_{i}",
                content=f"stale memory {i}",
                confidence=0.1,
                valid_from=old_date,
            )

        # Also create one that should NOT be archived
        _create_memory(
            store,
            "mem_keep",
            content="keep this",
            confidence=0.9,
            valid_from=old_date,
        )

        archived = engine.sweep()
        assert archived == 3

        # Stale ones gone
        for i in range(3):
            assert store.get_node("Memory", f"mem_stale_{i}") is None

        # Kept one still present
        assert store.get_node("Memory", "mem_keep") is not None
        store.close()

    def test_stale_task_archived(self, tmp_path: Path) -> None:
        """Task nodes past grace period with low activation are archived."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=30,
            activation_threshold=0.5,
        )

        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        _create_task(
            store,
            "task_old",
            description="old completed task",
            created_at=old_date,
        )

        archived = engine.sweep()
        assert archived == 1

        node = store.get_node("Task", "task_old")
        assert node is None
        store.close()


class TestColdStorageFormat:
    """Cold storage JSON file format and accumulation."""

    def test_cold_storage_created_on_first_sweep(self, tmp_path: Path) -> None:
        """Cold storage file is created on the first sweep with archived nodes."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        _create_memory(store, "mem_1", content="test")

        engine.sweep()

        assert cold_path.exists()
        with open(cold_path, "r") as f:
            data = json.load(f)
        assert "archived_nodes" in data
        assert "version" in data
        store.close()

    def test_cold_storage_accumulates(self, tmp_path: Path) -> None:
        """Multiple sweeps accumulate entries in cold storage, not overwrite."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        _create_memory(store, "mem_1", content="first")
        engine.sweep()

        _create_memory(store, "mem_2", content="second")
        engine.sweep()

        with open(cold_path, "r") as f:
            data = json.load(f)
        assert len(data["archived_nodes"]) == 2
        ids = {entry["node_id"] for entry in data["archived_nodes"]}
        assert ids == {"mem_1", "mem_2"}
        store.close()

    def test_archived_entry_has_timestamp(self, tmp_path: Path) -> None:
        """Each archived entry has an archived_at timestamp."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        _create_memory(store, "mem_1", content="test")
        engine.sweep()

        with open(cold_path, "r") as f:
            data = json.load(f)
        entry = data["archived_nodes"][0]
        assert "archived_at" in entry
        # Verify it is a valid ISO timestamp
        datetime.fromisoformat(entry["archived_at"])
        store.close()

    def test_no_archive_file_when_nothing_archived(self, tmp_path: Path) -> None:
        """Cold storage file is not created if no nodes are archived."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=30,
            activation_threshold=0.1,
        )

        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        _create_memory(
            store,
            "mem_recent",
            content="recent",
            confidence=0.9,
            valid_from=recent_date,
        )

        archived = engine.sweep()
        assert archived == 0
        assert not cold_path.exists()
        store.close()


class TestRecovery:
    """Archived nodes can be restored from cold storage."""

    def test_restore_archived_memory(self, tmp_path: Path) -> None:
        """restore() re-creates a Memory node from cold storage."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        _create_memory(store, "mem_restore", content="will be restored", confidence=0.5)
        engine.sweep()

        # Verify it was archived
        assert store.get_node("Memory", "mem_restore") is None

        # Restore it
        restored = engine.restore("mem_restore")
        assert restored is True

        # Verify it is back in the graph
        node = store.get_node("Memory", "mem_restore")
        assert node is not None
        assert node["content"] == "will be restored"
        store.close()

    def test_restore_removes_from_cold_storage(self, tmp_path: Path) -> None:
        """restore() removes the entry from cold storage after restoring."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        _create_memory(store, "mem_1", content="first")
        _create_memory(store, "mem_2", content="second")
        engine.sweep()

        engine.restore("mem_1")

        with open(cold_path, "r") as f:
            data = json.load(f)
        remaining_ids = {e["node_id"] for e in data["archived_nodes"]}
        assert "mem_1" not in remaining_ids
        assert "mem_2" in remaining_ids
        store.close()

    def test_restore_nonexistent_node_returns_false(self, tmp_path: Path) -> None:
        """restore() returns False when the node is not in cold storage."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
        )

        restored = engine.restore("nonexistent_id")
        assert restored is False
        store.close()

    def test_restore_when_no_cold_storage_file(self, tmp_path: Path) -> None:
        """restore() returns False when cold storage file does not exist."""
        store = _make_store()
        cold_path = tmp_path / "does_not_exist.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
        )

        restored = engine.restore("some_id")
        assert restored is False
        store.close()

    def test_restore_archived_task(self, tmp_path: Path) -> None:
        """restore() can restore a Task node from cold storage."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        _create_task(store, "task_restore", description="will be restored")
        engine.sweep()

        assert store.get_node("Task", "task_restore") is None

        restored = engine.restore("task_restore")
        assert restored is True

        node = store.get_node("Task", "task_restore")
        assert node is not None
        assert node["description"] == "will be restored"
        store.close()


class TestSweepReturnValue:
    """sweep() returns the count of archived nodes."""

    def test_returns_zero_when_nothing_archived(self, tmp_path: Path) -> None:
        """sweep() returns 0 when no nodes qualify for archival."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=30,
            activation_threshold=0.1,
        )

        archived = engine.sweep()
        assert archived == 0
        store.close()

    def test_returns_exact_count(self, tmp_path: Path) -> None:
        """sweep() returns the exact number of archived nodes."""
        store = _make_store()
        cold_path = tmp_path / "cold.json"
        engine = ForgettingEngine(
            store,
            cold_storage_path=cold_path,
            grace_period_days=0,
            activation_threshold=1.0,
        )

        _create_memory(store, "mem_1", content="a")
        _create_memory(store, "mem_2", content="b")

        archived = engine.sweep()
        assert archived == 2
        store.close()
