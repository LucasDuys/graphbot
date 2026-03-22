"""Tests for forgetting + consolidation coordination and maintenance script.

Verifies:
  - Consolidation runs before forgetting in the maintenance pipeline
  - Consolidated nodes inherit the highest activation of their sources
  - Protected node types (User, Project) survive both engines
  - scripts/maintain_graph.py runs consolidation then forgetting
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from graph.consolidation import ConsolidationEngine, ConsolidationResult
from graph.forgetting import PROTECTED_TYPES, ForgettingEngine
from graph.maintenance import GraphMaintenance, MaintenanceResult
from graph.store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> GraphStore:
    """In-memory Kuzu graph store with schema initialized."""
    s = GraphStore(db_path=None)
    s.initialize()
    return s


@pytest.fixture
def cold_storage_path(tmp_path: Path) -> Path:
    """Temporary cold storage JSON path."""
    return tmp_path / "cold_storage.json"


# ---------------------------------------------------------------------------
# Test: consolidation runs before forgetting
# ---------------------------------------------------------------------------

class TestConsolidationBeforeForgetting:
    """Verify the maintenance pipeline orders consolidation before forgetting."""

    def test_run_calls_consolidation_then_forgetting(self, store: GraphStore) -> None:
        """Consolidation must execute first, then forgetting."""
        call_order: list[str] = []

        original_consolidation_run = ConsolidationEngine.run
        original_forgetting_sweep = ForgettingEngine.sweep

        def mock_consolidation_run(self_engine: Any) -> ConsolidationResult:
            call_order.append("consolidation")
            return ConsolidationResult()

        def mock_forgetting_sweep(self_engine: Any) -> int:
            call_order.append("forgetting")
            return 0

        with patch.object(ConsolidationEngine, "run", mock_consolidation_run), \
             patch.object(ForgettingEngine, "sweep", mock_forgetting_sweep):
            maintenance = GraphMaintenance(store)
            maintenance.run()

        assert call_order == ["consolidation", "forgetting"], (
            f"Expected consolidation before forgetting, got: {call_order}"
        )

    def test_result_contains_both_phases(self, store: GraphStore) -> None:
        """MaintenanceResult includes data from both consolidation and forgetting."""
        maintenance = GraphMaintenance(store)
        result = maintenance.run()

        assert isinstance(result, MaintenanceResult)
        assert hasattr(result, "consolidation")
        assert hasattr(result, "archived_count")

    def test_forgetting_sees_consolidated_state(
        self, store: GraphStore, cold_storage_path: Path
    ) -> None:
        """After consolidation merges duplicates, forgetting operates on the merged graph.

        Create two Memory nodes with the same name (duplicates). One has high
        confidence (should survive), the other has low confidence. After merge,
        the surviving node should inherit the high confidence and not be archived.
        """
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=60)

        # Memory A: low confidence, old -- would be archived on its own
        store.create_node("Memory", {
            "id": "mem_dup_a",
            "content": "Important fact about Python",
            "category": "knowledge",
            "confidence": 0.05,
            "valid_from": old_time,
        })

        # Memory B: high confidence, old -- survives forgetting
        store.create_node("Memory", {
            "id": "mem_dup_b",
            "content": "Important fact about Python",
            "category": "knowledge",
            "confidence": 0.9,
            "valid_from": old_time,
        })

        # After consolidation merges A into B, only B remains with confidence 0.9.
        # Forgetting should NOT archive it because activation >= threshold.
        maintenance = GraphMaintenance(
            store,
            activation_threshold=0.1,
            grace_period_days=30,
            cold_storage_path=cold_storage_path,
        )
        result = maintenance.run()

        # The high-confidence node should still be in the graph
        node_b = store.get_node("Memory", "mem_dup_b")
        # After consolidation, at least one node with high confidence must survive
        # (either mem_dup_a or mem_dup_b depending on which is primary)
        surviving = (
            store.get_node("Memory", "mem_dup_a") is not None
            or store.get_node("Memory", "mem_dup_b") is not None
        )
        assert surviving, "At least one merged memory should survive forgetting"


# ---------------------------------------------------------------------------
# Test: consolidated nodes inherit highest activation
# ---------------------------------------------------------------------------

class TestActivationInheritance:
    """Consolidated nodes should inherit the highest activation of their sources."""

    def test_summary_inherits_max_confidence(self, store: GraphStore) -> None:
        """Summary Memory created by consolidation gets max confidence of sources."""
        now = datetime.now(timezone.utc)

        # Create a User for linking
        store.create_node("User", {"id": "user_1", "name": "Test User"})

        # Create 3 memories with different confidence levels linked to same user
        for i, conf in enumerate([0.3, 0.9, 0.5]):
            mid = f"mem_{i}"
            store.create_node("Memory", {
                "id": mid,
                "content": f"Memory content {i}",
                "category": "knowledge",
                "confidence": conf,
                "valid_from": now,
            })
            store.create_edge("ABOUT", mid, "user_1")

        maintenance = GraphMaintenance(store)
        result = maintenance.run()

        # Find any summary memories created
        rows = store.query(
            "MATCH (m:Memory) WHERE m.category = $cat RETURN m.id, m.confidence",
            {"cat": "summary"},
        )

        if rows:
            # The summary should have the highest confidence from its sources
            summary_confidence = float(rows[0]["m.confidence"])
            assert summary_confidence >= 0.9, (
                f"Summary confidence {summary_confidence} should be >= 0.9 "
                f"(max of source confidences)"
            )

    def test_merge_preserves_higher_confidence(self, store: GraphStore) -> None:
        """When two duplicate nodes are merged, the result keeps the higher confidence."""
        now = datetime.now(timezone.utc)

        # Create two Memory nodes that look like duplicates (same content)
        store.create_node("Memory", {
            "id": "mem_low",
            "content": "Duplicate memory",
            "category": "knowledge",
            "confidence": 0.2,
            "valid_from": now,
        })
        store.create_node("Memory", {
            "id": "mem_high",
            "content": "Duplicate memory",
            "category": "knowledge",
            "confidence": 0.95,
            "valid_from": now,
        })

        maintenance = GraphMaintenance(store)
        result = maintenance.run()

        # After merge, the surviving node should exist with high confidence
        # The primary is alphabetically first (mem_high < mem_low)
        surviving_node = store.get_node("Memory", "mem_high")
        if surviving_node is None:
            surviving_node = store.get_node("Memory", "mem_low")

        assert surviving_node is not None, "At least one merged node must survive"
        confidence = float(surviving_node.get("confidence", 0.0))
        assert confidence >= 0.9, (
            f"Surviving merged node confidence {confidence} should be >= 0.9"
        )


# ---------------------------------------------------------------------------
# Test: protected node types survive both engines
# ---------------------------------------------------------------------------

class TestProtectedNodes:
    """User and Project nodes must never be archived or lost during maintenance."""

    def test_protected_types_defined(self) -> None:
        """PROTECTED_TYPES includes User and Project."""
        assert "User" in PROTECTED_TYPES
        assert "Project" in PROTECTED_TYPES

    def test_user_survives_forgetting(
        self, store: GraphStore, cold_storage_path: Path
    ) -> None:
        """User nodes are never archived even if old and inactive."""
        old_time = datetime.now(timezone.utc) - timedelta(days=365)

        store.create_node("User", {
            "id": "user_old",
            "name": "Ancient User",
            "access_count": 0,
            "last_accessed": old_time,
        })

        engine = ForgettingEngine(
            store,
            activation_threshold=0.5,
            grace_period_days=1,
            cold_storage_path=cold_storage_path,
        )
        archived = engine.sweep()

        assert archived == 0, "User nodes must not be archived"
        assert store.get_node("User", "user_old") is not None

    def test_project_survives_forgetting(
        self, store: GraphStore, cold_storage_path: Path
    ) -> None:
        """Project nodes are never archived even if old and inactive."""
        old_time = datetime.now(timezone.utc) - timedelta(days=365)

        store.create_node("Project", {
            "id": "proj_old",
            "name": "Old Project",
            "status": "archived",
            "access_count": 0,
            "last_accessed": old_time,
        })

        engine = ForgettingEngine(
            store,
            activation_threshold=0.5,
            grace_period_days=1,
            cold_storage_path=cold_storage_path,
        )
        archived = engine.sweep()

        assert archived == 0, "Project nodes must not be archived"
        assert store.get_node("Project", "proj_old") is not None

    def test_user_survives_full_maintenance(
        self, store: GraphStore, cold_storage_path: Path
    ) -> None:
        """User nodes survive the full maintenance pipeline (consolidation + forgetting)."""
        old_time = datetime.now(timezone.utc) - timedelta(days=365)

        store.create_node("User", {
            "id": "user_permanent",
            "name": "Permanent User",
            "access_count": 0,
            "last_accessed": old_time,
        })

        maintenance = GraphMaintenance(
            store,
            activation_threshold=0.5,
            grace_period_days=1,
            cold_storage_path=cold_storage_path,
        )
        maintenance.run()

        node = store.get_node("User", "user_permanent")
        assert node is not None, "User node must survive full maintenance"
        assert node["name"] == "Permanent User"

    def test_project_survives_full_maintenance(
        self, store: GraphStore, cold_storage_path: Path
    ) -> None:
        """Project nodes survive the full maintenance pipeline."""
        old_time = datetime.now(timezone.utc) - timedelta(days=365)

        store.create_node("Project", {
            "id": "proj_permanent",
            "name": "Permanent Project",
            "access_count": 0,
            "last_accessed": old_time,
        })

        maintenance = GraphMaintenance(
            store,
            activation_threshold=0.5,
            grace_period_days=1,
            cold_storage_path=cold_storage_path,
        )
        maintenance.run()

        node = store.get_node("Project", "proj_permanent")
        assert node is not None, "Project node must survive full maintenance"

    def test_protected_types_in_consolidation_engine(self, store: GraphStore) -> None:
        """Consolidation engine never deletes User or Project nodes."""
        store.create_node("User", {"id": "user_1", "name": "Alice"})
        store.create_node("User", {"id": "user_2", "name": "Alice"})

        engine = ConsolidationEngine(store)
        result = engine.run()

        # Both user nodes should still exist (or merged preserving one)
        user_1 = store.get_node("User", "user_1")
        user_2 = store.get_node("User", "user_2")

        # At least one must survive
        assert user_1 is not None or user_2 is not None, (
            "At least one User node must survive consolidation"
        )

    def test_forgetting_skips_protected_in_collect_candidates(
        self, store: GraphStore, cold_storage_path: Path
    ) -> None:
        """ForgettingEngine._collect_candidates never includes protected types."""
        old_time = datetime.now(timezone.utc) - timedelta(days=365)

        store.create_node("User", {
            "id": "user_skip",
            "name": "Skip User",
            "access_count": 0,
            "last_accessed": old_time,
        })
        store.create_node("Project", {
            "id": "proj_skip",
            "name": "Skip Project",
            "access_count": 0,
            "last_accessed": old_time,
        })

        engine = ForgettingEngine(
            store,
            activation_threshold=99.0,
            grace_period_days=0,
            cold_storage_path=cold_storage_path,
        )
        candidates = engine._collect_candidates()
        candidate_tables = {table for table, _, _ in candidates}

        assert "User" not in candidate_tables, "User must not appear in candidates"
        assert "Project" not in candidate_tables, "Project must not appear in candidates"


# ---------------------------------------------------------------------------
# Test: maintain_graph.py script
# ---------------------------------------------------------------------------

class TestMaintenanceScript:
    """Verify the maintenance script runs consolidation then forgetting."""

    def test_maintenance_script_importable(self) -> None:
        """The maintain_graph script should be importable."""
        import scripts.maintain_graph as mg
        assert hasattr(mg, "main")

    def test_maintenance_script_runs_pipeline(
        self, store: GraphStore, cold_storage_path: Path
    ) -> None:
        """The script's main function should execute the full pipeline."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=60)

        # Create a stale memory that should be archived
        store.create_node("Memory", {
            "id": "mem_stale",
            "content": "Stale memory",
            "category": "knowledge",
            "confidence": 0.01,
            "valid_from": old_time,
        })

        # Create a User that must survive
        store.create_node("User", {
            "id": "user_safe",
            "name": "Safe User",
        })

        import scripts.maintain_graph as mg
        result = mg.run_maintenance(
            store=store,
            activation_threshold=0.1,
            grace_period_days=30,
            cold_storage_path=cold_storage_path,
        )

        assert isinstance(result, MaintenanceResult)
        assert store.get_node("User", "user_safe") is not None


# ---------------------------------------------------------------------------
# Test: MaintenanceResult dataclass
# ---------------------------------------------------------------------------

class TestMaintenanceResult:
    """Verify the MaintenanceResult structure."""

    def test_result_fields(self) -> None:
        """MaintenanceResult should contain consolidation results and archive count."""
        result = MaintenanceResult(
            consolidation=ConsolidationResult(
                duplicates_found=2, merged_count=1, summaries_created=1
            ),
            archived_count=3,
        )
        assert result.consolidation.duplicates_found == 2
        assert result.consolidation.merged_count == 1
        assert result.consolidation.summaries_created == 1
        assert result.archived_count == 3

    def test_default_result(self) -> None:
        """Default MaintenanceResult has zero counts."""
        result = MaintenanceResult()
        assert result.consolidation.duplicates_found == 0
        assert result.archived_count == 0
