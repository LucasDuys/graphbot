"""Forgetting engine: archives stale nodes to cold storage JSON.

Nodes below the activation threshold that have exceeded the grace period are
removed from the live Kuzu graph and written to a cold storage JSON file.
Protected node types (User, Project) are never forgotten. Archived nodes can
be restored on demand.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from graph.schema import NODE_TYPES
from graph.store import GraphStore

logger = logging.getLogger(__name__)

# Node types that must never be archived.
PROTECTED_TYPES: frozenset[str] = frozenset({"User", "Project"})

# Default cold storage location relative to the project root.
_DEFAULT_COLD_STORAGE = Path("data/cold_storage.json")

# Cold storage format version for forward compatibility.
_COLD_STORAGE_VERSION = 1


class ForgettingEngine:
    """Archives inactive graph nodes to cold storage and supports recovery.

    The engine scans all non-protected node types for nodes that:
      1. Have been inactive longer than the grace period.
      2. Have an activation score below the threshold.

    Activation score is derived from the node's ``confidence`` property for
    Memory nodes. For other node types, nodes older than the grace period
    with no confidence field default to 0.0 activation.

    Archived nodes are written to a JSON file (cold storage) and removed
    from the live graph. They can be restored via ``restore()``.

    Args:
        store: The Kuzu graph store to sweep.
        activation_threshold: Minimum activation score to keep a node (default 0.1).
        grace_period_days: Days of inactivity before a node is eligible for archival (default 30).
        cold_storage_path: Path to the cold storage JSON file.
    """

    def __init__(
        self,
        store: GraphStore,
        activation_threshold: float = 0.1,
        grace_period_days: int = 30,
        cold_storage_path: Path | None = None,
    ) -> None:
        self._store = store
        self.activation_threshold = activation_threshold
        self.grace_period_days = grace_period_days
        self.cold_storage_path = cold_storage_path or _DEFAULT_COLD_STORAGE

    def _get_activation(self, table: str, props: dict[str, object]) -> float:
        """Compute the activation score for a node.

        Memory nodes use their ``confidence`` field directly. Other node types
        default to 0.0 (eligible for archival once past grace period).
        """
        if table == "Memory":
            confidence = props.get("confidence")
            if confidence is not None:
                try:
                    return float(confidence)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _get_node_age_days(self, table: str, props: dict[str, object]) -> float | None:
        """Return the age of a node in days, or None if no timestamp is available.

        Uses ``valid_from`` for Memory nodes and ``created_at`` for Task and
        ExecutionTree nodes. Falls back to checking both fields for other types.
        """
        now = datetime.now(timezone.utc)
        timestamp_fields = []

        if table == "Memory":
            timestamp_fields = ["valid_from"]
        elif table in ("Task", "ExecutionTree"):
            timestamp_fields = ["created_at"]
        else:
            timestamp_fields = ["created_at", "valid_from"]

        for field in timestamp_fields:
            value = props.get(field)
            if value is None:
                continue

            ts: datetime | None = None
            if isinstance(value, datetime):
                ts = value
            elif isinstance(value, str) and value.strip():
                try:
                    ts = datetime.fromisoformat(value)
                except ValueError:
                    continue

            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                delta = now - ts
                return delta.total_seconds() / 86400.0

        return None

    def _collect_candidates(self) -> list[tuple[str, str, dict[str, object]]]:
        """Collect all nodes eligible for archival.

        Returns a list of (table, node_id, properties) tuples for nodes that:
          - Are not in a protected type
          - Have exceeded the grace period
          - Have activation below the threshold
        """
        candidates: list[tuple[str, str, dict[str, object]]] = []

        for node_type in NODE_TYPES:
            if node_type.name in PROTECTED_TYPES:
                continue

            try:
                rows = self._store.query(
                    f"MATCH (n:{node_type.name}) RETURN n.*"
                )
            except Exception:
                logger.warning("Failed to query table %s during sweep", node_type.name)
                continue

            for row in rows:
                # Parse properties from column names like "n.prop"
                props: dict[str, object] = {}
                for col_name, value in row.items():
                    key = col_name.split(".", 1)[1] if "." in col_name else col_name
                    props[key] = value

                node_id = str(props.get("id", ""))
                if not node_id:
                    continue

                # Check grace period
                age_days = self._get_node_age_days(node_type.name, props)
                if age_days is not None and age_days < self.grace_period_days:
                    continue

                # If no timestamp is available, only archive if grace_period is 0
                if age_days is None and self.grace_period_days > 0:
                    continue

                # Check activation threshold
                activation = self._get_activation(node_type.name, props)
                if activation >= self.activation_threshold:
                    continue

                candidates.append((node_type.name, node_id, props))

        return candidates

    def _serialize_props(self, props: dict[str, object]) -> dict[str, Any]:
        """Serialize node properties to JSON-safe types."""
        result: dict[str, Any] = {}
        for key, value in props.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif value is None:
                result[key] = None
            else:
                result[key] = value
        return result

    def _load_cold_storage(self) -> dict[str, Any]:
        """Load existing cold storage data, or create a fresh structure."""
        if self.cold_storage_path.exists():
            try:
                with open(self.cold_storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Cold storage file corrupted, starting fresh: %s",
                    self.cold_storage_path,
                )
        return {"version": _COLD_STORAGE_VERSION, "archived_nodes": []}

    def _save_cold_storage(self, data: dict[str, Any]) -> None:
        """Write cold storage data to the JSON file."""
        self.cold_storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cold_storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def sweep(self) -> int:
        """Archive stale nodes from the graph to cold storage.

        Scans all non-protected node types for nodes past the grace period
        with activation below the threshold. Archives them to cold storage
        and removes them from the live graph.

        Returns:
            The number of nodes archived.
        """
        candidates = self._collect_candidates()
        if not candidates:
            return 0

        cold_data = self._load_cold_storage()
        now = datetime.now(timezone.utc)
        archived_count = 0

        for table, node_id, props in candidates:
            entry: dict[str, Any] = {
                "node_id": node_id,
                "table": table,
                "properties": self._serialize_props(props),
                "archived_at": now.isoformat(),
            }
            cold_data["archived_nodes"].append(entry)

            try:
                self._store.delete_node(table, node_id)
                archived_count += 1
                logger.info("Archived %s node %s to cold storage", table, node_id)
            except Exception:
                logger.exception(
                    "Failed to delete %s node %s after archiving", table, node_id
                )
                # Remove the entry we just added since deletion failed
                cold_data["archived_nodes"].pop()

        if archived_count > 0:
            self._save_cold_storage(cold_data)
            logger.info(
                "Sweep complete: archived %d nodes to %s",
                archived_count,
                self.cold_storage_path,
            )

        return archived_count

    def restore(self, node_id: str) -> bool:
        """Restore an archived node from cold storage to the live graph.

        Finds the node in cold storage by ID, re-creates it in the graph,
        and removes the entry from cold storage.

        Args:
            node_id: The ID of the node to restore.

        Returns:
            True if the node was found and restored, False otherwise.
        """
        if not self.cold_storage_path.exists():
            return False

        cold_data = self._load_cold_storage()
        archived_nodes: list[dict[str, Any]] = cold_data.get("archived_nodes", [])

        # Find the entry by node_id
        target_index: int | None = None
        for i, entry in enumerate(archived_nodes):
            if entry.get("node_id") == node_id:
                target_index = i
                break

        if target_index is None:
            return False

        entry = archived_nodes[target_index]
        table: str = entry["table"]
        props: dict[str, object] = entry["properties"]

        # Re-create the node in the graph
        try:
            self._store.create_node(table, props)
        except Exception:
            logger.exception(
                "Failed to restore %s node %s to graph", table, node_id
            )
            return False

        # Remove the entry from cold storage
        archived_nodes.pop(target_index)
        self._save_cold_storage(cold_data)

        logger.info("Restored %s node %s from cold storage", table, node_id)
        return True
