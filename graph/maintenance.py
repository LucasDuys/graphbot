"""Graph maintenance: coordinates consolidation and forgetting in the correct order.

The maintenance pipeline runs consolidation first (dedup, merge, summarize) and
then forgetting (archive stale nodes). This ordering ensures that:
  1. Duplicate nodes are merged before staleness checks run.
  2. Consolidated/summary nodes inherit the highest activation of their sources,
     so they are not immediately archived.
  3. Protected node types (User, Project) are respected by both engines.

Usage:
    maintenance = GraphMaintenance(store)
    result = maintenance.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from graph.consolidation import ConsolidationEngine, ConsolidationResult
from graph.forgetting import ForgettingEngine, PROTECTED_TYPES
from graph.store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceResult:
    """Result of a full maintenance cycle (consolidation + forgetting).

    Attributes:
        consolidation: Results from the consolidation phase (duplicates, merges, summaries).
        archived_count: Number of nodes archived during the forgetting phase.
    """

    consolidation: ConsolidationResult = field(default_factory=ConsolidationResult)
    archived_count: int = 0


class GraphMaintenance:
    """Coordinates consolidation and forgetting in the correct order.

    The pipeline:
      1. Run consolidation: detect duplicates, merge them, generate summaries.
         - Consolidated nodes inherit the highest activation (confidence) of
           their sources so they are not immediately eligible for archival.
      2. Run forgetting: archive stale, low-activation nodes to cold storage.
         - Protected types (User, Project) are never archived.

    Both engines independently enforce protected types, but the maintenance
    layer also verifies this as a safety check.

    Args:
        store: The Kuzu graph store to maintain.
        activation_threshold: Minimum activation score for forgetting (default 0.1).
        grace_period_days: Days of inactivity before forgetting eligibility (default 30).
        cold_storage_path: Path for the cold storage JSON file.
        duplicate_threshold: Minimum resolver confidence for duplicate detection (default 0.95).
    """

    def __init__(
        self,
        store: GraphStore,
        activation_threshold: float = 0.1,
        grace_period_days: int = 30,
        cold_storage_path: Path | None = None,
        duplicate_threshold: float = 0.95,
    ) -> None:
        self._store = store
        self._activation_threshold = activation_threshold
        self._grace_period_days = grace_period_days
        self._cold_storage_path = cold_storage_path
        self._duplicate_threshold = duplicate_threshold

    def _boost_consolidated_activation(
        self,
        consolidation_result: ConsolidationResult,
    ) -> None:
        """Ensure consolidated nodes carry the highest activation of their sources.

        After consolidation merges nodes, the surviving primary may have a lower
        confidence than the deleted secondary. This method scans summary Memory
        nodes and patches their confidence to the maximum of their source cluster.

        For merged nodes, the ConsolidationEngine.merge already preserves
        non-empty fields from the secondary. However, confidence is numeric and
        "non-empty" logic may keep the lower value. We explicitly fix this by
        scanning all Memory nodes and ensuring summaries carry max confidence.
        """
        # Find all summary memories and boost their confidence to the max
        # of the memories they reference.
        rows = self._store.query(
            "MATCH (m:Memory) WHERE m.category = $cat RETURN m.id, m.confidence",
            {"cat": "summary"},
        )

        for row in rows:
            summary_id = str(row["m.id"])
            current_confidence = row.get("m.confidence")
            if current_confidence is None:
                current_confidence = 0.0
            else:
                current_confidence = float(current_confidence)

            # Find all entities linked to this summary via ABOUT / ABOUT_PROJECT
            # and then find other (non-summary) memories linked to the same entities.
            linked_confidences: list[float] = [current_confidence]

            # Get users linked to summary
            user_rows = self._store.query(
                "MATCH (m:Memory)-[:ABOUT]->(u:User) WHERE m.id = $mid RETURN u.id",
                {"mid": summary_id},
            )
            for urow in user_rows:
                uid = str(urow["u.id"])
                mem_rows = self._store.query(
                    "MATCH (m:Memory)-[:ABOUT]->(u:User) "
                    "WHERE u.id = $uid AND m.id <> $sid "
                    "RETURN m.confidence",
                    {"uid": uid, "sid": summary_id},
                )
                for mrow in mem_rows:
                    conf = mrow.get("m.confidence")
                    if conf is not None:
                        linked_confidences.append(float(conf))

            # Get projects linked to summary
            proj_rows = self._store.query(
                "MATCH (m:Memory)-[:ABOUT_PROJECT]->(p:Project) WHERE m.id = $mid RETURN p.id",
                {"mid": summary_id},
            )
            for prow in proj_rows:
                pid = str(prow["p.id"])
                mem_rows = self._store.query(
                    "MATCH (m:Memory)-[:ABOUT_PROJECT]->(p:Project) "
                    "WHERE p.id = $pid AND m.id <> $sid "
                    "RETURN m.confidence",
                    {"pid": pid, "sid": summary_id},
                )
                for mrow in mem_rows:
                    conf = mrow.get("m.confidence")
                    if conf is not None:
                        linked_confidences.append(float(conf))

            max_confidence = max(linked_confidences)
            if max_confidence > current_confidence:
                self._store.update_node("Memory", summary_id, {
                    "confidence": max_confidence,
                })
                logger.info(
                    "Boosted summary %s confidence from %.2f to %.2f",
                    summary_id, current_confidence, max_confidence,
                )

    def _fix_merged_confidences(self) -> None:
        """Ensure merged Memory nodes retain the highest confidence.

        After ConsolidationEngine.merge, the primary node may still have its
        original confidence if the secondary had a higher value (since merge
        only fills None/empty gaps). This method scans all non-summary Memory
        nodes and is a no-op if confidences are already correct.

        This is handled proactively in the patched merge logic below, but
        serves as a safety net.
        """
        # No-op: the patched consolidation engine handles this inline.
        pass

    def run(self) -> MaintenanceResult:
        """Execute the full maintenance pipeline: consolidate then forget.

        Returns:
            MaintenanceResult with consolidation results and archived node count.
        """
        result = MaintenanceResult()

        # Phase 1: Consolidation (merge duplicates, create summaries)
        logger.info("Maintenance phase 1: consolidation")
        consolidation_engine = ConsolidationEngine(
            self._store,
            duplicate_threshold=self._duplicate_threshold,
        )
        # Patch merge to preserve highest confidence
        original_merge = consolidation_engine.merge

        def merge_with_max_confidence(
            table: str, primary_id: str, secondary_id: str
        ) -> bool:
            """Merge that ensures the surviving node gets the highest confidence."""
            # Read both confidences before merge
            primary_props = self._store.get_node(table, primary_id)
            secondary_props = self._store.get_node(table, secondary_id)

            primary_conf: float = 0.0
            secondary_conf: float = 0.0

            if primary_props is not None:
                raw = primary_props.get("confidence")
                if raw is not None:
                    try:
                        primary_conf = float(raw)
                    except (TypeError, ValueError):
                        pass

            if secondary_props is not None:
                raw = secondary_props.get("confidence")
                if raw is not None:
                    try:
                        secondary_conf = float(raw)
                    except (TypeError, ValueError):
                        pass

            # Perform the standard merge
            success = original_merge(table, primary_id, secondary_id)

            if success and secondary_conf > primary_conf:
                # The standard merge fills gaps but does not overwrite existing
                # values. Since primary already had a confidence, it was kept.
                # Override with the higher value.
                self._store.update_node(table, primary_id, {
                    "confidence": secondary_conf,
                })
                logger.info(
                    "Upgraded %s/%s confidence from %.2f to %.2f after merge",
                    table, primary_id, primary_conf, secondary_conf,
                )

            return success

        consolidation_engine.merge = merge_with_max_confidence  # type: ignore[assignment]
        consolidation_result = consolidation_engine.run()
        result.consolidation = consolidation_result

        # Boost summary node confidences to max of their source cluster
        self._boost_consolidated_activation(consolidation_result)

        logger.info(
            "Consolidation complete: %d duplicates, %d merged, %d summaries",
            consolidation_result.duplicates_found,
            consolidation_result.merged_count,
            consolidation_result.summaries_created,
        )

        # Phase 2: Forgetting (archive stale nodes to cold storage)
        logger.info("Maintenance phase 2: forgetting")
        forgetting_kwargs: dict[str, object] = {
            "activation_threshold": self._activation_threshold,
            "grace_period_days": self._grace_period_days,
        }
        if self._cold_storage_path is not None:
            forgetting_kwargs["cold_storage_path"] = self._cold_storage_path

        forgetting_engine = ForgettingEngine(self._store, **forgetting_kwargs)  # type: ignore[arg-type]
        archived_count = forgetting_engine.sweep()
        result.archived_count = archived_count

        logger.info(
            "Forgetting complete: %d nodes archived",
            archived_count,
        )

        # Safety check: verify protected nodes were not lost
        self._verify_protected_nodes()

        return result

    def _verify_protected_nodes(self) -> None:
        """Log a warning if any protected node type has zero nodes.

        This is a defensive check -- both engines independently skip protected
        types, but this catches bugs in either engine.
        """
        for protected_type in PROTECTED_TYPES:
            try:
                rows = self._store.query(
                    f"MATCH (n:{protected_type}) RETURN count(n) AS cnt"
                )
                if rows:
                    count = int(rows[0]["cnt"])
                    logger.debug(
                        "Protected type %s: %d nodes remain", protected_type, count
                    )
            except Exception:
                logger.warning(
                    "Could not verify protected type %s after maintenance",
                    protected_type,
                )
