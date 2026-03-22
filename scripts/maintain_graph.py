"""Run graph maintenance: consolidation followed by forgetting.

This script is the primary entry point for scheduled graph maintenance.
It runs consolidation first (dedup, merge, summarize) and then forgetting
(archive stale nodes), ensuring the graph stays clean and efficient.

Usage:
    python -m scripts.maintain_graph
    python -m scripts.maintain_graph --threshold 0.2 --grace-days 14
    python -m scripts.maintain_graph --cold-storage data/archive.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from graph.maintenance import GraphMaintenance, MaintenanceResult
from graph.store import GraphStore

logger = logging.getLogger(__name__)

# Default database and cold storage paths
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "graphbot.db"
_DEFAULT_COLD_STORAGE = _PROJECT_ROOT / "data" / "cold_storage.json"


def run_maintenance(
    store: GraphStore | None = None,
    db_path: Path | None = None,
    activation_threshold: float = 0.1,
    grace_period_days: int = 30,
    cold_storage_path: Path | None = None,
    duplicate_threshold: float = 0.95,
) -> MaintenanceResult:
    """Run the full maintenance pipeline on a graph store.

    This is the programmatic API for running maintenance. It can accept
    either an existing GraphStore instance or a database path.

    Args:
        store: An existing GraphStore instance. If None, one is created from db_path.
        db_path: Path to the Kuzu database. Ignored if store is provided.
        activation_threshold: Minimum activation score to keep a node (default 0.1).
        grace_period_days: Days of inactivity before archival eligibility (default 30).
        cold_storage_path: Path for the cold storage JSON file.
        duplicate_threshold: Minimum resolver confidence for duplicate detection.

    Returns:
        MaintenanceResult with consolidation and forgetting results.
    """
    owns_store = False
    if store is None:
        resolved_db = str(db_path or _DEFAULT_DB_PATH)
        store = GraphStore(db_path=resolved_db)
        store.initialize()
        owns_store = True

    resolved_cold = cold_storage_path or _DEFAULT_COLD_STORAGE

    try:
        maintenance = GraphMaintenance(
            store,
            activation_threshold=activation_threshold,
            grace_period_days=grace_period_days,
            cold_storage_path=resolved_cold,
            duplicate_threshold=duplicate_threshold,
        )
        result = maintenance.run()

        logger.info(
            "Maintenance complete: %d duplicates found, %d merged, "
            "%d summaries created, %d nodes archived",
            result.consolidation.duplicates_found,
            result.consolidation.merged_count,
            result.consolidation.summaries_created,
            result.archived_count,
        )

        return result
    finally:
        if owns_store:
            store.close()


def main() -> None:
    """CLI entry point for graph maintenance."""
    parser = argparse.ArgumentParser(
        description="Run graph maintenance: consolidation then forgetting."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=_DEFAULT_DB_PATH,
        help=f"Path to Kuzu database (default: {_DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Activation threshold for forgetting (default: 0.1)",
    )
    parser.add_argument(
        "--grace-days",
        type=int,
        default=30,
        help="Grace period in days before forgetting eligibility (default: 30)",
    )
    parser.add_argument(
        "--cold-storage",
        type=Path,
        default=_DEFAULT_COLD_STORAGE,
        help=f"Cold storage JSON path (default: {_DEFAULT_COLD_STORAGE})",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.95,
        help="Minimum confidence for duplicate detection (default: 0.95)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_maintenance(
        db_path=args.db_path,
        activation_threshold=args.threshold,
        grace_period_days=args.grace_days,
        cold_storage_path=args.cold_storage,
        duplicate_threshold=args.duplicate_threshold,
    )

    print(
        f"Maintenance complete:\n"
        f"  Duplicates found: {result.consolidation.duplicates_found}\n"
        f"  Nodes merged:     {result.consolidation.merged_count}\n"
        f"  Summaries created: {result.consolidation.summaries_created}\n"
        f"  Nodes archived:   {result.archived_count}"
    )

    if result.consolidation.errors:
        print(f"\n  Errors ({len(result.consolidation.errors)}):")
        for err in result.consolidation.errors:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
