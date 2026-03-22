"""Transactional execution wrapper for file and shell operations.

Provides snapshot-and-rollback semantics for DAG node execution. Before a
file write or shell command executes, the TransactionManager captures the
target state. On failure or safety violation, it restores the snapshot.

Rollback capabilities:
  - File operations: Restores original file content (or deletes if the file
    did not exist before the operation).
  - Shell operations: Best-effort only. Logs a warning since shell side
    effects are generally not reversible.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SnapshotType(str, Enum):
    """Type of state snapshot captured before an operation."""

    FILE = "file"
    SHELL = "shell"


@dataclass
class Snapshot:
    """Captured state before an operation, used for rollback.

    Attributes:
        snapshot_type: Whether this is a file or shell snapshot.
        target_path: For file operations, the absolute path of the target file.
            None for shell operations.
        original_content: For file operations, the original file content before
            the operation. None if the file did not exist.
        file_existed: Whether the target file existed before the operation.
        shell_command: For shell operations, the command that was executed.
        metadata: Additional metadata about the snapshot (e.g., permissions).
    """

    snapshot_type: SnapshotType
    target_path: str | None = None
    original_content: str | None = None
    file_existed: bool = False
    shell_command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransactionResult:
    """Result of a transactional operation.

    Attributes:
        success: Whether the operation completed successfully.
        rolled_back: Whether a rollback was performed.
        rollback_success: Whether the rollback itself succeeded. None if no
            rollback was attempted.
        error: Error message if the operation or rollback failed.
        snapshot: The snapshot that was captured before the operation.
    """

    success: bool
    rolled_back: bool = False
    rollback_success: bool | None = None
    error: str = ""
    snapshot: Snapshot | None = None


class TransactionManager:
    """Manages transactional execution with snapshot and rollback.

    Captures the state of targets before operations execute, and restores
    that state on failure. File operations get full content restoration;
    shell operations get best-effort logging only.

    Usage::

        txn = TransactionManager()
        snapshot = txn.snapshot_file("/path/to/file.txt")
        # ... perform operation ...
        if operation_failed:
            txn.rollback(snapshot)
    """

    def snapshot_file(self, file_path: str) -> Snapshot:
        """Capture the current state of a file before modification.

        If the file exists, reads and stores its content. If it does not
        exist, records that fact so rollback can delete the newly created
        file.

        Args:
            file_path: Absolute path to the file that will be modified.

        Returns:
            Snapshot containing the file's original state.
        """
        resolved = str(Path(file_path).resolve())
        snapshot = Snapshot(
            snapshot_type=SnapshotType.FILE,
            target_path=resolved,
        )

        if os.path.isfile(resolved):
            try:
                with open(resolved, "r", encoding="utf-8") as f:
                    snapshot.original_content = f.read()
                snapshot.file_existed = True
                logger.debug(
                    "Snapshot captured for existing file: %s (%d bytes)",
                    resolved,
                    len(snapshot.original_content),
                )
            except (OSError, UnicodeDecodeError) as exc:
                # For binary files or read errors, store what we can
                logger.warning(
                    "Could not read file for snapshot (will attempt binary): %s: %s",
                    resolved,
                    exc,
                )
                try:
                    with open(resolved, "rb") as fb:
                        binary_content = fb.read()
                    snapshot.original_content = binary_content.decode("latin-1")
                    snapshot.file_existed = True
                    snapshot.metadata["encoding"] = "latin-1"
                except OSError as binary_exc:
                    logger.error(
                        "Failed to read file for snapshot: %s: %s",
                        resolved,
                        binary_exc,
                    )
                    snapshot.file_existed = True
                    snapshot.original_content = None
        else:
            snapshot.file_existed = False
            logger.debug(
                "Snapshot captured for non-existent file: %s",
                resolved,
            )

        return snapshot

    def snapshot_shell(self, command: str) -> Snapshot:
        """Record a shell command before execution for rollback logging.

        Shell operations cannot be reliably rolled back. This method records
        the command so that a best-effort warning can be logged during
        rollback.

        Args:
            command: The shell command that will be executed.

        Returns:
            Snapshot containing the command for logging purposes.
        """
        snapshot = Snapshot(
            snapshot_type=SnapshotType.SHELL,
            shell_command=command,
        )
        logger.debug(
            "Snapshot captured for shell command: %s",
            command[:200],
        )
        return snapshot

    def rollback(self, snapshot: Snapshot) -> TransactionResult:
        """Rollback to the captured snapshot state.

        For file snapshots: restores original content or deletes newly
        created files. For shell snapshots: logs a warning (best-effort).

        Args:
            snapshot: The snapshot to rollback to.

        Returns:
            TransactionResult indicating whether rollback succeeded.
        """
        if snapshot.snapshot_type == SnapshotType.FILE:
            return self._rollback_file(snapshot)
        elif snapshot.snapshot_type == SnapshotType.SHELL:
            return self._rollback_shell(snapshot)
        else:
            return TransactionResult(
                success=False,
                rolled_back=False,
                error=f"Unknown snapshot type: {snapshot.snapshot_type}",
                snapshot=snapshot,
            )

    def _rollback_file(self, snapshot: Snapshot) -> TransactionResult:
        """Rollback a file operation to its snapshot state.

        If the file existed before: restores original content.
        If the file did not exist before: deletes the newly created file.
        If original content could not be captured: logs warning, no rollback.

        Args:
            snapshot: File snapshot to rollback to.

        Returns:
            TransactionResult indicating rollback outcome.
        """
        target = snapshot.target_path
        if target is None:
            return TransactionResult(
                success=False,
                rolled_back=False,
                error="File snapshot has no target path",
                snapshot=snapshot,
            )

        try:
            if snapshot.file_existed:
                if snapshot.original_content is not None:
                    encoding = snapshot.metadata.get("encoding", "utf-8")
                    with open(target, "w", encoding=encoding) as f:
                        f.write(snapshot.original_content)
                    logger.info(
                        "Rolled back file to original content: %s", target,
                    )
                    return TransactionResult(
                        success=True,
                        rolled_back=True,
                        rollback_success=True,
                        snapshot=snapshot,
                    )
                else:
                    logger.warning(
                        "Cannot rollback file %s: original content was not "
                        "captured during snapshot",
                        target,
                    )
                    return TransactionResult(
                        success=False,
                        rolled_back=True,
                        rollback_success=False,
                        error="Original content not available for rollback",
                        snapshot=snapshot,
                    )
            else:
                # File did not exist before -- delete the newly created file
                if os.path.isfile(target):
                    os.remove(target)
                    logger.info(
                        "Rolled back by deleting newly created file: %s",
                        target,
                    )
                else:
                    logger.debug(
                        "Rollback: file already does not exist: %s", target,
                    )
                return TransactionResult(
                    success=True,
                    rolled_back=True,
                    rollback_success=True,
                    snapshot=snapshot,
                )
        except OSError as exc:
            logger.error(
                "Rollback failed for file %s: %s", target, exc,
            )
            return TransactionResult(
                success=False,
                rolled_back=True,
                rollback_success=False,
                error=f"Rollback failed: {exc}",
                snapshot=snapshot,
            )

    def _rollback_shell(self, snapshot: Snapshot) -> TransactionResult:
        """Best-effort rollback for shell operations.

        Shell commands cannot be reliably undone. Logs a warning with the
        command that was executed so operators can take manual action if
        needed.

        Args:
            snapshot: Shell snapshot to rollback.

        Returns:
            TransactionResult with rollback_success=None (best-effort).
        """
        logger.warning(
            "Shell command rollback is best-effort only. "
            "The following command was executed and may have side effects "
            "that cannot be automatically reversed: %s",
            snapshot.shell_command,
        )
        return TransactionResult(
            success=True,
            rolled_back=True,
            rollback_success=None,
            error="Shell rollback is best-effort; side effects may persist",
            snapshot=snapshot,
        )
