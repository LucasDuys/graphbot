"""File operation tools for GraphBot DAG leaf execution.

Wraps Nanobot's filesystem tools with a simpler interface for DAG leaf tasks.
"""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileTool:
    """File operations for DAG leaf execution."""

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = Path(workspace).resolve() if workspace else Path.cwd().resolve()

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> dict[str, Any]:
        """Read a file with line numbers.

        Returns: {success, content, lines, path, truncated, error}
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"success": False, "error": f"Path outside workspace: {path}"}

        if not resolved.exists():
            return {"success": False, "error": f"File not found: {path}"}

        if not resolved.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        if self._is_binary(resolved):
            return {"success": False, "error": f"Binary file: {path}"}

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)

            selected = lines[offset : offset + limit]
            numbered = [f"{i + offset + 1:>6} | {line}" for i, line in enumerate(selected)]
            content = "\n".join(numbered)

            truncated = total > offset + limit
            if truncated:
                content += f"\n\n... ({total - offset - limit} more lines, use offset={offset + limit})"

            return {
                "success": True,
                "content": content,
                "lines": total,
                "path": str(resolved),
                "truncated": truncated,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def write(self, path: str, content: str) -> dict[str, Any]:
        """Write content to a file (creates directories if needed).

        Returns: {success, path, bytes_written, error}
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"success": False, "error": f"Path outside workspace: {path}"}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return {
                "success": True,
                "path": str(resolved),
                "bytes_written": len(content.encode("utf-8")),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def edit(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        """Search-and-replace edit in a file.

        Returns: {success, path, replacements, error}
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"success": False, "error": f"Path outside workspace: {path}"}

        if not resolved.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            count = content.count(old_text)

            if count == 0:
                return {
                    "success": False,
                    "error": f"Text not found in file. File has {len(content)} chars.",
                }

            new_content = content.replace(old_text, new_text, 1)
            resolved.write_text(new_content, encoding="utf-8")
            return {"success": True, "path": str(resolved), "replacements": 1}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_dir(self, directory: str = ".", pattern: str = "*") -> dict[str, Any]:
        """List files in a directory matching a pattern.

        Returns: {success, files: [str], count, error}
        """
        resolved = self._resolve_path(directory)
        if resolved is None:
            return {"success": False, "error": f"Path outside workspace: {directory}"}

        if not resolved.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        try:
            files = sorted(
                str(p.relative_to(resolved)) for p in resolved.glob(pattern) if p.is_file()
            )
            return {"success": True, "files": files[:500], "count": len(files)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def search(
        self,
        directory: str,
        query: str,
        extensions: list[str] | None = None,
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Search file contents for a query string (grep-like).

        Returns: {success, results: [{path, line_number, line}], count, error}
        """
        resolved = self._resolve_path(directory)
        if resolved is None:
            return {"success": False, "error": f"Path outside workspace: {directory}"}

        if not resolved.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        results: list[dict[str, Any]] = []
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

            for filepath in resolved.rglob("*"):
                if not filepath.is_file():
                    continue
                if extensions and filepath.suffix not in extensions:
                    continue
                if self._is_binary(filepath):
                    continue

                try:
                    text = filepath.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pattern.search(line):
                            results.append({
                                "path": str(filepath.relative_to(resolved)),
                                "line_number": i,
                                "line": line.strip()[:200],
                            })
                            if len(results) >= max_results:
                                return {
                                    "success": True,
                                    "results": results,
                                    "count": len(results),
                                    "truncated": True,
                                }
                except (UnicodeDecodeError, PermissionError):
                    continue

            return {
                "success": True,
                "results": results,
                "count": len(results),
                "truncated": False,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _resolve_path(self, path: str) -> Path | None:
        """Resolve path and verify it's within workspace."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self._workspace / p
            resolved = p.resolve()
            resolved.relative_to(self._workspace)
            return resolved
        except (ValueError, OSError):
            return None

    @staticmethod
    def _is_binary(path: Path) -> bool:
        """Check if file is binary by reading first 8KB."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
            return b"\x00" in chunk
        except (OSError, PermissionError):
            return True
