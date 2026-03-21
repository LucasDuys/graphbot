"""Tests for tools_gb.file.FileTool."""

from pathlib import Path

import pytest

from tools_gb.file import FileTool


@pytest.fixture()
def tool(tmp_path: Path) -> FileTool:
    return FileTool(workspace=str(tmp_path))


class TestRead:
    def test_read_file(self, tool: FileTool, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("line one\nline two\nline three\n", encoding="utf-8")

        result = tool.read("hello.txt")

        assert result["success"] is True
        assert result["lines"] == 3
        assert "     1 | line one" in result["content"]
        assert "     2 | line two" in result["content"]
        assert result["truncated"] is False

    def test_read_with_offset(self, tool: FileTool, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        lines = [f"line {i}" for i in range(100)]
        f.write_text("\n".join(lines), encoding="utf-8")

        result = tool.read("big.txt", offset=10, limit=5)

        assert result["success"] is True
        assert "    11 | line 10" in result["content"]
        assert "    15 | line 14" in result["content"]
        assert result["truncated"] is True

    def test_read_nonexistent(self, tool: FileTool) -> None:
        result = tool.read("no_such_file.txt")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_read_binary(self, tool: FileTool, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02\x03binary content")

        result = tool.read("data.bin")

        assert result["success"] is False
        assert "binary" in result["error"].lower()


class TestWrite:
    def test_write_creates_file(self, tool: FileTool, tmp_path: Path) -> None:
        result = tool.write("output.txt", "hello world")

        assert result["success"] is True
        assert result["bytes_written"] == len(b"hello world")
        assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "hello world"

    def test_write_creates_dirs(self, tool: FileTool, tmp_path: Path) -> None:
        result = tool.write("a/b/c/deep.txt", "nested content")

        assert result["success"] is True
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").exists()
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text(encoding="utf-8") == "nested content"


class TestEdit:
    def test_edit_replaces_text(self, tool: FileTool, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")

        result = tool.edit("code.py", "y = 2", "y = 42")

        assert result["success"] is True
        assert result["replacements"] == 1
        content = f.read_text(encoding="utf-8")
        assert "y = 42" in content
        assert "y = 2" not in content

    def test_edit_text_not_found(self, tool: FileTool, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("x = 1\n", encoding="utf-8")

        result = tool.edit("code.py", "nonexistent text", "replacement")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestListDir:
    def test_list_dir(self, tool: FileTool, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        (tmp_path / "c.py").write_text("c", encoding="utf-8")

        result = tool.list_dir(".")

        assert result["success"] is True
        assert result["count"] == 3
        assert "a.txt" in result["files"]
        assert "b.txt" in result["files"]
        assert "c.py" in result["files"]


class TestSearch:
    def test_search_finds_matches(self, tool: FileTool, tmp_path: Path) -> None:
        f = tmp_path / "haystack.txt"
        f.write_text("first line\nneedle here\nthird line\nneedle again\n", encoding="utf-8")

        result = tool.search(".", "needle")

        assert result["success"] is True
        assert result["count"] == 2
        line_numbers = [r["line_number"] for r in result["results"]]
        assert 2 in line_numbers
        assert 4 in line_numbers


class TestSecurity:
    def test_path_traversal_blocked(self, tool: FileTool) -> None:
        result = tool.read("../../../etc/passwd")

        assert result["success"] is False
        assert "outside workspace" in result["error"].lower()

    def test_path_traversal_blocked_write(self, tool: FileTool) -> None:
        result = tool.write("../../../tmp/evil.txt", "pwned")

        assert result["success"] is False
        assert "outside workspace" in result["error"].lower()

    def test_path_traversal_blocked_edit(self, tool: FileTool) -> None:
        result = tool.edit("../../../etc/hosts", "old", "new")

        assert result["success"] is False
        assert "outside workspace" in result["error"].lower()
