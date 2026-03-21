"""Tests for tools_gb.web -- WebTool fetch, search, and HTML conversion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tools_gb.web import WebTool

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Main Heading</h1>
<p>This is a <a href="https://example.com">link</a> in a paragraph.</p>
<ul>
  <li>Item one</li>
  <li>Item two</li>
</ul>
<script>alert('bad');</script>
<style>.hidden { display: none; }</style>
<h2>Subheading</h2>
<p>More content here with &amp; entities.</p>
</body>
</html>
"""


@pytest.fixture
def tool() -> WebTool:
    return WebTool(max_tokens=4000, timeout=10)


def _mock_response(
    text: str = SAMPLE_HTML,
    status_code: int = 200,
    url: str = "https://example.com/page",
) -> httpx.Response:
    """Build a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        text=text,
        request=httpx.Request("GET", url),
    )
    return resp


# -- test_fetch_valid_url --


@pytest.mark.asyncio
async def test_fetch_valid_url(tool: WebTool) -> None:
    """Mock httpx to return sample HTML, verify clean markdown output."""
    mock_resp = _mock_response()

    async def fake_get(self, url, **kwargs):  # noqa: ANN001, ANN003
        return mock_resp

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await tool.fetch("https://example.com/page")

    assert result["success"] is True
    assert "Main Heading" in result["content"]
    assert "[link]" in result["content"]
    assert result["title"]
    assert result["tokens"] >= 1
    assert result["truncated"] is False
    # Script content must be stripped
    assert "alert" not in result["content"]


# -- test_fetch_invalid_url --


@pytest.mark.asyncio
async def test_fetch_invalid_url(tool: WebTool) -> None:
    """ftp:// URLs should return error without making a request."""
    result = await tool.fetch("ftp://files.example.com/data")
    assert result["success"] is False
    assert "Invalid URL scheme" in result["error"]


# -- test_fetch_http_error --


@pytest.mark.asyncio
async def test_fetch_http_error(tool: WebTool) -> None:
    """404 response should return error with HTTP status."""

    async def fake_get(self, url, **kwargs):  # noqa: ANN001, ANN003
        resp = _mock_response(text="Not Found", status_code=404, url=url)
        resp.raise_for_status()  # will raise
        return resp  # pragma: no cover

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await tool.fetch("https://example.com/missing")

    assert result["success"] is False
    assert "404" in result["error"]


# -- test_search_returns_results --


@pytest.mark.asyncio
async def test_search_returns_results(tool: WebTool) -> None:
    """Mock ddgs.text, verify structured results."""
    fake_raw = [
        {"title": "Result 1", "href": "https://r1.com", "body": "Snippet one"},
        {"title": "Result 2", "href": "https://r2.com", "body": "Snippet two"},
    ]

    with patch("tools_gb.web.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = fake_raw
        result = await tool.search("test query", max_results=5)

    assert result["success"] is True
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Result 1"
    assert result["results"][0]["url"] == "https://r1.com"
    assert result["results"][1]["snippet"] == "Snippet two"


# -- test_search_and_summarize --


@pytest.mark.asyncio
async def test_search_and_summarize(tool: WebTool) -> None:
    """Mock search + fetch, verify combined output."""
    search_return = {
        "success": True,
        "results": [
            {"title": "Page A", "url": "https://a.com", "snippet": "snip a"},
            {"title": "Page B", "url": "https://b.com", "snippet": "snip b"},
        ],
    }
    fetch_return = {
        "success": True,
        "content": "Fetched content here",
        "title": "Page",
        "url": "https://a.com",
        "tokens": 10,
        "truncated": False,
    }

    with (
        patch.object(tool, "search", new_callable=AsyncMock, return_value=search_return),
        patch.object(tool, "fetch", new_callable=AsyncMock, return_value=fetch_return),
    ):
        result = await tool.search_and_summarize("test", max_results=2)

    assert result["success"] is True
    assert len(result["sources"]) == 2
    assert "Fetched content here" in result["content"]
    assert result["tokens"] >= 1
    assert "---" in result["content"]  # separator between sources


# -- test_html_to_markdown --


def test_html_to_markdown(tool: WebTool) -> None:
    """Test conversion with headings, links, lists, scripts."""
    md = tool._html_to_markdown(SAMPLE_HTML)

    # Headings converted
    assert "# Main Heading" in md
    assert "## Subheading" in md
    # Links converted
    assert "[link](https://example.com)" in md
    # List items converted
    assert "- Item one" in md
    assert "- Item two" in md
    # Scripts removed
    assert "alert" not in md
    assert "bad" not in md
    # Styles removed
    assert ".hidden" not in md
    # Entities decoded
    assert "&" in md
    assert "&amp;" not in md


# -- test_token_estimation --


@pytest.mark.asyncio
async def test_token_estimation(tool: WebTool) -> None:
    """Verify token count is reasonable (word_count * 1.3)."""
    mock_resp = _mock_response()

    async def fake_get(self, url, **kwargs):  # noqa: ANN001, ANN003
        return mock_resp

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await tool.fetch("https://example.com/page")

    assert result["success"] is True
    word_count = len(result["content"].split())
    expected = max(1, int(word_count * 1.3))
    assert result["tokens"] == expected


# -- test_truncation --


@pytest.mark.asyncio
async def test_truncation(tool: WebTool) -> None:
    """Large content gets truncated with marker."""
    small_tool = WebTool(max_tokens=10, timeout=10)  # 10 * 4 = 40 chars limit
    # Build HTML with content well over 40 chars
    long_text = "word " * 200
    long_html = f"<html><body><p>{long_text}</p></body></html>"
    mock_resp = _mock_response(text=long_html)

    async def fake_get(self, url, **kwargs):  # noqa: ANN001, ANN003
        return mock_resp

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await small_tool.fetch("https://example.com/long")

    assert result["success"] is True
    assert result["truncated"] is True
    assert "[... truncated ...]" in result["content"]
